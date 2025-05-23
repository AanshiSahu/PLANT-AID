# -*- coding: utf-8 -*-
"""
Plant Disease Detection API - Lightweight Version
A Flask-based REST API for plant disease detection without TensorFlow
Uses scikit-learn and basic image processing for demonstration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import base64
import cv2
from werkzeug.utils import secure_filename
import os
import logging
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Plant disease class names
CLASS_NAMES = [
    'Apple Scab',
    'Apple Black Rot',
    'Apple Cedar Rust',
    'Apple Healthy',
    'Cherry Powdery Mildew',
    'Cherry Healthy',
    'Corn Gray Leaf Spot',
    'Corn Common Rust',
    'Corn Northern Leaf Blight',
    'Corn Healthy',
    'Grape Black Rot',
    'Grape Esca',
    'Grape Leaf Blight',
    'Grape Healthy',
    'Peach Bacterial Spot',
    'Peach Healthy',
    'Pepper Bacterial Spot',
    'Pepper Healthy',
    'Potato Early Blight',
    'Potato Late Blight',
    'Potato Healthy',
    'Strawberry Leaf Scorch',
    'Strawberry Healthy',
    'Tomato Bacterial Spot',
    'Tomato Early Blight',
    'Tomato Late Blight',
    'Tomato Leaf Mold',
    'Tomato Septoria Leaf Spot',
    'Tomato Spider Mites',
    'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus',
    'Tomato Mosaic Virus',
    'Tomato Healthy'
]

# Disease information database
DISEASE_INFO = {
    'Apple Scab': {
        'type': 'diseased',
        'severity': 'moderate',
        'description': 'Fungal disease causing dark spots on leaves and fruit',
        'treatment': 'Apply fungicide, improve air circulation, remove infected leaves'
    },
    'Apple Black Rot': {
        'type': 'diseased',
        'severity': 'severe',
        'description': 'Fungal disease causing black rot on fruit and cankers on branches',
        'treatment': 'Prune infected areas, apply copper-based fungicide'
    },
    'Apple Cedar Rust': {
        'type': 'diseased',
        'severity': 'moderate',
        'description': 'Fungal disease causing orange spots on apple leaves',
        'treatment': 'Apply preventive fungicide, remove nearby cedar trees'
    },
    'Apple Healthy': {
        'type': 'healthy',
        'severity': 'none',
        'description': 'Apple plant shows no signs of disease',
        'treatment': 'Continue regular care and monitoring'
    },
    'Tomato Late Blight': {
        'type': 'diseased',
        'severity': 'severe',
        'description': 'Devastating disease that can destroy entire tomato crops',
        'treatment': 'Remove infected plants, apply copper fungicide, improve ventilation'
    },
    'Tomato Early Blight': {
        'type': 'diseased',
        'severity': 'moderate',
        'description': 'Common fungal disease causing dark spots with concentric rings',
        'treatment': 'Apply fungicide, remove affected leaves, improve air circulation'
    },
    'Tomato Healthy': {
        'type': 'healthy',
        'severity': 'none',
        'description': 'Tomato plant shows no signs of disease',
        'treatment': 'Maintain proper watering and nutrition'
    },
    'Corn Common Rust': {
        'type': 'diseased',
        'severity': 'moderate',
        'description': 'Fungal disease causing rust-colored pustules on corn leaves',
        'treatment': 'Plant resistant varieties, apply fungicide if severe'
    },
    'Corn Healthy': {
        'type': 'healthy',
        'severity': 'none',
        'description': 'Corn plant shows no signs of disease',
        'treatment': 'Continue regular care and monitoring'
    },
    'Potato Late Blight': {
        'type': 'diseased',
        'severity': 'severe',
        'description': 'Serious disease that can destroy potato crops quickly',
        'treatment': 'Apply copper fungicide, improve drainage, remove infected plants'
    },
    'Potato Early Blight': {
        'type': 'diseased',
        'severity': 'moderate',
        'description': 'Common potato disease causing dark lesions on leaves',
        'treatment': 'Apply fungicide, rotate crops, improve air circulation'
    },
    'Potato Healthy': {
        'type': 'healthy',
        'severity': 'none',
        'description': 'Potato plant shows no signs of disease',
        'treatment': 'Maintain proper growing conditions'
    }
}

class LightweightPlantAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.loaded = False
        self.create_model()
        
    def create_model(self):
        """Create a lightweight model using scikit-learn"""
        try:
            # Create a simple Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.scaler = StandardScaler()
            
            # Create some dummy training data for demonstration
            # In a real scenario, you would load actual training data
            self.train_dummy_model()
            self.loaded = True
            logger.info("✅ Lightweight model created successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            return False
    
    def train_dummy_model(self):
        """Train the model with dummy data for demonstration"""
        # Generate dummy features for each class
        n_samples_per_class = 10
        n_features = 50  # Number of image features we'll extract
        
        X = []
        y = []
        
        for i, class_name in enumerate(CLASS_NAMES):
            for _ in range(n_samples_per_class):
                # Generate random features that somewhat simulate real image features
                features = np.random.randn(n_features)
                
                # Add some class-specific patterns
                if 'healthy' in class_name.lower():
                    features[0:5] += np.random.normal(2, 0.5, 5)  # Higher "green" features
                else:
                    features[5:10] += np.random.normal(2, 0.5, 5)  # Higher "disease" features
                
                X.append(features)
                y.append(i)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
    
    def extract_image_features(self, image):
        """Extract features from image for classification"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to standard size
            image = image.resize((224, 224))
            img_array = np.array(image)
            
            features = []
            
            # Color features
            mean_rgb = np.mean(img_array, axis=(0, 1))
            std_rgb = np.std(img_array, axis=(0, 1))
            features.extend(mean_rgb)
            features.extend(std_rgb)
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            mean_hsv = np.mean(hsv, axis=(0, 1))
            std_hsv = np.std(hsv, axis=(0, 1))
            features.extend(mean_hsv)
            features.extend(std_hsv)
            
            # Texture features using edge detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # Histogram features
            hist_r = np.histogram(img_array[:,:,0], bins=8, range=(0, 256))[0]
            hist_g = np.histogram(img_array[:,:,1], bins=8, range=(0, 256))[0]
            hist_b = np.histogram(img_array[:,:,2], bins=8, range=(0, 256))[0]
            
            hist_r = hist_r / np.sum(hist_r)
            hist_g = hist_g / np.sum(hist_g)
            hist_b = hist_b / np.sum(hist_b)
            
            features.extend(hist_r)
            features.extend(hist_g)
            features.extend(hist_b)
            
            # Green pixel ratio (important for plant detection)
            green_mask = (img_array[:,:,1] > img_array[:,:,0]) & (img_array[:,:,1] > img_array[:,:,2])
            green_ratio = np.sum(green_mask) / (img_array.shape[0] * img_array.shape[1])
            features.append(green_ratio)
            
            # Brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            features.extend([brightness, contrast])
            
            # Pad or truncate to exactly 50 features
            features = features[:50]
            while len(features) < 50:
                features.append(0.0)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None
    
    def predict(self, image):
        """Make prediction on image"""
        if not self.loaded or self.model is None:
            return None
            
        try:
            # Extract features
            features = self.extract_image_features(image)
            if features is None:
                return None
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get top 5 predictions
            top_indices = np.argsort(probabilities)[::-1][:5]
            
            results = []
            for idx in top_indices:
                class_name = CLASS_NAMES[idx]
                confidence = float(probabilities[idx])
                
                # Get disease info
                disease_info = DISEASE_INFO.get(class_name, {
                    'type': 'healthy' if 'healthy' in class_name.lower() else 'diseased',
                    'severity': 'unknown',
                    'description': 'Disease information not available',
                    'treatment': 'Consult agricultural expert'
                })
                
                results.append({
                    'class_name': class_name,
                    'disease_name': class_name,
                    'confidence': confidence,
                    'type': disease_info['type'],
                    'severity': disease_info['severity'],
                    'description': disease_info['description'],
                    'treatment': disease_info['treatment']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None

# Initialize analyzer
plant_analyzer = LightweightPlantAnalyzer()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_plant_in_image(image):
    """Simple plant detection using color analysis"""
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        if len(img_array.shape) != 3:
            return False
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define range for green colors (plants)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green pixels
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate percentage of green pixels
        green_pixels = cv2.countNonZero(mask)
        total_pixels = img_array.shape[0] * img_array.shape[1]
        green_ratio = green_pixels / total_pixels
        
        # Consider it a plant if more than 8% green pixels
        return green_ratio > 0.08
        
    except Exception as e:
        logger.error(f"Error in plant detection: {str(e)}")
        return True  # Default to True if detection fails

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'Plant Disease Detection API is running (Lightweight Version)',
        'model_loaded': plant_analyzer.loaded,
        'version': '1.0-lite',
        'message': 'Welcome to Plant Disease Detection API - No TensorFlow Required!'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if model is loaded
        if not plant_analyzer.loaded:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please wait for model to initialize'
            }), 503
        
        # Check if image is provided
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({
                'error': 'No image provided',
                'message': 'Please provide an image file or base64 image data'
            }), 400
        
        image = None
        
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if file and allowed_file(file.filename):
                try:
                    image = Image.open(file.stream)
                except Exception as e:
                    return jsonify({
                        'error': 'Invalid image file',
                        'message': str(e)
                    }), 400
        
        # Handle base64 image data
        elif 'image_data' in request.json:
            try:
                image_data = request.json['image_data']
                # Remove data URL prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                # Decode base64
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                return jsonify({
                    'error': 'Invalid base64 image data',
                    'message': str(e)
                }), 400
        
        if image is None:
            return jsonify({'error': 'Could not process image'}), 400
        
        # Check if image contains plant material
        is_plant = detect_plant_in_image(image)
        
        if not is_plant:
            return jsonify({
                'is_plant': False,
                'message': 'No plant material detected in the image',
                'recommendations': [
                    'Please upload an image containing plant material',
                    'Ensure the plant is clearly visible in the image',
                    'Take a photo of leaves, stems, or fruits',
                    'Avoid images with only backgrounds or objects'
                ]
            })
        
        # Make prediction
        predictions = plant_analyzer.predict(image)
        
        if predictions is None:
            return jsonify({
                'error': 'Prediction failed',
                'message': 'Could not analyze the image'
            }), 500
        
        # Calculate green content for additional info
        img_array = np.array(image)
        green_pixels = np.sum((img_array[:, :, 1] > img_array[:, :, 0]) & 
                             (img_array[:, :, 1] > img_array[:, :, 2]))
        total_pixels = img_array.shape[0] * img_array.shape[1]
        green_ratio = green_pixels / total_pixels
        
        return jsonify({
            'is_plant': True,
            'green_content': float(green_ratio),
            'predictions': predictions,
            'top_prediction': predictions[0] if predictions else None,
            'analysis_info': {
                'model_type': 'lightweight_rf',
                'model_confidence': 'high' if predictions[0]['confidence'] > 0.7 else 'moderate' if predictions[0]['confidence'] > 0.5 else 'low',
                'image_quality': 'good' if green_ratio > 0.2 else 'fair',
                'note': 'Using scikit-learn based classifier'
            }
        })
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of supported plant disease classes"""
    return jsonify({
        'classes': CLASS_NAMES,
        'total_classes': len(CLASS_NAMES),
        'disease_info': DISEASE_INFO,
        'model_type': 'lightweight'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': plant_analyzer.loaded,
        'model_type': 'scikit-learn RandomForest',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': '16MB',
        'tensorflow_required': False,
        'dependencies': ['flask', 'pillow', 'opencv-python', 'scikit-learn', 'numpy']
    })

if __name__ == '__main__':
    # Initialize analyzer
    logger.info("Starting Plant Disease Detection API (Lightweight Version)...")
    logger.info("=" * 60)
    
    if plant_analyzer.loaded:
        logger.info("✅ Lightweight model loaded successfully!")
        logger.info("🔬 Using: scikit-learn RandomForest classifier")
        logger.info("🚀 No TensorFlow required!")
    else:
        logger.warning("⚠️  Model loading failed!")
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Flask server...")
    logger.info("📡 API will be available at: http://localhost:5000")
    logger.info("🌐 Frontend should connect to: http://localhost:5000")
    logger.info("💡 This version works without TensorFlow!")
    logger.info("=" * 60)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)