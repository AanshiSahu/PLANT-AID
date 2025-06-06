import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# Plant disease class names (38 classes)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot', 
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def create_model(num_classes=38):
    """Create the plant disease detection model"""
    # Load MobileNetV2 as base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification layers
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', name='dense_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(256, activation='relu', name='dense_2')(x)
    x = Dropout(0.3, name='dropout_2')(x)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs)
    
    return model, base_model

def setup_data_generators(train_dir, val_dir, test_dir=None):
    """Setup data generators with augmentation"""
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    # Validation and test data generator (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        classes=CLASS_NAMES
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        classes=CLASS_NAMES
    )
    
    test_generator = None
    if test_dir and os.path.exists(test_dir):
        test_generator = val_datagen.flow_from_directory(
            test_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False,
            classes=CLASS_NAMES
        )
    
    return train_generator, val_generator, test_generator

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training & validation accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Plot training & validation loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Plot learning rate
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_generator, class_names):
    """Evaluate model and generate classification report"""
    if test_generator is None:
        print("No test data available for evaluation")
        return
    
    # Make predictions
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_generator.classes
    
    # Generate classification report
    report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=class_names,
        output_dict=True
    )
    
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return report

def train_model(train_dir, val_dir, test_dir=None, save_path='plant_disease_model.h5'):
    """Main training function"""
    
    print("Setting up data generators...")
    train_gen, val_gen, test_gen = setup_data_generators(train_dir, val_dir, test_dir)
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    if test_gen:
        print(f"Test samples: {test_gen.samples}")
    
    print("\nCreating model...")
    model, base_model = create_model(num_classes=len(CLASS_NAMES))
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Calculate steps
    steps_per_epoch = train_gen.samples // BATCH_SIZE
    validation_steps = val_gen.samples // BATCH_SIZE
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    # Train the model (first phase - frozen base)
    print("\n=== Phase 1: Training with frozen base model ===")
    history1 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS//2,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning (second phase - unfreeze base model)
    print("\n=== Phase 2: Fine-tuning with unfrozen base model ===")
    base_model.trainable = True
    
    # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS//2,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
        initial_epoch=len(history1.history['loss'])
    )
    
    # Combine histories
    combined_history = {}
    for key in history1.history.keys():
        combined_history[key] = history1.history[key] + history2.history[key]
    
    # Create a mock history object
    class CombinedHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    combined_hist = CombinedHistory(combined_history)
    
    # Plot training history
    plot_training_history(combined_hist)
    
    # Evaluate on test set if available
    if test_gen:
        print("\n=== Model Evaluation ===")
        test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Generate detailed evaluation report
        evaluate_model(model, test_gen, CLASS_NAMES)
    
    print(f"\nTraining completed! Model saved to {save_path}")
    return model, combined_hist

def create_sample_prediction_script():
    """Create a sample script to test the trained model"""
    script_content = '''
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Class names
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def predict_disease(image_path):
    """Predict disease from image path"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    print(f"Predicted: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    
    # Show top 5 predictions
    top_indices = np.argsort(predictions[0])[::-1][:5]
    print("\\nTop 5 predictions:")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {CLASS_NAMES[idx]}: {predictions[0][idx]:.4f}")

# Example usage
if __name__ == "__main__":
    predict_disease("path_to_your_image.jpg")
'''
    
    with open('test_model.py', 'w') as f:
        f.write(script_content)
    print("Sample prediction script created: test_model.py")

if __name__ == "__main__":
    # Example usage
    print("Plant Disease Detection Model Trainer")
    print("=====================================")
    
    # Define your data directories
    # You need to download the PlantVillage dataset and organize it properly
    TRAIN_DIR = "data/train"  # Path to training data
    VAL_DIR = "data/val"      # Path to validation data  
    TEST_DIR = "data/test"    # Path to test data (optional)
    
    # Check if data directories exist
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training directory '{TRAIN_DIR}' not found!")
        print("\nTo use this script, you need to:")
        print("1. Download the PlantVillage dataset")
        print("2. Organize it into train/val/test folders")
        print("3. Update the directory paths above")
        print("\nDataset structure should be:")
        print("data/")
        print("├── train/")
        print("│   ├── Apple___Apple_scab/")
        print("│   ├── Apple___Black_rot/")
        print("│   └── ...")
        print("├── val/")
        print("│   ├── Apple___Apple_scab/")
        print("│   └── ...")
        print("└── test/ (optional)")
        print("    ├── Apple___Apple_scab/")
        print("    └── ...")
        exit(1)
    
    # Start training
    try:
        model, history = train_model(
            train_dir=TRAIN_DIR,
            val_dir=VAL_DIR,
            test_dir=TEST_DIR if os.path.exists(TEST_DIR) else None,
            save_path='plant_disease_model.h5'
        )
        
        # Create sample prediction script
        create_sample_prediction_script()
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("Files created:")
        print("- plant_disease_model.h5 (trained model)")
        print("- training_history.png (training plots)")
        print("- confusion_matrix.png (evaluation)")
        print("- test_model.py (sample prediction script)")
        print("="*50)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("\nPlease check your data directories and try again.")