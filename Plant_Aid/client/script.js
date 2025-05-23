// API Configuration
const API_BASE_URL = 'http://localhost:5000';

// Global variables
let currentImagePath = null;
let predictionResults = null;

// DOM elements
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const progressBar = document.getElementById('progressBar');

// API Helper Functions
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log('API Health:', data);
        return data.status === 'healthy';
    } catch (error) {
        console.error('API Health Check Failed:', error);
        showNotification('Backend API is not available. Please start the Flask server.', 'error');
        return false;
    }
}

async function analyzeImageWithAPI(imageFile) {
    try {
        const formData = new FormData();
        formData.append('image', imageFile);

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('API Analysis Error:', error);
        throw error;
    }
}

async function analyzeImageWithBase64(base64Data) {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_data: base64Data
            })
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('API Analysis Error:', error);
        throw error;
    }
}

async function getSupportedClasses() {
    try {
        const response = await fetch(`${API_BASE_URL}/classes`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching classes:', error);
        return null;
    }
}

// File upload handling
fileInput.addEventListener('change', handleFileSelect);

// Enhanced drag and drop functionality
uploadSection.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadSection.classList.add('drag-over');
});

uploadSection.addEventListener('dragleave', (e) => {
    // Only remove drag-over if we're actually leaving the upload section
    if (!uploadSection.contains(e.relatedTarget)) {
        uploadSection.classList.remove('drag-over');
    }
});

uploadSection.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadSection.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showNotification('Please select an image file', 'error');
        return;
    }

    // Check file size (16MB limit as per Flask config)
    const maxSize = 16 * 1024 * 1024; // 16MB
    if (file.size > maxSize) {
        showNotification('File size too large. Please select an image smaller than 16MB.', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        currentImagePath = file; // Store the actual file object
        previewContainer.style.display = 'block';
        analyzeBtn.disabled = false;
        
        // Add animation
        setTimeout(() => {
            imagePreview.style.opacity = '1';
            imagePreview.style.transform = 'scale(1)';
        }, 100);

        showNotification('Image loaded successfully! Click Analyze to detect diseases.', 'success');
    };
    reader.readAsDataURL(file);
}

// Main analysis function with real API integration
async function analyzeImage() {
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const placeholder = document.getElementById('placeholder');
    
    if (!currentImagePath) {
        showNotification('Please select an image first', 'error');
        return;
    }

    // Check API health first
    const isAPIHealthy = await checkAPIHealth();
    if (!isAPIHealthy) {
        showNotification('Cannot connect to backend API. Please ensure Flask server is running on port 5000.', 'error');
        return;
    }

    // Show loading
    placeholder.style.display = 'none';
    results.style.display = 'none';
    loading.style.display = 'block';
    progressBar.classList.add('active');
    analyzeBtn.disabled = true;

    try {
        showNotification('Analyzing image...', 'info');
        
        // Call the real API
        const apiResult = await analyzeImageWithAPI(currentImagePath);
        
        console.log('API Result:', apiResult);
        
        // Store results
        predictionResults = apiResult;
        
        // Hide loading and show results
        loading.style.display = 'none';
        results.style.display = 'block';
        progressBar.classList.remove('active');
        analyzeBtn.disabled = false;

        displayResults(apiResult);
        
        if (apiResult.is_plant) {
            showNotification('Analysis complete!', 'success');
        } else {
            showNotification('No plant material detected in the image', 'warning');
        }

    } catch (error) {
        console.error('Analysis Error:', error);
        
        // Hide loading
        loading.style.display = 'none';
        progressBar.classList.remove('active');
        analyzeBtn.disabled = false;
        
        // Show error message
        showNotification('Analysis failed. Please try again or check if the Flask server is running.', 'error');
        
        // Show placeholder again
        placeholder.style.display = 'block';
        
        // Show detailed error for debugging
        const errorDetails = document.createElement('div');
        errorDetails.style.cssText = `
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 4px solid #dc3545;
        `;
        errorDetails.innerHTML = `
            <h4>❌ Analysis Error</h4>
            <p><strong>Error:</strong> ${error.message}</p>
            <p><strong>Solution:</strong> Make sure the Flask server is running:</p>
            <ul>
                <li>Open terminal and navigate to your project folder</li>
                <li>Run: <code>python app.py</code></li>
                <li>Server should start on http://localhost:5000</li>
            </ul>
        `;
        
        // Insert error details in results section
        results.style.display = 'block';
        results.innerHTML = '';
        results.appendChild(errorDetails);
    }
}

function displayResults(apiResult) {
    const predictionsContainer = document.getElementById('predictions');
    const confidenceChart = document.getElementById('confidenceChart');
    const recommendations = document.getElementById('recommendations');
    const recommendationsList = document.getElementById('recommendationsList');

    // Clear previous results
    predictionsContainer.innerHTML = '';
    confidenceChart.innerHTML = '';
    recommendationsList.innerHTML = '';

    // Check if this is a non-plant image
    const isNonPlant = !apiResult.is_plant;

    // Add analysis info banner
    if (isNonPlant) {
        const banner = document.createElement('div');
        banner.style.cssText = `
            background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        `;
        banner.innerHTML = `
            <h4 style="margin: 0 0 10px 0;">⚠️ Non-Plant Image Detected</h4>
            <p style="margin: 0; opacity: 0.9;">${apiResult.message}</p>
        `;
        predictionsContainer.appendChild(banner);

        // Show recommendations for non-plant images
        if (apiResult.recommendations) {
            apiResult.recommendations.forEach(recommendation => {
                const li = document.createElement('li');
                li.textContent = recommendation;
                recommendationsList.appendChild(li);
            });
            recommendations.style.display = 'block';
            recommendations.style.background = 'linear-gradient(135deg, #ffe4e1, #ffcccb)';
            recommendations.style.borderLeft = '4px solid #ff6b6b';
            const recHeader = recommendations.querySelector('h4');
            if (recHeader) recHeader.style.color = '#a94442';
        }
        return;
    }

    // Plant detected - show analysis results
    const banner = document.createElement('div');
    banner.style.cssText = `
        background: linear-gradient(135deg, #2E8B57, #3CB371);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(46, 139, 87, 0.3);
    `;
    
    const greenContent = apiResult.green_content ? (apiResult.green_content * 100).toFixed(1) : 'N/A';
    const modelConfidence = apiResult.analysis_info?.model_confidence || 'moderate';
    
    banner.innerHTML = `
        <h4 style="margin: 0 0 10px 0;">🌿 Plant Analysis Complete</h4>
        <p style="margin: 0; opacity: 0.9;">
            Green content: ${greenContent}% | Model confidence: ${modelConfidence}
        </p>
        <p style="margin: 5px 0 0 0; font-size: 0.9rem; opacity: 0.8;">
            ${apiResult.analysis_info?.note || 'Analysis using AI model'}
        </p>
    `;
    predictionsContainer.appendChild(banner);

    // Create prediction cards
    const predictions = apiResult.predictions || [];
    const cardsToShow = Math.min(predictions.length, 3);
    
    predictions.slice(0, cardsToShow).forEach((prediction, index) => {
        const card = document.createElement('div');
        card.className = 'prediction-card';
        
        let statusBadge = '';
        let statusClass = '';
        
        if (prediction.type === 'healthy') {
            statusBadge = '✅ Healthy';
            statusClass = 'healthy';
        } else if (prediction.type === 'diseased') {
            statusBadge = '⚠️ Disease';
            statusClass = 'diseased';
        } else {
            statusBadge = '❓ Unknown';
            statusClass = 'invalid';
        }
        
        card.innerHTML = `
            <div class="disease-name">
                <span>${prediction.disease_name || prediction.class_name}</span>
                <span class="status-badge ${statusClass}">
                    ${statusBadge}
                </span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" data-width="${(prediction.confidence * 100).toFixed(1)}"></div>
            </div>
            <div class="confidence-text">
                Confidence: ${(prediction.confidence * 100).toFixed(1)}%
            </div>
            <p style="margin-top: 10px; color: #666; font-size: 0.9rem; line-height: 1.4;">
                ${prediction.description || 'No description available'}
            </p>
            ${prediction.severity ? `
                <div style="margin-top: 8px;">
                    <span style="font-weight: bold; color: #666;">Severity:</span> 
                    <span style="color: ${getSeverityColor(prediction.severity)}; font-weight: bold;">
                        ${prediction.severity.charAt(0).toUpperCase() + prediction.severity.slice(1)}
                    </span>
                </div>
            ` : ''}
        `;

        predictionsContainer.appendChild(card);

        // Animate card appearance
        setTimeout(() => {
            card.classList.add('show');
            const fillBar = card.querySelector('.confidence-fill');
            setTimeout(() => {
                fillBar.style.width = fillBar.dataset.width + '%';
            }, 300);
        }, index * 200);
    });

    // Create confidence chart
    const chartItems = Math.min(predictions.length, 5);
    predictions.slice(0, chartItems).forEach((prediction, index) => {
        const chartItem = document.createElement('div');
        chartItem.className = 'chart-bar';
        
        chartItem.innerHTML = `
            <div class="chart-label">${prediction.disease_name || prediction.class_name}</div>
            <div class="chart-bar-bg">
                <div class="chart-bar-fill" data-width="${(prediction.confidence * 100).toFixed(1)}"></div>
            </div>
            <div class="chart-percentage">${(prediction.confidence * 100).toFixed(1)}%</div>
        `;

        confidenceChart.appendChild(chartItem);

        // Animate chart bars
        setTimeout(() => {
            const fillBar = chartItem.querySelector('.chart-bar-fill');
            fillBar.style.width = fillBar.dataset.width + '%';
        }, index * 150);
    });

    // Show recommendations for the top prediction
    if (predictions.length > 0) {
        // Add treatment recommendation if available
        if (predictions[0].treatment) {
            const treatment = predictions[0].treatment;
            const li = document.createElement('li');
            li.innerHTML = `<strong>Treatment:</strong> ${treatment}`;
            recommendationsList.appendChild(li);
        }
        
        // Add general recommendations based on disease type
        const generalRecs = predictions[0].type === 'diseased' ? 
            [
                'Monitor the plant closely for changes',
                'Isolate affected plants if possible',
                'Maintain proper plant hygiene',
                'Consult with local agricultural extension services'
            ] : [
                'Continue current care routine',
                'Monitor for early disease signs',
                'Ensure proper nutrition and watering',
                'Maintain good garden hygiene'
            ];
        
        generalRecs.forEach(rec => {
            const li = document.createElement('li');
            li.textContent = rec;
            recommendationsList.appendChild(li);
        });
        
        recommendations.style.display = 'block';
    }
}

function getSeverityColor(severity) {
    switch(severity?.toLowerCase()) {
        case 'severe':
            return '#dc3545';
        case 'moderate':
            return '#ffc107';
        case 'mild':
        case 'low':
            return '#28a745';
        case 'none':
            return '#28a745';
        default:
            return '#6c757d';
    }
}

function clearResults() {
    // Reset image preview
    previewContainer.style.display = 'none';
    imagePreview.src = '';
    fileInput.value = '';
    analyzeBtn.disabled = true;
    currentImagePath = null;
    predictionResults = null;

    // Reset results
    document.getElementById('results').style.display = 'none';
    document.getElementById('placeholder').style.display = 'block';
    document.getElementById('loading').style.display = 'none';
    progressBar.classList.remove('active');

    // Reset upload section
    uploadSection.classList.remove('drag-over');
    
    showNotification('Results cleared', 'info');
}

// Smooth scroll to developer information
function scrollToDeveloperInfo(event) {
    event.preventDefault();
    const developerSection = document.getElementById('developer-info');
    if (developerSection) {
        developerSection.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
        
        // Add a subtle highlight effect
        developerSection.style.boxShadow = '0 0 20px rgba(52, 152, 219, 0.3)';
        setTimeout(() => {
            developerSection.style.boxShadow = '';
        }, 2000);
    }
}

// Initialize page
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Plant Disease Detection System Loaded');
    
    // Add some initialization animations
    setTimeout(() => {
        const container = document.querySelector('.container');
        if (container) {
            container.style.transform = 'scale(1)';
        }
    }, 100);
    
    // Initialize tooltips or other interactive elements
    initializeInteractiveElements();
    
    // Check API health on page load
    const isAPIHealthy = await checkAPIHealth();
    if (isAPIHealthy) {
        showNotification('✅ Connected to backend API successfully!', 'success');
        
        // Optionally load and display supported classes
        const classesData = await getSupportedClasses();
        if (classesData) {
            console.log('Supported classes:', classesData.classes);
            console.log('Total classes:', classesData.total_classes);
        }
    } else {
        showNotification('⚠️ Backend API not available. Please start the Flask server.', 'warning');
    }
});

// Initialize interactive elements
function initializeInteractiveElements() {
    // Add hover effects to social links
    const socialLinks = document.querySelectorAll('.social-link');
    socialLinks.forEach(link => {
        link.addEventListener('mouseenter', () => {
            link.style.transform = 'scale(1.1) rotate(5deg)';
        });
        
        link.addEventListener('mouseleave', () => {
            link.style.transform = 'scale(1) rotate(0deg)';
        });
    });
    
    // Add click feedback to buttons
    const buttons = document.querySelectorAll('.btn, .upload-btn');
    buttons.forEach(button => {
        button.addEventListener('click', () => {
            button.style.transform = 'scale(0.95)';
            setTimeout(() => {
                button.style.transform = '';
            }, 150);
        });
    });
}

// Enhanced notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        z-index: 1000;
        animation: slideInRight 0.3s ease;
        max-width: 300px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    `;
    
    // Set background color and icon based on type
    let icon = '';
    switch(type) {
        case 'success':
            notification.style.background = 'linear-gradient(135deg, #28a745, #20c997)';
            icon = '✅ ';
            break;
        case 'error':
            notification.style.background = 'linear-gradient(135deg, #dc3545, #fd7e14)';
            icon = '❌ ';
            break;
        case 'warning':
            notification.style.background = 'linear-gradient(135deg, #ffc107, #fd7e14)';
            icon = '⚠️ ';
            break;
        case 'info':
            notification.style.background = 'linear-gradient(135deg, #007bff, #6f42c1)';
            icon = 'ℹ️ ';
            break;
        default:
            notification.style.background = 'linear-gradient(135deg, #007bff, #6f42c1)';
            icon = 'ℹ️ ';
    }
    
    notification.innerHTML = `${icon}${message}`;
    document.body.appendChild(notification);
    
    // Remove notification after 4 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 4000);
}

// Add CSS animations for notifications dynamically
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .prediction-card.show {
        opacity: 1;
        transform: translateX(0);
    }
    
    /* Additional styles for better error display */
    code {
        background: rgba(0,0,0,0.1);
        padding: 2px 4px;
        border-radius: 3px;
        font-family: monospace;
    }
`;
document.head.appendChild(notificationStyles);