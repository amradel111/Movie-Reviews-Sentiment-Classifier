/**
 * IMDb Sentiment Analyzer - Client-side JavaScript
 */

// DOM elements
const reviewForm = document.getElementById('review-form');
const reviewText = document.getElementById('review-text');
const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn = document.getElementById('clear-btn');
const resultsCard = document.getElementById('results-card');
const resultsHeader = document.getElementById('results-header');
const sentimentIcon = document.getElementById('sentiment-icon');
const sentimentResult = document.getElementById('sentiment-result');
const confidenceBar = document.getElementById('confidence-bar');
const processedText = document.getElementById('processed-text');
const loadingSpinner = document.getElementById('loading-spinner');

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    reviewForm.addEventListener('submit', handleSubmit);
    clearBtn.addEventListener('click', clearForm);
});

/**
 * Handle form submission
 * @param {Event} event - The submit event
 */
function handleSubmit(event) {
    event.preventDefault();
    
    const review = reviewText.value.trim();
    
    if (!review) {
        showError('Please enter a movie review.');
        return;
    }
    
    // Show loading spinner
    loadingSpinner.style.display = 'block';
    resultsCard.style.display = 'none';
    
    // Disable the button during prediction
    analyzeBtn.disabled = true;
    
    // Send request to server
    analyzeSentiment(review);
}

/**
 * Send the review to the backend for sentiment analysis
 * @param {string} review - The review text
 */
function analyzeSentiment(review) {
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ review: review })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        displayResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
        showError('An error occurred while analyzing the sentiment. Please try again.');
    })
    .finally(() => {
        // Hide loading spinner and re-enable button
        loadingSpinner.style.display = 'none';
        analyzeBtn.disabled = false;
    });
}

/**
 * Display the sentiment analysis results
 * @param {Object} data - The sentiment analysis results
 */
function displayResults(data) {
    if (data.error) {
        showError(data.error);
        return;
    }
    
    // Remove existing classes
    resultsCard.className = 'card shadow-sm';
    
    // Update sentiment result text and styling
    const sentiment = data.sentiment; // "Positive" or "Negative"
    
    // Update UI based on sentiment
    if (sentiment === 'Positive') {
        sentimentIcon.innerHTML = '<i class="fas fa-smile text-success"></i>';
        resultsCard.classList.add('positive');
    } else if (sentiment === 'Negative') {
        sentimentIcon.innerHTML = '<i class="fas fa-frown text-danger"></i>';
        resultsCard.classList.add('negative');
    } else {
        sentimentIcon.innerHTML = '<i class="fas fa-meh text-warning"></i>';
    }
    
    // Update result text
    sentimentResult.textContent = sentiment;
    
    // Update confidence bar
    const confidencePercent = Math.round(data.confidence * 100);
    confidenceBar.style.width = `${confidencePercent}%`;
    confidenceBar.setAttribute('aria-valuenow', confidencePercent);
    confidenceBar.textContent = `${confidencePercent}%`;
    
    // Set confidence bar class based on confidence level
    confidenceBar.className = 'progress-bar';
    if (confidencePercent >= 75) {
        confidenceBar.classList.add('bg-success');
    } else if (confidencePercent >= 50) {
        confidenceBar.classList.add('bg-primary');
    } else {
        confidenceBar.classList.add('bg-warning');
    }
    
    // Update processed text display
    processedText.textContent = data.processed_text;
    
    // Show results with animation
    resultsCard.style.display = 'block';
    resultsCard.classList.add('fade-in');
    
    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Display an error message
 * @param {string} message - The error message
 */
function showError(message) {
    // Create a Bootstrap alert
    const alertHtml = `
        <div class="alert alert-danger alert-dismissible fade show mt-3" role="alert">
            <i class="fas fa-exclamation-circle me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    
    // Insert the alert before the form
    reviewForm.insertAdjacentHTML('beforebegin', alertHtml);
    
    // Remove the alert after 5 seconds
    setTimeout(() => {
        const alert = document.querySelector('.alert');
        if (alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }
    }, 5000);
}

/**
 * Clear the form and hide results
 */
function clearForm() {
    reviewText.value = '';
    resultsCard.style.display = 'none';
    // Set focus back to the text area
    reviewText.focus();
} 