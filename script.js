// scripts.js

// Example coefficients and intercept (replace with actual model values)
const coefficients = [0.25, 0.3, -0.1, 0.02, 0.05, 0.18, 0.2, 0.1];
const intercept = 0.5;

// Function to normalize input data using StandardScaler
function normalize(input) {
    const mean = [3.8, 120, 69, 20, 80, 32, 0.5, 33]; // Replace with actual means
    const stdDev = [3.5, 31, 19, 16, 115, 7, 0.3, 11]; // Replace with actual std deviations
    return input.map((val, idx) => (val - mean[idx]) / stdDev[idx]);
}

// Logistic regression prediction function
function predictOutcome(features) {
    // Normalize the input features
    const normalizedFeatures = normalize(features);

    // Calculate the linear combination (log-odds)
    let logOdds = intercept;
    for (let i = 0; i < normalizedFeatures.length; i++) {
        logOdds += coefficients[i] * normalizedFeatures[i];
    }

    // Convert log-odds to probability
    const probability = 1 / (1 + Math.exp(-logOdds));
    return probability >= 0.5 ? "Diabetic" : "Non-Diabetic";
}

// Event listener for prediction
document.getElementById("predict-button").addEventListener("click", () => {
    const form = document.getElementById("prediction-form");
    const featureValues = Array.from(form.elements)
        .filter((el) => el.tagName === "INPUT")
        .map((el) => parseFloat(el.value));

    if (featureValues.some(isNaN)) {
        alert("Please fill in all fields with valid numbers.");
        return;
    }

    const result = predictOutcome(featureValues);

    document.getElementById("result").innerHTML = `<strong>Prediction:</strong> ${result}`;
});