const API_URL = "http://127.0.0.1:8000";  // Change this if running on a different port

// Function to call the backend for Bias Detection
async function detectBias() {
    const inputText = document.getElementById("inputText").value;

    const response = await fetch("http://127.0.0.1:8000/detect-bias/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: inputText }) // Ensure the JSON structure matches the backend model
    });

    const data = await response.json();
    if (response.ok) {
        document.getElementById("output").innerText = data.neutral_text || "Error in processing!";
    } else {
        console.error("Backend Error:", data);
        document.getElementById("output").innerText = "Error processing the request!";
    }
}


// Function to call the backend for Bias Correction
function correctBias() {
    let text = document.getElementById("inputText").value;
    fetch(`${API_URL}/correct-bias/`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ sentence: text }),
    })
    .then(response => response.json())
    .then(data => document.getElementById("output").innerText = data.corrected_sentence)
    .catch(error => console.error("Error:", error));
}

// Function to call the backend for Bias Scoring
function getBiasScore() {
    let text = document.getElementById("inputText").value;
    fetch(`${API_URL}/bias-score/`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ sentence: text }),
    })
    .then(response => response.json())
    .then(data => document.getElementById("output").innerText = "Bias Score: " + data.score)
    .catch(error => console.error("Error:", error));
}
