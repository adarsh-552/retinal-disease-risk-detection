// script.js
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const resultDiv = document.getElementById("result");
const errorDiv = document.getElementById("error");
const predictBtn = document.getElementById("predictBtn");

// Add event listener to the predict button
predictBtn?.addEventListener("click", uploadImage);

// Show image preview
imageInput?.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
    resultDiv.style.display = "none";
    errorDiv.style.display = "none";
  }
});

// Upload and predict
function uploadImage() {
  const file = imageInput?.files[0];
  if (!file) {
    showError("Please select a retinal image first.");
    return;
  }

  const allowedTypes = ["image/jpeg", "image/png", "image/jpg"];
  if (!allowedTypes.includes(file.type)) {
    showError("Please upload a JPG or PNG image.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  // UI Feedback
  predictBtn.disabled = true;
  predictBtn.textContent = "Analyzing...";
  errorDiv.style.display = "none";

  fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    body: formData,
  })
    .then((res) => {
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      return res.json();
    })
    .then((data) => {
      if (!data.prediction || typeof data.confidence !== "number") {
        throw new Error("Invalid response from server");
      }

      const pred = data.prediction;
      const conf = (data.confidence * 100).toFixed(1);

      document.getElementById("prediction-text").textContent = pred;
      document.getElementById("confidenceLabel").textContent = `${conf}% confidence`;
      document.getElementById("confidenceFill").style.width = `${conf}%`;

      // Update status badge
      const statusBadge = document.getElementById("statusBadge");
      if (pred.includes("Normal") || pred.includes("Low")) {
        statusBadge.textContent = "Normal";
        statusBadge.className = "status-badge status-normal";
      } else if (pred.includes("Abnormal") || pred.includes("High")) {
        statusBadge.textContent = "Abnormal";
        statusBadge.className = "status-badge status-abnormal";
      } else {
        statusBadge.textContent = "Uncertain";
        statusBadge.className = "status-badge status-uncertain";
      }

      // Color logic
      const isHealthy = pred.toLowerCase().includes("normal") || pred.toLowerCase().includes("low");
      resultDiv.style.backgroundColor = isHealthy ? "#e8f5e9" : "#ffebee";
      document.getElementById("prediction-text").style.color = isHealthy
        ? "var(--success)"
        : "var(--danger)";

      resultDiv.style.display = "block";
    })
    .catch((err) => {
      console.error("Prediction error:", err);
      showError("Analysis failed. Ensure the backend is running at http://127.0.0.1:8000");
    })
    .finally(() => {
      predictBtn.disabled = false;
      predictBtn.textContent = "Analyze Retina";
    });
}

function showError(message) {
  errorDiv.textContent = message;
  errorDiv.style.display = "block";
}


