document.addEventListener("DOMContentLoaded", () => {

  // Image Upload and Preview
 
  const fileInput = document.getElementById("file-upload");
  const dropZone = document.getElementById("drop-zone");
  const form = document.getElementById("upload-form");
  const imagePreview = document.getElementById("image-preview");

  // Prevent form from triggering file input
  form.addEventListener("click", (e) => {
    // Stop click from propagating to form if it's coming from the drop zone
    if (e.target === dropZone || dropZone.contains(e.target)) {
      e.stopPropagation();
    }
  });

  // Only trigger file input when the drop zone is clicked
  dropZone.addEventListener("click", (e) => {
    e.preventDefault(); // Prevent any default behavior
    fileInput.click();
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      fileInput.files = files;
      previewImage(); // Show the preview when dropped
    }
  });

  // Display image preview when file is selected
  fileInput.addEventListener("change", previewImage);

  // Preview the uploaded image
  function previewImage() {
    const file = fileInput.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = function (e) {
        imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview" style="max-width: 100%; height: auto;">`;
      };
      reader.readAsDataURL(file);
    }
  }

  // Prediction Logic

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) {
      alert("Please select an image first.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      displayPrediction(result);
    } catch (error) {
      console.error("Prediction failed:", error);
      alert("An error occurred while predicting.");
    }
  });

  // Display the predicted class and confidence after the "Predict" button is clicked
  function displayPrediction(result) {
    // Show the prediction result
    const predictionResult = document.getElementById("prediction-result");
    predictionResult.style.display = "block"; // Make it visible after prediction

    // Display class and confidence in respective spans
    const predictedClass = document
      .getElementById("predicted-class")
      .querySelector("span");
    const confidence = document
      .getElementById("confidence")
      .querySelector("span");

    predictedClass.textContent = result.class || "Unknown"; // Replace with model's actual class field
    confidence.textContent = result.confidence
      ? `${(result.confidence * 100).toFixed(2)}%`
      : "N/A"; // Assuming confidence is in decimal form
  }
});
