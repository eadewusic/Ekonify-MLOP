document.addEventListener("DOMContentLoaded", () => {
  // Check if we're on the home page with counters
  const itemsCounter = document.getElementById("items-counter");
  const tonsCounter = document.getElementById("tons-counter");
  const accuracyCounter = document.getElementById("accuracy-counter");

  // Run counter animation if on home page
  if (itemsCounter && tonsCounter && accuracyCounter) {
    animateCounters(1274985, 367, 75);
  }

  // Check if we're on a page with file upload elements
  const fileInput = document.getElementById("file-upload");
  const dropZone = document.getElementById("drop-zone");

  // Only run file upload code if these elements exist
  if (fileInput && dropZone) {
    const form = document.getElementById("upload-form");
    const imagePreview = document.getElementById("image-preview");
    const retrainBtn = document.querySelector(".retrain-btn");

    // Check if we're on the retrain page
    const isRetrainPage = document.querySelector(".retrain-container") !== null;

    // Fix file input accepting attribute for retrain page
    if (isRetrainPage && fileInput) {
      fileInput.setAttribute("accept", ".zip");
    }

    // Prevent form from triggering file input
    if (form) {
      form.addEventListener("click", (e) => {
        // Stop click from propagating to form if it's coming from the drop zone
        if (e.target === dropZone || dropZone.contains(e.target)) {
          e.stopPropagation();
        }
      });
    }

    // Drag and drop functionality
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

        if (isRetrainPage) {
          handleRetrainFile(files[0]);
        } else if (imagePreview) {
          previewImage(); // Show the preview when dropped
        }
      }
    });

    // Handle file selection
    fileInput.addEventListener("change", function () {
      const file = this.files[0];

      if (isRetrainPage) {
        handleRetrainFile(file);
      } else if (imagePreview) {
        previewImage();
      }
    });

    // Handle retrain file selection
    function handleRetrainFile(file) {
      if (!file) return;

      // Check if it's a ZIP file
      if (
        file.type === "application/zip" ||
        file.name.toLowerCase().endsWith(".zip")
      ) {
        // Update the drop zone text
        const uploadText = dropZone.querySelector(".retrain-upload-text");
        if (uploadText) {
          uploadText.innerHTML = `Selected file: <strong>${file.name}</strong><br>Click "Start Retraining" to begin`;
        }

        // Update button styling if it exists
        if (retrainBtn) {
          retrainBtn.style.backgroundColor = "#2e7d32";
          retrainBtn.style.opacity = "1";
        }
      } else {
        alert("Please select a ZIP file");
        resetFileInput();
      }
    }

    function resetFileInput() {
      fileInput.value = "";
      const uploadText = dropZone.querySelector(".retrain-upload-text");
      if (uploadText) {
        uploadText.innerHTML =
          'Drag and drop your ZIP file here, or <span class="retrain-upload-link">browse</span>';
      }
    }

    // Preview the uploaded image (for prediction page)
    function previewImage() {
      if (!imagePreview) return;

      const file = fileInput.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
          imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview" style="max-width: 100%; height: auto;">`;
        };
        reader.readAsDataURL(file);
      }
    }

    // Form submission handler
    if (form) {
      form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const file = fileInput.files[0];
        if (!file) {
          alert("Please select a file first.");
          return;
        }

        // Create FormData object
        const formData = new FormData();
        formData.append("file", file);

        // Determine which endpoint to use
        let endpoint = "/predict";
        if (isRetrainPage) {
          endpoint = "/retrain";
          if (retrainBtn) {
            retrainBtn.textContent = "Uploading...";
            retrainBtn.disabled = true;
          }
        }

        try {
          const response = await fetch(endpoint, {
            method: "POST",
            body: formData,
          });

          const result = await response.json();

          if (isRetrainPage) {
            // Handle retrain response
            if (retrainBtn) {
              retrainBtn.textContent = "Start Retraining";
              retrainBtn.disabled = false;
            }

            if (response.ok) {
              alert("Model retraining started successfully!");
              resetFileInput();
            } else {
              alert(`Error: ${result.message || "Something went wrong."}`);
            }
          } else {
            // Handle predict response
            displayPrediction(result);
          }
        } catch (error) {
          console.error(
            `${isRetrainPage ? "Retraining" : "Prediction"} failed:`,
            error
          );
          alert(
            `An error occurred while ${
              isRetrainPage ? "retraining" : "predicting"
            }.`
          );

          if (isRetrainPage && retrainBtn) {
            retrainBtn.textContent = "Start Retraining";
            retrainBtn.disabled = false;
          }
        }
      });
    }

    // Display the predicted class and confidence after the "Predict" button is clicked
    function displayPrediction(result) {
      // Check if we're on prediction page
      const predictionResult = document.getElementById("prediction-result");
      if (!predictionResult) return;

      // Show the prediction result
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
  }

  // Counter animation function
  function animateCounters(itemsTarget, tonsTarget, accuracyTarget) {
    // Define animation duration
    const duration = 2000; // 2 seconds
    const frameDuration = 1000 / 60; // 60fps
    const totalFrames = Math.round(duration / frameDuration);

    // Animate items counter
    animateValue(
      itemsCounter,
      0,
      itemsTarget,
      totalFrames,
      frameDuration,
      numberWithCommas
    );

    // Animate tons counter
    animateValue(
      tonsCounter,
      0,
      tonsTarget,
      totalFrames,
      frameDuration,
      numberWithCommas
    );

    // Animate accuracy counter
    animateValue(
      accuracyCounter,
      0,
      accuracyTarget,
      totalFrames,
      frameDuration
    );
  }

  // Helper function for counter animation
  function animateValue(
    element,
    start,
    end,
    totalFrames,
    frameDuration,
    formatter = (val) => val
  ) {
    if (!element) return;

    let frame = 0;
    const range = end - start;

    const counter = setInterval(() => {
      frame++;
      const progress = frame / totalFrames;
      const currentValue = Math.round(start + progress * range);

      element.textContent = formatter(currentValue);

      if (frame === totalFrames) {
        clearInterval(counter);
      }
    }, frameDuration);
  }

  // Format numbers with commas
  function numberWithCommas(x) {
    return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }
});
