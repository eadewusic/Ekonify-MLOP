document.addEventListener("DOMContentLoaded", () => {
  // Check if we're on the home page with counters
  const homeItemsCounter = document.getElementById("items-counter");
  const homeTonsCounter = document.getElementById("tons-counter");
  const homeAccuracyCounter = document.getElementById("accuracy-counter");

  // Run counter animation if on home page
  if (homeItemsCounter && homeTonsCounter && homeAccuracyCounter) {
    animateCounters(
      1274985,
      367,
      75,
      homeItemsCounter,
      homeTonsCounter,
      homeAccuracyCounter
    );
  }

  // Check if we're on the visualization page
  const visualizationPage = document.querySelector(".visualize-container");
  if (visualizationPage) {
    // Get all stat elements from the visualization page
    const statElements = visualizationPage.querySelectorAll(".stat strong");

    if (statElements.length === 3) {
      // Extract the target values from the existing content
      const accuracyValue = parseInt(
        statElements[0].textContent.replace(/\D/g, "")
      );
      const predictionsValue = parseInt(
        statElements[1].textContent.replace(/[^\d]/g, "")
      );
      const tonsValue = parseInt(
        statElements[2].textContent.replace(/\D/g, "")
      );

      // Replace the content with zero to start animation
      statElements.forEach((el) => (el.textContent = "0"));

      // Animate the counters
      animateValue(
        statElements[0],
        0,
        accuracyValue,
        120,
        1000 / 60,
        (val) => `${val}%`
      );
      animateValue(
        statElements[1],
        0,
        predictionsValue,
        120,
        1000 / 60,
        numberWithCommas
      );
      animateValue(
        statElements[2],
        0,
        tonsValue,
        120,
        1000 / 60,
        numberWithCommas
      );
    }
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

  // RETRAIN TABS FUNCTIONALITY - Only execute if we're on the retrain page with tabs
  const retrainTabs = document.querySelectorAll(".retrain-tab");

  if (retrainTabs.length > 0) {
    const tabContents = document.querySelectorAll(".retrain-tab-content");

    // Tab switching functionality
    retrainTabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const tabId = tab.getAttribute("data-tab");

        // Remove active class from all tabs and contents
        retrainTabs.forEach((t) => t.classList.remove("active"));
        tabContents.forEach((c) => c.classList.remove("active"));

        // Add active class to clicked tab and corresponding content
        tab.classList.add("active");
        document.getElementById(`${tabId}-tab`).classList.add("active");

        // Hide the progress container when switching tabs
        const progressContainer = document.getElementById("progress-container");
        if (progressContainer) {
          progressContainer.style.display = "none";
        }
      });
    });

    // Set up the Continue Training tab
    setupContinueTrainingTab();

    // Set up the Use Existing Datasets tab
    setupExistingDatasetsTab();

    // Set up the progress and results functionality
    setupProgressAndResults();

    // Handle the original upload form submission
    setupOriginalUploadForm();
  }

  // Counter animation function for homepage
  function animateCounters(
    itemsTarget,
    tonsTarget,
    accuracyTarget,
    itemsEl,
    tonsEl,
    accuracyEl
  ) {
    // Define animation duration
    const duration = 2000; // 2 seconds
    const frameDuration = 1000 / 60; // 60fps
    const totalFrames = Math.round(duration / frameDuration);

    // Animate items counter
    animateValue(
      itemsEl,
      0,
      itemsTarget,
      totalFrames,
      frameDuration,
      numberWithCommas
    );

    // Animate tons counter
    animateValue(
      tonsEl,
      0,
      tonsTarget,
      totalFrames,
      frameDuration,
      numberWithCommas
    );

    // Animate accuracy counter
    animateValue(accuracyEl, 0, accuracyTarget, totalFrames, frameDuration);
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

// ===== RETRAIN FUNCTIONALITY ====

/**
 * Sets up the Continue Training tab functionality
 */
function setupContinueTrainingTab() {
  const continueModelSelect = document.getElementById("continue-model-select");
  const modelInfoPlaceholder = document.getElementById(
    "model-info-placeholder"
  );
  const modelInfoDetails = document.getElementById("model-info-details");
  const continueTrainingBtn = document.getElementById("continue-training-btn");

  if (
    !continueModelSelect ||
    !modelInfoPlaceholder ||
    !modelInfoDetails ||
    !continueTrainingBtn
  )
    return;

  continueModelSelect.addEventListener("change", async function () {
    const modelName = this.value;
    if (!modelName) return;

    modelInfoPlaceholder.style.display = "block";
    modelInfoDetails.style.display = "none";
    modelInfoPlaceholder.textContent = "Loading model information...";

    try {
      // In production, replace with real API call to /api/models/:modelName/latest
      await simulateApiCall(800);

      // Mock data - replace with actual API response
      const mockData = {
        model_name: `${modelName}_retrained_20250321_142536`,
        trained_on: "March 21, 2025",
        epochs: 5,
        accuracy: 0.923,
      };

      // Update UI with model info
      document.getElementById("last-model-name").textContent =
        mockData.model_name;
      document.getElementById("last-model-date").textContent =
        mockData.trained_on;
      document.getElementById("last-model-epochs").textContent =
        mockData.epochs;
      document.getElementById("last-model-accuracy").textContent =
        (mockData.accuracy * 100).toFixed(2) + "%";

      modelInfoPlaceholder.style.display = "none";
      modelInfoDetails.style.display = "block";
    } catch (error) {
      console.error("Error fetching model info:", error);
      modelInfoPlaceholder.textContent =
        "Error loading model information. Please try again.";
      modelInfoDetails.style.display = "none";
    }
  });

  continueTrainingBtn.addEventListener("click", async function () {
    const modelName = continueModelSelect.value;
    const additionalEpochs = document.getElementById("additional-epochs").value;

    if (!modelName) {
      alert("Please select a model");
      return;
    }

    // Show progress UI
    document.getElementById("continue-training-tab").style.display = "none";
    const progressContainer = document.getElementById("progress-container");
    progressContainer.style.display = "block";

    const progressBar = document.getElementById("progress-bar");
    const progressStatus = document.getElementById("progress-status");
    const resultsContainer = document.getElementById("results-container");

    progressBar.style.width = "0%";
    progressStatus.textContent = "Initializing training...";
    resultsContainer.style.display = "none";

    try {
      // Simulate progress
      await simulateTrainingProgress(progressBar, progressStatus);

      // Mock training results - replace with actual API response
      const previousEpochs = parseInt(
        document.getElementById("last-model-epochs").textContent || "0"
      );
      const mockResults = {
        model_name: `${modelName}_retrained_extended_20250407_103045`,
        total_epochs: previousEpochs + parseInt(additionalEpochs),
        model_performance_metrics: {
          final_accuracy: 0.953,
          final_val_accuracy: 0.927,
          f1_score_macro: 0.9281,
        },
      };

      // Display results
      displayRetrainingResults(mockResults);
    } catch (error) {
      console.error("Error during training:", error);
      progressStatus.textContent =
        "Error occurred during training. Please try again.";
      progressBar.style.backgroundColor = "#f44336";
    }
  });
}

/**
 * Sets up the Use Existing Datasets tab functionality
 */
function setupExistingDatasetsTab() {
  const datasetsContainer = document.getElementById("datasets-container");
  const datasetsLoading = document.getElementById("datasets-loading");
  const retrainFromDbBtn = document.getElementById("retrain-from-db-btn");

  if (!datasetsContainer || !retrainFromDbBtn) return;

  // Simulate fetching datasets - in production replace with real API call
  setTimeout(() => {
    // Mock dataset data
    const mockDatasets = [
      {
        id: "dataset_20250324_143256",
        upload_date: "March 24, 2025",
        size: "23.4 MB",
        categories: ["paper", "plastic", "metal", "glass", "organic"],
        image_count: 1247,
      },
      {
        id: "dataset_20250318_092145",
        upload_date: "March 18, 2025",
        size: "15.7 MB",
        categories: ["paper", "plastic", "metal", "glass"],
        image_count: 875,
      },
      {
        id: "dataset_20250310_164532",
        upload_date: "March 10, 2025",
        size: "30.2 MB",
        categories: [
          "paper",
          "plastic",
          "metal",
          "glass",
          "organic",
          "electronic",
        ],
        image_count: 1653,
      },
    ];

    // Hide loading indicator
    if (datasetsLoading) {
      datasetsLoading.style.display = "none";
    }

    if (mockDatasets.length === 0) {
      datasetsContainer.innerHTML = `
        <div class="info-alert" style="background-color: #fff9c4; border-left-color: #fbc02d; color: #7b5800;">
          No datasets found in the database. Please upload a dataset first.
        </div>
      `;
      return;
    }

    let datasetsHTML = "";
    mockDatasets.forEach((dataset) => {
      datasetsHTML += `
        <div class="dataset-card">
          <div class="dataset-header">
            <div class="dataset-name">${dataset.id}</div>
            <div class="dataset-size">${dataset.size}</div>
          </div>
          <div class="dataset-info">
            <div><strong>Uploaded:</strong> ${dataset.upload_date}</div>
            <div><strong>Images:</strong> ${dataset.image_count}</div>
            <div><strong>Categories:</strong> ${dataset.categories.join(
              ", "
            )}</div>
          </div>
        </div>
      `;
    });

    // Keep the loading container but hide it, then add the datasets
    const loadingContainer = datasetsLoading ? datasetsLoading.outerHTML : "";
    datasetsContainer.innerHTML = loadingContainer + datasetsHTML;
    if (datasetsLoading) {
      datasetsLoading.style.display = "none";
    }
  }, 1500);

  retrainFromDbBtn.addEventListener("click", async function () {
    const modelName = document.getElementById("existing-model-select").value;
    const epochs = document.getElementById("training-epochs").value;

    if (!modelName) {
      alert("Please select a model");
      return;
    }

    // Show progress UI
    document.getElementById("existing-datasets-tab").style.display = "none";
    const progressContainer = document.getElementById("progress-container");
    progressContainer.style.display = "block";

    const progressBar = document.getElementById("progress-bar");
    const progressStatus = document.getElementById("progress-status");
    const resultsContainer = document.getElementById("results-container");

    progressBar.style.width = "0%";
    progressStatus.textContent = "Initializing retraining process...";
    resultsContainer.style.display = "none";

    try {
      // Simulate progress with more detailed status messages
      await simulateRetrainingProgress(progressBar, progressStatus, epochs);

      // Mock training results
      const mockResults = {
        model_name: `${modelName}_retrained_20250407_152312`,
        epochs: parseInt(epochs),
        model_performance_metrics: {
          final_accuracy: 0.968,
          final_val_accuracy: 0.942,
          f1_score_macro: 0.951,
        },
      };

      // Display results
      displayRetrainingResults(mockResults);
    } catch (error) {
      console.error("Error during training:", error);
      progressStatus.textContent =
        "Error occurred during training. Please try again.";
      progressBar.style.backgroundColor = "#f44336";
    }
  });
}

/**
 * Sets up the progress and results UI functionality
 */
function setupProgressAndResults() {
  const resetBtn = document.getElementById("reset-btn");
  if (!resetBtn) return;

  resetBtn.addEventListener("click", function () {
    // Hide progress container
    const progressContainer = document.getElementById("progress-container");
    if (progressContainer) {
      progressContainer.style.display = "none";
    }

    // Show the currently active tab content
    const activeTab = document.querySelector(".retrain-tab.active");
    if (activeTab) {
      const tabId = activeTab.getAttribute("data-tab");
      const tabContent = document.getElementById(`${tabId}-tab`);
      if (tabContent) {
        tabContent.style.display = "block";
      }
    }

    // Reset form fields if needed
    const additionalEpochs = document.getElementById("additional-epochs");
    if (additionalEpochs) {
      additionalEpochs.value = 3;
    }

    const trainingEpochs = document.getElementById("training-epochs");
    if (trainingEpochs) {
      trainingEpochs.value = 5;
    }
  });
}

/**
 * Sets up the original upload form to work with the tabbed interface
 */
function setupOriginalUploadForm() {
  const uploadForm = document.getElementById("upload-form");
  const uploadBtn = document.querySelector('.retrain-btn[form="upload-form"]');

  if (!uploadForm || !uploadBtn) return;

  // Update the original form submission
  const originalSubmitHandler = uploadForm.onsubmit;
  uploadForm.addEventListener("submit", function (e) {
    e.preventDefault();

    const fileInput = document.getElementById("file-upload");
    const file = fileInput?.files[0];

    if (!file) {
      alert("Please select a ZIP file first");
      return;
    }

    const modelName =
      document.getElementById("model-select")?.value || "baseline_cnn";

    // Create FormData for the API call
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_name", modelName);

    // Show progress UI
    document.getElementById("new-dataset-tab").style.display = "none";
    const progressContainer = document.getElementById("progress-container");
    progressContainer.style.display = "block";

    const progressBar = document.getElementById("progress-bar");
    const progressStatus = document.getElementById("progress-status");
    const resultsContainer = document.getElementById("results-container");

    progressBar.style.width = "0%";
    progressStatus.textContent = "Uploading dataset...";
    resultsContainer.style.display = "none";

    // Simulate upload and training process
    simulateUploadAndTraining(progressBar, progressStatus, modelName)
      .then((mockResults) => {
        displayRetrainingResults(mockResults);
      })
      .catch((error) => {
        console.error("Error during upload/training:", error);
        progressStatus.textContent =
          "Error occurred during upload or training. Please try again.";
        progressBar.style.backgroundColor = "#f44336";
      });
  });
}

/**
 * Displays training results in the results container
 */
function displayRetrainingResults(results) {
  const resultsContainer = document.getElementById("results-container");
  if (!resultsContainer) return;

  // Set result values
  document.getElementById("result-model-name").textContent =
    results.model_name || "-";
  document.getElementById("result-epochs").textContent =
    results.total_epochs || results.epochs || "-";

  // Format metrics with percentages if they exist
  const metrics = results.model_performance_metrics || {};
  document.getElementById("result-accuracy").textContent =
    metrics.final_accuracy
      ? (metrics.final_accuracy * 100).toFixed(2) + "%"
      : "-";
  document.getElementById("result-val-accuracy").textContent =
    metrics.final_val_accuracy
      ? (metrics.final_val_accuracy * 100).toFixed(2) + "%"
      : "-";
  document.getElementById("result-f1-score").textContent =
    metrics.f1_score_macro
      ? (metrics.f1_score_macro * 100).toFixed(2) + "%"
      : "-";

  // Show results container
  resultsContainer.style.display = "block";
}

/**
 * Simulates an API call with a delay
 */
function simulateApiCall(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Simulates training progress with updates to the progress bar and status message
 */
async function simulateTrainingProgress(
  progressBar,
  statusElement,
  intervalMs = 500
) {
  for (let i = 0; i <= 100; i += 10) {
    await simulateApiCall(intervalMs);
    progressBar.style.width = `${i}%`;

    if (i === 10) statusElement.textContent = "Loading model from database...";
    if (i === 30) statusElement.textContent = "Preparing datasets...";
    if (i === 50) statusElement.textContent = "Training in progress...";
    if (i === 80) statusElement.textContent = "Evaluating model performance...";
    if (i === 100) statusElement.textContent = "Training complete!";
  }
}

/**
 * Simulates retraining progress with more detailed status updates
 */
async function simulateRetrainingProgress(
  progressBar,
  statusElement,
  epochs,
  intervalMs = 400
) {
  for (let i = 0; i <= 100; i += 5) {
    await simulateApiCall(intervalMs);
    progressBar.style.width = `${i}%`;

    if (i === 5) statusElement.textContent = "Loading model from storage...";
    if (i === 15)
      statusElement.textContent = "Fetching datasets from database...";
    if (i === 25) statusElement.textContent = "Preparing combined dataset...";
    if (i === 40)
      statusElement.textContent = "Setting up model architecture...";
    if (i === 50)
      statusElement.textContent = `Training in progress (epoch 1/${epochs})...`;
    if (i === 70)
      statusElement.textContent = `Training in progress (epoch ${Math.ceil(
        epochs / 2
      )}/${epochs})...`;
    if (i === 90)
      statusElement.textContent = `Training in progress (epoch ${epochs}/${epochs})...`;
    if (i === 100) statusElement.textContent = "Training complete!";
  }
}

/**
 * Simulates the upload and training process for the original upload form
 */
async function simulateUploadAndTraining(
  progressBar,
  statusElement,
  modelName
) {
  // Upload phase (0-30%)
  for (let i = 0; i <= 30; i += 5) {
    await simulateApiCall(300);
    progressBar.style.width = `${i}%`;

    if (i === 5) statusElement.textContent = "Uploading dataset...";
    if (i === 15) statusElement.textContent = "Validating dataset...";
    if (i === 25) statusElement.textContent = "Processing dataset...";
  }

  // Training phase (30-100%)
  for (let i = 35; i <= 100; i += 5) {
    await simulateApiCall(400);
    progressBar.style.width = `${i}%`;

    if (i === 35) statusElement.textContent = "Loading model...";
    if (i === 45)
      statusElement.textContent = "Setting up training environment...";
    if (i === 55)
      statusElement.textContent = "Training in progress (epoch 1/5)...";
    if (i === 65)
      statusElement.textContent = "Training in progress (epoch 2/5)...";
    if (i === 75)
      statusElement.textContent = "Training in progress (epoch 3/5)...";
    if (i === 85)
      statusElement.textContent = "Training in progress (epoch 4/5)...";
    if (i === 95)
      statusElement.textContent = "Training in progress (epoch 5/5)...";
    if (i === 100) statusElement.textContent = "Training complete!";
  }

  // Return mock results
  return {
    model_name: `${modelName}_retrained_20250407_163415`,
    total_epochs: 5,
    model_performance_metrics: {
      final_accuracy: 0.945,
      final_val_accuracy: 0.922,
      f1_score_macro: 0.934,
    },
  };
}
