/**
 * Ekonify Model Retraining - Enhanced Functionality
 * This script adds tab-based functionality to the model retraining page
 */

document.addEventListener("DOMContentLoaded", function () {
  // Tab switching functionality
  const tabs = document.querySelectorAll(".retrain-tab");
  const tabContents = document.querySelectorAll(".retrain-tab-content");

  if (tabs.length > 0 && tabContents.length > 0) {
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const tabId = tab.getAttribute("data-tab");

        // Remove active class from all tabs and contents
        tabs.forEach((t) => t.classList.remove("active"));
        tabContents.forEach((c) => c.classList.remove("active"));

        // Add active class to clicked tab and corresponding content
        tab.classList.add("active");
        document.getElementById(`${tabId}-tab`).classList.add("active");

        // Hide the progress container when switching tabs
        const progressContainer = document.getElementById("progress-container");
        if (progressContainer) {
          progressContainer.style.display = "none";
        }

        // Reset forms if needed
        if (tabId === "new-dataset") {
          const uploadForm = document.getElementById("upload-form");
          if (uploadForm) uploadForm.reset();
        }
      });
    });

    // Continue Training Tab Functionality
    setupContinueTrainingTab();

    // Use Existing Datasets Tab Functionality
    setupExistingDatasetsTab();

    // Setup progress and results functionality
    setupProgressAndResults();

    // Make sure the original form submission is updated
    updateOriginalFormSubmission();
  }
});

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
  ) {
    return; // Exit if any elements are missing
  }

  // Handle model selection change
  continueModelSelect.addEventListener("change", async function () {
    const modelName = this.value;
    if (!modelName) return;

    modelInfoPlaceholder.style.display = "block";
    modelInfoDetails.style.display = "none";
    modelInfoPlaceholder.textContent = "Loading model information...";

    try {
      // In production, replace this with a real API call to /api/models/:modelName/latest
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

  // Handle continue training button click
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
    if (!progressContainer) return;

    progressContainer.style.display = "block";

    const progressBar = document.getElementById("progress-bar");
    const progressStatus = document.getElementById("progress-status");
    const resultsContainer = document.getElementById("results-container");

    if (!progressBar || !progressStatus || !resultsContainer) return;

    progressBar.style.width = "0%";
    progressStatus.textContent = "Initializing training...";
    resultsContainer.style.display = "none";

    try {
      // In production, replace this with real API call to /continue-training
      // e.g. fetch('/api/continue-training', { method: 'POST', body: JSON.stringify({ model_name: modelName, additional_epochs: additionalEpochs }) })

      // Simulate progress
      await simulateTrainingProgress(progressBar, progressStatus, 800);

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
      displayResults(mockResults);
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

  if (!datasetsContainer || !retrainFromDbBtn) {
    return; // Exit if elements are missing
  }

  // Simulate fetching datasets - in production replace with real API call
  // e.g. fetch('/api/datasets').then(res => res.json()).then(datasets => { ... })
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
                        <div><strong>Uploaded:</strong> ${
                          dataset.upload_date
                        }</div>
                        <div><strong>Images:</strong> ${
                          dataset.image_count
                        }</div>
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

  // Handle retrain from database button click
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
    if (!progressContainer) return;

    progressContainer.style.display = "block";

    const progressBar = document.getElementById("progress-bar");
    const progressStatus = document.getElementById("progress-status");
    const resultsContainer = document.getElementById("results-container");

    if (!progressBar || !progressStatus || !resultsContainer) return;

    progressBar.style.width = "0%";
    progressStatus.textContent = "Initializing retraining process...";
    resultsContainer.style.display = "none";

    try {
      // In production, replace with real API call
      // e.g. fetch('/api/retrain-from-db', { method: 'POST', body: JSON.stringify({ model_name: modelName, epochs: epochs }) })

      // Simulate progress with more detailed status messages
      await simulateRetrainingProgress(
        progressBar,
        progressStatus,
        epochs,
        400
      );

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
      displayResults(mockResults);
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
 * Displays training results in the results container
 * @param {Object} results - The training results object
 */
function displayResults(results) {
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
 * @param {number} ms - Milliseconds to delay
 */
function simulateApiCall(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Simulates training progress with updates to the progress bar and status message
 * @param {HTMLElement} progressBar - The progress bar element
 * @param {HTMLElement} statusElement - The status message element
 * @param {number} intervalMs - Milliseconds between updates
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
 * @param {HTMLElement} progressBar - The progress bar element
 * @param {HTMLElement} statusElement - The status message element
 * @param {number} epochs - Number of training epochs
 * @param {number} intervalMs - Milliseconds between updates
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
 * @param {HTMLElement} progressBar - The progress bar element
 * @param {HTMLElement} statusElement - The status message element
 * @param {string} modelName - The name of the model being trained
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

/**
 * Updates the original form submission to work with the new interface
 */
function updateOriginalFormSubmission() {
  const uploadForm = document.getElementById("upload-form");
  const uploadBtn = document.querySelector('.retrain-btn[form="upload-form"]');

  if (!uploadForm || !uploadBtn) return;

  // Replace the original form submission with our custom handler
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
    if (!progressContainer) return;

    progressContainer.style.display = "block";

    const progressBar = document.getElementById("progress-bar");
    const progressStatus = document.getElementById("progress-status");
    const resultsContainer = document.getElementById("results-container");

    if (!progressBar || !progressStatus || !resultsContainer) return;

    progressBar.style.width = "0%";
    progressStatus.textContent = "Uploading dataset...";
    resultsContainer.style.display = "none";

    // In production, replace with real API call
    // fetch('/upload', { method: 'POST', body: formData })
    //   .then(response => response.json())
    //   .then(data => { ... })

    // Simulate upload and training process
    simulateUploadAndTraining(progressBar, progressStatus, modelName)
      .then((mockResults) => {
        displayResults(mockResults);
      })
      .catch((error) => {
        console.error("Error during upload/training:", error);
        progressStatus.textContent =
          "Error occurred during upload or training. Please try again.";
        progressBar.style.backgroundColor = "#f44336";
      });
  });
}
