document.addEventListener("DOMContentLoaded", () => {
  // API base URL - change this for production deployment
  const API_BASE_URL = "http://localhost:8000"; // Your API server URL

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
          // Use the animated checkmark function
          showFileSelectedAnimatedCheckmark(files[0]);
        } else if (imagePreview) {
          previewImage(); // Show the preview when dropped
        }
      }
    });

    // Handle file selection
    fileInput.addEventListener("change", function () {
      const file = this.files[0];

      if (isRetrainPage) {
        // Use the animated checkmark function
        showFileSelectedAnimatedCheckmark(file);
      } else if (imagePreview) {
        previewImage();
      }
    });

    // Show file selected with animated checkmark
    function showFileSelectedAnimatedCheckmark(file) {
      if (!file) return;

      // Check if it's a ZIP file
      if (
        file.type === "application/zip" ||
        file.name.toLowerCase().endsWith(".zip")
      ) {
        // Create container for the success animation
        dropZone.innerHTML = `
          <div class="file-success-container">
            <div class="checkmark-circle">
              <div class="checkmark-check"></div>
            </div>
            <p class="file-selected-text">
              Selected file: <strong>${file.name}</strong>
            </p>
          </div>
        `;

        // Add the success class to trigger animation
        setTimeout(() => {
          const checkmarkCircle = document.querySelector(".checkmark-circle");
          if (checkmarkCircle) {
            checkmarkCircle.classList.add("animate");
          }
        }, 100);

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

    // Reset file input
    function resetFileInput() {
      fileInput.value = "";
      if (isRetrainPage) {
        dropZone.innerHTML = `
          <div class="retrain-upload-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#999" stroke-width="1.5"
              stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
          </div>
          <p class="retrain-upload-text">
            Drag and drop your ZIP file here, or <span class="retrain-upload-link">browse</span>
          </p>
        `;
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
        let endpoint = `${API_BASE_URL}/predict/`;
        if (isRetrainPage) {
          endpoint = `${API_BASE_URL}/upload`;

          // Add model name for retraining
          const modelSelect = document.getElementById("model-select");
          if (modelSelect) {
            formData.append("model_name", modelSelect.value);
          }

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

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "Request failed");
          }

          const result = await response.json();

          if (isRetrainPage) {
            // Handle retrain response
            if (retrainBtn) {
              retrainBtn.textContent = "Start Retraining";
              retrainBtn.disabled = false;
            }

            // Show progress UI with real results
            document.getElementById("new-dataset-tab").style.display = "none";
            const progressContainer =
              document.getElementById("progress-container");
            progressContainer.style.display = "block";

            const progressBar = document.getElementById("progress-bar");
            const progressStatus = document.getElementById("progress-status");

            // Update with real training results
            progressBar.style.width = "100%";
            progressStatus.textContent = "Training complete!";

            // Display the real results
            displayRetrainingResults(result.results);
          } else {
            // Handle predict response
            displayPrediction({
              class: result.prediction,
              confidence: result.confidence,
            });
          }
        } catch (error) {
          console.error(
            `${isRetrainPage ? "Retraining" : "Prediction"} failed:`,
            error
          );
          alert(
            `An error occurred while ${
              isRetrainPage ? "retraining" : "predicting"
            }: ${error.message}`
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

      predictedClass.textContent = result.class || "Unknown";
      confidence.textContent = result.confidence || "N/A";
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

    // Initialize the application by fetching available models
    initializeModelLists();

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

  // Initialize model lists for all selects
  async function initializeModelLists() {
    const API_BASE_URL = "http://localhost:8000";

    try {
      const response = await fetch(`${API_BASE_URL}/models`);
      if (!response.ok) {
        throw new Error("Failed to fetch models");
      }

      const models = await response.json();

      // Populate all model select dropdowns
      const modelSelects = [
        document.getElementById("model-select"),
        document.getElementById("continue-model-select"),
        document.getElementById("existing-model-select"),
      ];

      modelSelects.forEach((select) => {
        if (select) {
          // Clear existing options except the first one (if it's a placeholder)
          const firstOption = select.firstElementChild;
          select.innerHTML = "";
          if (firstOption && firstOption.disabled) {
            select.appendChild(firstOption);
          }

          // Add base models
          const baseModels = models.filter((model) => model.type === "base");
          if (baseModels.length > 0) {
            const baseOptgroup = document.createElement("optgroup");
            baseOptgroup.label = "Base Models";

            baseModels.forEach((model) => {
              const option = document.createElement("option");
              option.value = model.name;
              option.textContent = model.name;
              baseOptgroup.appendChild(option);
            });

            select.appendChild(baseOptgroup);
          }

          // Add retrained models if any
          const retrainedModels = models.filter(
            (model) => model.type === "retrained"
          );
          if (retrainedModels.length > 0) {
            const retrainedOptgroup = document.createElement("optgroup");
            retrainedOptgroup.label = "Retrained Models";

            retrainedModels.forEach((model) => {
              const option = document.createElement("option");
              option.value = model.name;
              option.textContent = `${model.name} (${model.created_at})`;
              retrainedOptgroup.appendChild(option);
            });

            select.appendChild(retrainedOptgroup);
          }
        }
      });

      console.log("Model dropdowns populated successfully");
    } catch (error) {
      console.error("Error fetching models:", error);
    }
  }
});

// ===== RETRAIN FUNCTIONALITY ====

/**
 * Sets up the Continue Training tab functionality
 */
function setupContinueTrainingTab() {
  const API_BASE_URL = "http://localhost:8000"; // Match the URL from top of file

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
      // Call API to get model info
      const response = await fetch(
        `${API_BASE_URL}/models/${modelName}/latest`
      );
      if (!response.ok) {
        throw new Error(`Failed to fetch model info: ${response.statusText}`);
      }

      const modelData = await response.json();

      // Update UI with model info
      document.getElementById("last-model-name").textContent =
        modelData.model_name || "Unknown";
      document.getElementById("last-model-date").textContent =
        modelData.timestamp
          ? new Date(modelData.timestamp).toLocaleDateString()
          : "Unknown";
      document.getElementById("last-model-epochs").textContent =
        modelData.metrics?.epochs || "Unknown";
      document.getElementById("last-model-accuracy").textContent = modelData
        .metrics?.final_accuracy
        ? `${(modelData.metrics.final_accuracy * 100).toFixed(2)}%`
        : "Unknown";

      modelInfoPlaceholder.style.display = "none";
      modelInfoDetails.style.display = "block";
    } catch (error) {
      console.error("Error fetching model info:", error);
      modelInfoPlaceholder.textContent =
        "Error loading model information: " + error.message;
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

    progressBar.style.width = "10%";
    progressStatus.textContent = "Initializing training...";
    resultsContainer.style.display = "none";

    try {
      // Create form data for the API call
      const formData = new FormData();
      formData.append("model_name", modelName);
      formData.append("additional_epochs", additionalEpochs);

      // Call the continue-training API
      const response = await fetch(`${API_BASE_URL}/continue-training`, {
        method: "POST",
        body: formData,
      });

      // Update progress bar while waiting for response
      let progress = 10;
      const progressInterval = setInterval(() => {
        if (progress < 90) {
          progress += 5;
          progressBar.style.width = `${progress}%`;
          progressStatus.textContent = "Training in progress...";
        } else {
          clearInterval(progressInterval);
        }
      }, 1000);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Training failed");
      }

      // Get response data
      const result = await response.json();
      clearInterval(progressInterval);

      // Show complete progress
      progressBar.style.width = "100%";
      progressStatus.textContent = "Training complete!";

      // Display results
      displayRetrainingResults(result);
    } catch (error) {
      console.error("Error during training:", error);
      progressStatus.textContent =
        "Error occurred during training: " + error.message;
      progressBar.style.backgroundColor = "#f44336";
    }
  });
}

/**
 * Sets up the Use Existing Datasets tab functionality
 */
function setupExistingDatasetsTab() {
  const API_BASE_URL = "http://localhost:8000"; // Match the URL from top of file

  const datasetsContainer = document.getElementById("datasets-container");
  const datasetsLoading = document.getElementById("datasets-loading");
  const retrainFromDbBtn = document.getElementById("retrain-from-db-btn");

  if (!datasetsContainer || !retrainFromDbBtn) return;

  // Load datasets from API
  loadDatasets();

  async function loadDatasets() {
    // Add a loading indicator if it doesn't exist
    if (!datasetsLoading) {
      const loadingDiv = document.createElement("div");
      loadingDiv.id = "datasets-loading";
      loadingDiv.className = "text-center py-4";
      loadingDiv.innerHTML = "Loading datasets...";
      datasetsContainer.prepend(loadingDiv);
    } else {
      datasetsLoading.style.display = "block";
    }

    try {
      const response = await fetch(`${API_BASE_URL}/datasets`);
      if (!response.ok) {
        throw new Error(`Failed to fetch datasets: ${response.statusText}`);
      }

      const datasets = await response.json();

      // Hide loading indicator
      const loadingElement = document.getElementById("datasets-loading");
      if (loadingElement) {
        loadingElement.style.display = "none";
      }

      if (datasets.length === 0) {
        datasetsContainer.innerHTML = `
          <div class="info-alert" style="background-color: #fff9c4; border-left-color: #fbc02d; color: #7b5800;">
            No datasets found in the database. Please upload a dataset first.
          </div>
        `;
        return;
      }

      // Clear previous content except loading indicator
      const currentContent = datasetsContainer.innerHTML;
      const loadingHTML = loadingElement ? loadingElement.outerHTML : "";
      datasetsContainer.innerHTML = loadingHTML;

      // Add dataset cards
      datasets.forEach((dataset) => {
        // Format size in MB or KB
        const sizeInMB = dataset.size_bytes
          ? (dataset.size_bytes / (1024 * 1024)).toFixed(2)
          : "Unknown";
        const formattedSize =
          sizeInMB < 0.01
            ? `${(dataset.size_bytes / 1024).toFixed(2)} KB`
            : `${sizeInMB} MB`;

        const datasetCard = document.createElement("div");
        datasetCard.className = "dataset-card";
        datasetCard.innerHTML = `
          <div class="dataset-header">
            <div class="dataset-name">${dataset.filename || "Dataset"}</div>
            <div class="dataset-size">${formattedSize}</div>
          </div>
          <div class="dataset-info">
            <div><strong>Uploaded:</strong> ${new Date(
              dataset.timestamp
            ).toLocaleString()}</div>
            <div><strong>Dataset ID:</strong> ${dataset._id || "Unknown"}</div>
          </div>
        `;

        datasetsContainer.appendChild(datasetCard);
      });

      // Hide loading element if it exists
      if (loadingElement) {
        loadingElement.style.display = "none";
      }
    } catch (error) {
      console.error("Error fetching datasets:", error);

      // Hide loading indicator
      const loadingElement = document.getElementById("datasets-loading");
      if (loadingElement) {
        loadingElement.style.display = "none";
      }

      datasetsContainer.innerHTML = `
        <div class="info-alert" style="background-color: #ffebee; border-left-color: #f44336; color: #b71c1c;">
          Error loading datasets: ${error.message}
        </div>
      `;
    }
  }

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
      // Create form data for the API call
      const formData = new FormData();
      formData.append("model_name", modelName);

      // Update progress
      progressBar.style.width = "10%";
      progressStatus.textContent = "Sending request to server...";

      // Call the retrain-from-db API
      const response = await fetch(`${API_BASE_URL}/retrain-from-db`, {
        method: "POST",
        body: formData,
      });

      // Update progress bar while waiting for response
      let progress = 10;
      const progressInterval = setInterval(() => {
        if (progress < 90) {
          progress += 5;
          progressBar.style.width = `${progress}%`;
          progressStatus.textContent = `Retraining in progress... ${progress}%`;
        } else {
          clearInterval(progressInterval);
        }
      }, 2000);

      if (!response.ok) {
        clearInterval(progressInterval);
        const errorData = await response.json();
        throw new Error(errorData.detail || "Retraining failed");
      }

      // Get response data
      const result = await response.json();
      clearInterval(progressInterval);

      // Update progress to complete
      progressBar.style.width = "100%";
      progressStatus.textContent = "Retraining complete!";

      // Display results
      displayRetrainingResults(result.results);
    } catch (error) {
      console.error("Error during training:", error);
      progressStatus.textContent =
        "Error occurred during training: " + error.message;
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
  const API_BASE_URL = "http://localhost:8000"; // Match the URL from top of file

  const uploadForm = document.getElementById("upload-form");
  const uploadBtn = document.querySelector('.retrain-btn[form="upload-form"]');
  const fileInput = document.getElementById("file-upload");

  if (!uploadForm || !uploadBtn) return;

  // Update the form submission
  uploadForm.addEventListener("submit", async function (e) {
    e.preventDefault();

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

    try {
      // Update progress to show upload started
      progressBar.style.width = "20%";
      progressStatus.textContent = "Uploading dataset to server...";

      // Make the API call
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      // Update progress while processing
      let progress = 20;
      const progressInterval = setInterval(() => {
        if (progress < 80) {
          progress += 5;
          progressBar.style.width = `${progress}%`;
          progressStatus.textContent =
            progress < 50 ? "Processing dataset..." : "Training model...";
        } else {
          clearInterval(progressInterval);
        }
      }, 1000);

      // Check if response is successful
      if (!response.ok) {
        clearInterval(progressInterval);
        const errorData = await response.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      const data = await response.json();
      clearInterval(progressInterval);

      // Update to complete
      progressBar.style.width = "100%";
      progressStatus.textContent = "Upload and training complete!";

      // Display results
      displayRetrainingResults(data.results);
    } catch (error) {
      console.error("Error during upload/training:", error);
      progressStatus.textContent = "Error: " + error.message;
      progressBar.style.backgroundColor = "#f44336";
    }
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
    results.total_epochs || results.model_performance_metrics?.epochs || "-";

  // Format metrics with percentages if they exist
  const metrics = results.model_performance_metrics || {};
  document.getElementById("result-accuracy").textContent =
    metrics.final_accuracy !== undefined
      ? (metrics.final_accuracy * 100).toFixed(2) + "%"
      : "-";
  document.getElementById("result-val-accuracy").textContent =
    metrics.final_val_accuracy !== undefined
      ? (metrics.final_val_accuracy * 100).toFixed(2) + "%"
      : "-";
  document.getElementById("result-f1-score").textContent =
    metrics.f1_score_macro !== undefined
      ? (metrics.f1_score_macro * 100).toFixed(2) + "%"
      : "-";

  // Show results container
  resultsContainer.style.display = "block";
}
