# Ekonify - Revolutionizing Waste Classification for a Sustainable Future

Ekonify is an AI-powered waste classification and recycling optimization platform with a complete MLOps pipeline. The system facilitates automated image classification, model retraining, and performance evaluation to support sustainable waste management solutions.

![image](./front-end/static/images/Ekonify-H.S..png)

## Core Features

### 1. Prediction Pipeline
- **Image Upload & Processing**: Submit waste images through an intuitive UI
- **Real-time Classification**: Instant prediction with confidence scoring
- **Automated Database Storage**: All predictions stored for future model enhancement

### 2. Retraining Pipeline
- **Seamless Data Upload**: Import new training datasets via web interface
- **Custom Model Selection**: Choose from baseline or previously retrained models
- **Progressive Training**: Extend existing models with new data
- **Automated Preprocessing**: Dataset normalization and augmentation
- **Comprehensive Metrics**: Track performance improvements across retraining sessions

### 3. Data Visualization & Analytics
- **Model Performance Tracking**: Visualize accuracy, precision, recall, and F1 scores
- **Environmental Impact Metrics**: Monitor waste diversion statistics
- **Class Distribution Analysis**: Understand dataset characteristics

## Technology Stack

### Backend
- **FastAPI**: High-performance API framework with automatic documentation
- **TensorFlow**: Deep learning for image classification
- **MongoDB**: NoSQL database with GridFS for efficient image and model storage
- **Python-Multipart**: Handling file uploads and form data
- **Python-dotenv**: Environment variable management

### Frontend
- **HTML/CSS/JavaScript**: Responsive UI with dynamic content loading
- **Feature-rich Interface**: Real-time progress tracking during model training
- **Visualization Components**: Interactive charts and statistics

### MLOps Components
- **Model Versioning**: Tracking model evolution across retraining sessions
- **A/B Testing Support**: Compare model performance metrics
- **Containerized Deployment**: Ready for cloud scaling

## MLOps Pipeline Architecture

```
                                     ┌─────────────────┐
                                     │                 │
                                     │    Frontend     │
                                     │                 │
                                     └────────┬────────┘
                                              │
                                              ▼
┌───────────────┐             ┌─────────────────────────────┐              ┌────────────────┐
│               │             │                             │              │                │
│  Image Input  │────────────▶│       FastAPI Backend       │◀─────────────│  MongoDB       │
│               │             │                             │              │                │
└───────────────┘             └──────────────┬──────────────┘              └────────────────┘
                                             │                                     ▲
                                             │                                     │
                                             ▼                                     │
                              ┌─────────────────────────────┐                     │
                              │                             │                     │
                              │     Model Prediction        │                     │
                              │                             │                     │
                              └──────────────┬──────────────┘                     │
                                             │                                     │
                                             │                                     │
                              ┌──────────────▼──────────────┐                     │
                              │                             │                     │
                              │   Retraining Process        │─────────────────────┘
                              │                             │
                              └─────────────────────────────┘
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- MongoDB 4.4+
- Node.js 14+ (for frontend development)

### Backend Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/eadewusic/Ekonify-MLOP.git
   cd Ekonify-MLOP
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure MongoDB:
   - Create a `.env` file in the API directory:
     ```
     MONGODB_URI=mongodb://adminUser:password@localhost:27017/?authSource=admin
     PORT=8000
     ALLOWED_ORIGINS=http://localhost:3000,https://ekonify-frontend.onrender.com
     ```

5. Start the backend server:
   ```bash
   cd API
   uvicorn main:app --reload
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd front-end
   ```

2. Update API configuration in `static/config.js`:
   ```javascript
   const config = {
     apiBaseUrl: "http://localhost:8000"  // Change to your API URL
   };
   ```

3. Serve the frontend using your preferred method:
   ```bash
   python -m http.server 3000
   ```

## API Documentation

The API documentation is automatically generated and available at `/docs` when the server is running. Key endpoints include:

### Prediction
- `POST /predict/`: Upload an image and get a waste classification prediction

### Retraining
- `POST /upload`: Upload a ZIP dataset and retrain a model
- `POST /continue-training`: Continue training with additional epochs
- `POST /retrain-from-db`: Retrain using all stored datasets

### Model Management
- `GET /models`: List all available models
- `GET /models/{model_name}/latest`: Get information about the latest version of a specific model
- `GET /datasets`: List all available datasets

## MLOps Process Demonstration

### Prediction Process
1. The user uploads a waste image via the web interface
2. The image is preprocessed and fed into the current model
3. The model generates a prediction with confidence score
4. Results are displayed to the user and saved to MongoDB
5. Saved images are available for future retraining

### Retraining Process
1. Upload a dataset containing labeled waste images
2. Select a baseline or previously trained model
3. Configure training parameters (epochs, etc.)
4. Initiate the retraining process
5. Monitor progress and performance metrics in real-time
6. The retrained model is saved and available for future predictions

## Performance Metrics

The system tracks comprehensive performance metrics for each model version:

- Accuracy: Overall predictive performance
- Precision: Measure of exactness in predictions
- Recall: Measure of completeness in predictions
- F1 Score: Harmonic mean of precision and recall
- Loss: Training and validation error metrics

## Continuous Improvement

The MLOps pipeline facilitates continuous model enhancement through:

1. **Automated Data Collection**: All predictions are stored for future retraining
2. **Incremental Learning**: Models can be extended with new data
3. **Performance Tracking**: Clear visibility of improvements across versions
4. **Simplified Deployment**: New models can be quickly deployed to production

## Deployment
- Ekonify Front-End: See it on [Render](https://ekonify-frontend.onrender.com)
- Ekonify API: See it on [Railway](https://railway.com/)

## Video Presentation
Watch my demo video [here](https://drive.google.com/file/d/1OzsWdZ60iQo4WA-le_D9295rnVbKTey0/view?usp=drive_link) 

## Initial Repo for Model Training
See it here - https://github.com/eadewusic/Ekonify.git