from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your API routers
from predict import app as predict_app
from retrain import app as retrain_app

app = FastAPI(
    title="Ekonify API",
    description="Combined API for waste classification prediction and model retraining",
    version="2.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import all routes from both apps
# Use APIRouter from FastAPI to combine the routes from both apps
from fastapi.routing import APIRouter

# Create routers from the existing apps
predict_router = APIRouter()
retrain_router = APIRouter()

# Copy routes from predict_app to predict_router
for route in predict_app.routes:
    predict_router.routes.append(route)

# Copy routes from retrain_app to retrain_router
for route in retrain_app.routes:
    retrain_router.routes.append(route)

# Include the routers in the main app
app.include_router(predict_router)
app.include_router(retrain_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)