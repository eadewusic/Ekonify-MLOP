from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Import your API routers
from predict import app as predict_app
from retrain import app as retrain_app

app = FastAPI(
    title="Ekonify API",
    description="Combined API for waste classification prediction and model retraining",
    version="2.0"
)

# Get allowed origins from environment variable or use default for development
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Use environment variable in production
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

@app.get("/")
async def root():
    return {"message": "Ekonify API v2.0 is running!"}

if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable for production (Render sets this)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)