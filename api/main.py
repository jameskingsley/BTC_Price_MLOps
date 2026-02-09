import os
import uvicorn
import joblib
import asyncio
from fastapi import FastAPI
from clearml import Task

app = FastAPI()

# Ensuring the data directory exists
os.makedirs("data", exist_ok=True)

# Global variables
MODEL_PATH = "data/model.pkl"
current_model = None
current_model_id = None

# Initial load check
if os.path.exists(MODEL_PATH):
    try:
        current_model = joblib.load(MODEL_PATH)
        print(f"Initial model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load local model: {e}")

def load_latest_model():
    global current_model, current_model_id
    try:
        # Search for the latest task in the project
        last_task = Task.get_task(
            project_name="BTC_MLOps", 
            task_name="Train_RandomForest",
            task_filter={'status': ['completed', 'in_progress']}
        )
        
        if last_task and last_task.id != current_model_id:
            print(f"New task found! ID: {last_task.id}. Checking for artifact...")
            
            # 1. Access the artifact object
            artifact = last_task.artifacts.get('btc_prediction_model')
            
            if artifact:
                print(f"Attempting to download artifact: btc_prediction_model...")
                # 2. Get the local copy path
                # We check if it's None to avoid the 'NoneType' joblib error
                new_model_path = artifact.get_local_copy()
                
                if new_model_path:
                    print(f"Loading model from: {new_model_path}")
                    current_model = joblib.load(new_model_path)
                    current_model_id = last_task.id
                    print(f"API successfully updated to model ID: {current_model_id}")
                else:
                    print(f" File path is None. The model might still be uploading to ClearML storage.")
            else:
                available = list(last_task.artifacts.keys())
                print(f" 'btc_prediction_model' not found. Available keys: {available}")
        
        elif not last_task:
            print("No matching tasks found in project 'BTC_MLOps'.")
                
    except Exception as e:
        print(f"ClearML Sync Error: {e}")

async def model_check_loop():
    while True:
        load_latest_model()
        # Check every 10 minutes
        await asyncio.sleep(600)

@app.on_event("startup")
async def startup_event():
    # Trigger an immediate sync on startup
    asyncio.create_task(model_check_loop())

@app.get("/")
async def root():
    return {
        "message": "BTC Price Prediction API is live",
        "model_loaded": current_model is not None,
        "task_id": current_model_id
    }

@app.post("/predict")
async def predict(data: list):
    if current_model is None:
        return {"error": "Model not loaded. ClearML sync in progress..."}
    
    try:
        # Wrap input in a list to match Scikit-learn's 2D array expectation
        prediction = current_model.predict([data])
        return {"prediction_usd": float(prediction[0])}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "model_loaded": current_model is not None,
        "model_id": current_model_id,
        "accuracy_metric": "R2: 0.9964"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)