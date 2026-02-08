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
        # Ensuring it Matches ClearML dashboard exactly
        last_task = Task.get_task(
            project_name="BTC_MLOps", 
            task_name="Train_RandomForest",
            task_filter={'status': ['completed']}
        )
        
        if last_task:
            if last_task.id != current_model_id:
                print(f"✨ New model found! Task ID: {last_task.id}. Downloading...")
                models = last_task.get_models()
                if models and 'output' in models:
                    new_model_path = models['output'][0].get_local_copy()
                    current_model = joblib.load(new_model_path)
                    current_model_id = last_task.id
                    print(f"API successfully updated to model ID: {current_model_id}")
        else:
            print("Sync check: No completed tasks found for 'Train_RandomForest'.")
            
    except Exception as e:
        print(f"ClearML Sync Error: {e}")

async def model_check_loop():
    while True:
        load_latest_model()
        await asyncio.sleep(600) # Check every 10 minutes

@app.on_event("startup")
async def startup_event():
    # Sync immediately on startup
    asyncio.create_task(model_check_loop())

@app.post("/predict")
async def predict(data: list):
    if current_model is None:
        return {"error": "Model not loaded. ClearML sync in progress..."}
    
    # Ensuring data is provided as a 2D array for Scikit-learn
    prediction = current_model.predict([data])
    return {"prediction_usd": float(prediction[0])}

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