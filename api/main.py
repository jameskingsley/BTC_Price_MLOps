import os
import uvicorn
import joblib
import asyncio
from fastapi import FastAPI, BackgroundTasks
from clearml import Task, Model

app = FastAPI()

# Global variables to hold the loaded model and its ID
MODEL_PATH = "data/model.pkl"

# Initial load of the model
try:
    current_model = joblib.load(MODEL_PATH)
    print(f"Initial model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Local model not found. Waiting for ClearML sync. Error: {e}")
    current_model = None

current_model_id = None

def load_latest_model():
    global current_model, current_model_id
    try:
        # Find the latest successful training task in your project
        last_task = Task.get_task(
            project_name="BTC_Price_Prediction",
            task_name="BTC_Model_Training",
            task_filter={'status': ['completed']}
        )
        
        if last_task and last_task.id != current_model_id:
            print(f"New model found! Task ID: {last_task.id}. Downloading...")
            # Get the output model from that task
            models = last_task.get_models()
            if models and 'output' in models:
                # This downloads the model from ClearML server to local cache
                new_model_path = models['output'][0].get_local_copy()
                current_model = joblib.load(new_model_path)
                current_model_id = last_task.id
                print(f" API successfully updated to model ID: {current_model_id}")
    except Exception as e:
        print(f"Failed to check for new model: {e}")

# This runs in the background every 10 minutes
async def model_check_loop():
    while True:
        # Run the sync logic
        load_latest_model()
        await asyncio.sleep(600) # 10 minutes

@app.on_event("startup")
async def startup_event():
    # Start the background checking loop when the API starts
    asyncio.create_task(model_check_loop())

@app.post("/predict")
async def predict(data: list):
    if current_model is None:
        return {"error": "Model is not yet loaded. Please wait for background sync."}
    
    # data is expected to be the sequence of last 10 prices
    prediction = current_model.predict([data])
    return {"prediction_usd": float(prediction[0])}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_id": current_model_id}

if __name__ == "__main__":
    # Render port environment
    port = int(os.environ.get("PORT", 8000))
    # host 0.0.0.0 is mandatory for cloud deployments
    uvicorn.run(app, host="0.0.0.0", port=port)