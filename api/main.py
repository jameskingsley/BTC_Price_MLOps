import os
import uvicorn
import joblib
import asyncio
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from clearml import Task

app = FastAPI()

# Ensuring the data directory exists
os.makedirs("data", exist_ok=True)

# Global variables
current_model = None
current_scaler = None
current_model_id = None

# Input Schema for Data Validation
class PredictionInput(BaseModel):
    data: list

def load_latest_model():
    global current_model, current_model_id, current_scaler
    try:
        # Search for the latest successful task
        last_task = Task.get_task(
            project_name="BTC_MLOps", 
            task_name="Train_RandomForest",
            task_filter={'status': ['completed']}
        )
        
        if last_task and last_task.id != current_model_id:
            print(f"New task found! ID: {last_task.id}. Checking for artifacts...")
            
            # 1. Access the model and scaler artifacts
            model_art = last_task.artifacts.get('btc_prediction_model')
            
            # Note: Ensuring the training/preprocess script uploads 'scaler.pkl' 
            # to ClearML, or keep it locally if it doesn't change.
            if model_art:
                new_model_path = model_art.get_local_copy()
                if new_model_path:
                    current_model = joblib.load(new_model_path)
                    current_model_id = last_task.id
                    
                    # Load scaler locally (ensure data/processed/scaler.pkl exists on Render)
                    if os.path.exists("data/processed/scaler.pkl"):
                        current_scaler = joblib.load("data/processed/scaler.pkl")
                    
                    print(f"API successfully updated to model ID: {current_model_id}")
        
        elif not last_task:
            print("No matching tasks found in project 'BTC_MLOps'.")
                
    except Exception as e:
        print(f"ClearML Sync Error: {e}")

async def model_check_loop():
    while True:
        load_latest_model()
        await asyncio.sleep(600)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(model_check_loop())

@app.get("/")
async def root():
    return {
        "message": "BTC Price Prediction API is live",
        "model_loaded": current_model is not None,
        "scaler_loaded": current_scaler is not None,
        "task_id": current_model_id
    }

@app.post("/predict")
async def predict(input_data: PredictionInput):
    if current_model is None:
        return {"error": "Model not loaded. ClearML sync in progress..."}
    
    try:
        # 1. Predict (Output is scaled)
        scaled_pred = current_model.predict([input_data.data])
        
        # 2. Inverse Transform to get actual USD price
        if current_scaler:
            # Reshape for scaler requirement (N, 1)
            raw_pred = current_scaler.inverse_transform(scaled_pred.reshape(-1, 1))
            final_price = float(raw_pred[0][0])
        else:
            final_price = float(scaled_pred[0]) # Fallback to scaled if scaler missing

        return {
            "prediction_usd": round(final_price, 2),
            "is_scaled": current_scaler is None
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)