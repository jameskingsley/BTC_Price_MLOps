from fastapi import FastAPI, BackgroundTasks
from clearml import Task, Model
import joblib
import asyncio

app = FastAPI()

# Global variables to hold the loaded model and its ID
MODEL_PATH = "data/model.pkl"
current_model = joblib.load(MODEL_PATH)
current_model_id = None

def load_latest_model():
    global current_model, current_model_id
    try:
        # Find the latest successful training task in your project
        last_task = Task.get_task(
            project_name="BTC_Price_Prediction",
            task_name="BTC_Model_Training",
            status="completed"
        )
        
        if last_task and last_task.id != current_model_id:
            print(f"New model found! Task ID: {last_task.id}. Downloading...")
            # Get the output model from that task
            models = last_task.get_models()
            if models and 'output' in models:
                new_model_path = models['output'][0].get_local_copy()
                current_model = joblib.load(new_model_path)
                current_model_id = last_task.id
                print("API successfully updated to the latest model.")
    except Exception as e:
        print(f"Failed to check for new model: {e}")

# This runs every 10 minutes in the background
async def model_check_loop():
    while True:
        load_latest_model()
        await asyncio.sleep(600) # 600 seconds = 10 mins

@app.on_event("startup")
async def startup_event():
    # Start the background checking loop when the API starts
    asyncio.create_task(model_check_loop())

@app.post("/predict")
async def predict(data: list):
    # This always uses the 'current_model' which is updated by the background task
    prediction = current_model.predict([data])
    return {"prediction_usd": float(prediction[0])}