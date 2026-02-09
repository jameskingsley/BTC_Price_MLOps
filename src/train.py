import numpy as np
import os
from clearml import Task, OutputModel
from prefect import task, flow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

@task
def load_processed_data():
    """Load the scaled data created in the previous step"""
    # allow_pickle=True is necessary for certain numpy configurations
    data = np.load("data/processed/btc_scaled.npy", allow_pickle=True)
    return data

@task
def train_model(data):
    """Simple Sliding Window training"""
    # Create sequences: Use last 10 days to predict today
    X, y = [], []
    window = 10
    
    # Ensure data is 1D for the sliding window if it was saved as (N, 1)
    data_series = data.flatten()
    
    for i in range(len(data_series) - window):
        X.append(data_series[i:i+window])
        y.append(data_series[i+window])
    
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize ClearML Task
    cl_task = Task.init(project_name="BTC_MLOps", task_name="Train_RandomForest")
    
    # Model Training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("--- Training Model ---")
    model.fit(X_train, y_train)
    
    # Log Metrics
    score = model.score(X_test, y_test)
    cl_task.get_logger().report_single_value(name="R2_Score", value=score)
    print(f"Model R2 Score: {score}")
    
    # Save Model Locally
    os.makedirs("models", exist_ok=True)
    model_path = "models/btc_model.pkl"
    joblib.dump(model, model_path)
    
    # --- FIXED: Removed incompatible 'auto_delete_local' argument ---
    print("Uploading model to ClearML Registry...")
    output_model = OutputModel(task=cl_task, name="btc_prediction_model")
    output_model.update_weights(weights_filename=model_path)
    
    # Upload as an artifact as well (used by your API's artifacts.get logic)
    cl_task.upload_artifact(name="btc_prediction_model", artifact_object=model_path)
    
    # Ensure all data is sent to the server before closing
    cl_task.flush()
    cl_task.close()
    print("--- Training and Upload Complete ---")
    return model_path

@flow(name="BTC_Model_Training")
def run_training():
    data = load_processed_data()
    train_model(data)

if __name__ == "__main__":
    run_training()