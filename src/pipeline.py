import sys
import os
from prefect import flow

# This ensures Python can see ingestion.py, preprocess.py, and train.py 
# inside the src folder when running from the root.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the flows from your existing scripts
from ingestion import run_ingestion
from preprocess import run_preprocessing
from train import run_training

@flow(name="BTC_Full_MLOps_Pipeline", log_prints=True)
def main_pipeline():
    """
    The Master Entry Point. 
    It runs the three stages of the ML lifecycle in order.
    """
    print("[START] Initiating Full BTC MLOps Cycle...")
    
    # 1. DATA INGESTION
    # Downloads BTC data and saves it to data/raw/btc_raw.csv
    print("\n Phase 1: Ingesting Raw Data...")
    run_ingestion()
    
    # 2. DATA PREPROCESSING
    # Cleans data and saves data/processed/btc_scaled.npy & scaler.pkl
    print("\nPhase 2: Preprocessing and Scaling...")
    run_preprocessing()
    
    # 3. MODEL TRAINING
    # Trains the model and pushes it to the ClearML Registry
    print("\nPhase 3: Training & Model Registration...")
    run_training()
    
    print("\n[SUCCESS] Pipeline complete! Your model is now live in ClearML.")

if __name__ == "__main__":
    main_pipeline()