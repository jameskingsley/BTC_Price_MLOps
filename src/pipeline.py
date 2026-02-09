import sys
import os
from prefect import flow

# Dynamic pathing: Ensures the 'src' folder is always in the python path
# regardless of where the script is executed from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "src"))

# Import the logic from scripts
from ingestion import run_ingestion
from preprocess import run_preprocessing
from train import run_training

@flow(
    name="BTC_Full_MLOps_Pipeline", 
    log_prints=True,
    retries=2,            # Automatically retry the whole pipeline if it fails
    retry_delay_seconds=60
)
def main_pipeline():
    """
    Master Orchestrator: Connects Ingestion, Preprocessing, and Training.
    """
    print("[START] Initiating Full BTC MLOps Cycle...")

    try:
        # 1. DATA INGESTION
        print("\nPhase 1: Ingesting Raw Data...")
        run_ingestion()
        
        # 2. DATA PREPROCESSING
        print("\nPhase 2: Preprocessing and Scaling...")
        run_preprocessing()
        
        # 3. MODEL TRAINING
        print("\nPhase 3: Training & Model Registration...")
        run_training()
        
        print("\n[SUCCESS] Pipeline complete! Your model is now live in ClearML.")
        
    except Exception as e:
        print(f"[FAILURE] Pipeline crashed during execution: {e}")
        raise  # Re-raise so Prefect marks the run as 'Failed'

if __name__ == "__main__":
    main_pipeline()