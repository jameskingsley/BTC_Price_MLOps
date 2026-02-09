import sys
import os

# 1. Add the current directory (src) to the path so it finds pipeline.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 2. Import 'main_pipeline' from 'pipeline.py'
from pipeline import main_pipeline 

if __name__ == "__main__":
    # 3. Use the correctly imported name 'main_pipeline'
    main_pipeline.serve(
        name="BTC-Daily-Retraining",
        cron="0 0 * * *", 
        description="End-to-end MLOps: Ingests data, trains RF model, and pushes to ClearML.",
        tags=["production", "crypto", "mlops"],
        version="1.0.0"
    )