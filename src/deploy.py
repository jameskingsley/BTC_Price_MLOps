from pipeline import main_pipeline  # main flow
from prefect.schedules import Cron

if __name__ == "__main__":
    # This registers the flow with the server
    main_pipeline.serve(
        name="BTC-Daily-Retraining",
        # Schedule: Every night at midnight
        cron="0 0 * * *", 
        description="Daily data ingestion, training, and registration to ClearML.",
        tags=["production", "crypto"]
    )