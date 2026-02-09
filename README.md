# BTC_Price_MLOps
* Bitcoin Live Predictor: End-to-End MLOps Pipeline
An automated Machine Learning operations (MLOps) project that fetches live Bitcoin data, trains a Random Forest regressor, tracks experiments, and serves real-time price predictions via a cloud-native architecture.

###### System Architecture
This project demonstrates a production-ready ML lifecycle:

* Data Orchestration (Prefect): Automates the daily ETL process—fetching data from Yahoo Finance, preprocessing, and triggering model retraining.

* Experiment Tracking & Registry (ClearML): Manages model versions, hyperparameters, and stores model artifacts (Pickle files) in a centralized cloud registry.

* Backend API (FastAPI): A high-performance REST API hosted on Render that pulls the latest "Published" model from ClearML to serve predictions.

* Frontend Dashboard (Streamlit): A user-facing web app hosted on Streamlit Cloud that visualizes historical trends and queries the API for tomorrow's forecast.

###### Tech Stack
Language: Python 3.9+

Orchestration: Prefect

ML Framework: Scikit-Learn (Random Forest)

Experiment Tracking: ClearML

API Framework: FastAPI / Uvicorn

Deployment: Render (Backend), Streamlit Cloud (Frontend)

Data Source: yfinance (Yahoo Finance API)

###### Project Structure
├── app/
│   ├── ui.py                 # Streamlit dashboard code
│   └── requirements.txt      # Dependencies for Streamlit Cloud
├── src/
│   ├── pipeline.py           # Prefect flow for ETL & training
│   ├── train.py              # Model training logic & ClearML integration
│   ├── preprocess.py         # Data cleaning and scaling
│   └── deploy.py             # Script to register Prefect deployments
├── data/
│   ├── raw/                  # (Gitignored) Raw BTC CSVs
│   └── processed/            # (Gitignored) Scalers and processed arrays
├── models/                   # Local model snapshots
├── requirements.txt          # Global project dependencies
└── README.md
###### Setup & Installation
* Clone the Repository:

* Bash
git clone https://github.com/jameskingsley/BTC_Price_MLOps.git

cd BTC_Price_MLOps

Install Dependencies:

* Bash
pip install -r requirements.txt

Configure Environment Variables: Create a .env file or export your keys

 for ClearML and Prefect:

* Bash
CLEARML_API_ACCESS_KEY="your_key"

CLEARML_API_SECRET_KEY="your_secret"

PREFECT_API_URL="your_prefect_url"

* Run the Pipeline:
* python src/pipeline.py

###### Model Performance
The model currently uses a Random Forest Regressor trained on a 10-day rolling window of BTC closing prices. Performance metrics (MAE, RMSE) are logged automatically to the ClearML dashboard after every retraining cycle.

###### Streamlit cloud Application:
https://btcpricemlops-6tasu8tjjeakv94sm8zehv.streamlit.app/

Developed by [James Kingsley] Looking to build scalable AI solutions? Let's connect