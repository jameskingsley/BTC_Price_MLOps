import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prefect import task, flow
import os
import joblib

@task
def load_and_clean_data(path: str):
    """Load CSV and ensure numeric types"""
    df = pd.read_csv(path)
    
    # Convert Date column to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Explicitly convert columns to numeric to avoid DataError
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows that failed conversion or are empty
    df.dropna(subset=['Close'], inplace=True)
    return df

@task
def create_features(df: pd.DataFrame):
    """Add technical indicators for time-series forecasting"""
    df = df.copy()
    # SMA (Simple Moving Average)
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    
    # Drop rows with NaN values created by the rolling windows
    df.dropna(inplace=True)
    return df

@task
def scale_features(df: pd.DataFrame):
    """Scale data for LSTM/Neural Network compatibility"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Focus on scaling 'Close' for this prediction task
    scaled_values = scaler.fit_transform(df[['Close']].values)
    
    # Save scaler for inference in FastAPI
    os.makedirs("data/processed", exist_ok=True)
    joblib.dump(scaler, "data/processed/scaler.pkl")
    
    return scaled_values

@flow(name="BTC_Preprocessing")
def run_preprocessing():
    df = load_and_clean_data("data/raw/btc_raw.csv")
    df_with_features = create_features(df)
    scaled_data = scale_features(df_with_features)
    
    # Save as numpy array for fast loading during training
    np.save("data/processed/btc_scaled.npy", scaled_data)
    print("--- Preprocessing Complete: Scaler and Scaled Data saved ---")

if __name__ == "__main__":
    run_preprocessing()