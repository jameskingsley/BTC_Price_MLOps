import yfinance as yf
import pandas as pd
from prefect import task, flow
import os

@task(retries=3, retry_delay_seconds=10)
def fetch_btc_data(symbol: str = "BTC-USD"):
    """Fetch historical BTC data and clean MultiIndex headers"""
    print(f"--- Downloading {symbol} data ---")
    df = yf.download(symbol, period="5y", interval="1d")
    
    # Flatten MultiIndex columns (yfinance specific fix)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Reset index to make 'Date' a column for clean CSV storage
    df = df.reset_index()
    
    if df.empty:
        raise ValueError("No data fetched from API.")
    
    return df

@task
def persist_data(df: pd.DataFrame, filename: str = "data/raw/btc_raw.csv"):
    """Saves the cleaned dataframe to local storage"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False) # index=False because Date is now a column
    print(f"--- Data saved locally at {filename} ---")
    return filename

@flow(name="BTC_Data_Ingestion")
def run_ingestion():
    data = fetch_btc_data()
    persist_data(data)
 
if __name__ == "__main__":
    run_ingestion()