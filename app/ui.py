import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import joblib
import numpy as np
import os

st.set_page_config(page_title="BTC MLOps Predictor", page_icon="🪙")

st.title("🪙 Bitcoin Live Predictor")
st.caption("Powered by FastAPI, ClearML, and Prefect")

# 1. Sidebar Info
st.sidebar.header("Pipeline Status")
st.sidebar.success("Backend: Render (FastAPI)")
st.sidebar.info("Model: RandomForest (ClearML)")

# 2. Fetch Live Data
@st.cache_data(ttl=3600) 
def get_live_data():
    # Use period='60d' to ensure we have enough data for a clean tail
    df = yf.download("BTC-USD", period="60d", interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df['Close'].tail(15)

try:
    prices = get_live_data()
    st.line_chart(prices, y_label="USD Price")

    # 3. Robust Path Handling for the Scaler
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    possible_paths = [
        os.path.join(BASE_DIR, "data", "processed", "scaler.pkl"),
        os.path.join(BASE_DIR, "models", "scaler.pkl"),
        os.path.join(BASE_DIR, "models", "scaler.pki")
    ]
    
    scaler_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if not scaler_path:
        st.error("Scaler file not found!")
        st.info(f"Searched in: {possible_paths}")
    else:
        scaler = joblib.load(scaler_path)
        
        # Prepare input: Take the last 10 days
        last_10_days = prices.tail(10).values.reshape(-1, 1)
        scaled_input = scaler.transform(last_10_days).flatten().tolist()

        if st.button("Predict Tomorrow's Price"):
            with st.spinner('Waiting for Render API (this can take 60s if the server is sleeping)...'):
                url = "https://btc-price-mlops.onrender.com/predict"
                payload = {"data": scaled_input}
                
                try:
                    response = requests.post(url, json=payload, timeout=70)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Safe extraction to prevent the 'prediction_usd' error
                        prediction = data.get("prediction_usd")
                        
                        if prediction is not None:
                            current_price = float(prices.iloc[-1])
                            diff = prediction - current_price
                            st.metric("Predicted Price (Tomorrow)", f"${prediction:,.2f}", f"${diff:,.2f}")
                            st.write(f"**Current Price (Today):** ${current_price:,.2f}")
                        else:
                            st.error("API worked, but 'prediction_usd' was missing from the response.")
                            st.json(data) # Show the raw response to see what's inside
                    else:
                        st.error(f"API returned error {response.status_code}")
                        st.write("Server response details:")
                        st.text(response.text)

                except Exception as e:
                    st.error(f"Connection Failed: {e}")

except Exception as e:
    st.error(f"Critical Error: {e}")