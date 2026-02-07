import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="BTC MLOps Predictor", page_icon="₿", layout="wide")

st.title("₿ Bitcoin Price Prediction Dashboard")
st.markdown("""
This app is part of a full **MLOps Pipeline**. It fetches live data from Yahoo Finance, 
sends it to a **FastAPI** backend, which uses a model registered in **ClearML**.
""")

# Sidebar for Status
st.sidebar.header("System Status")
st.sidebar.success("Model: Random Forest v1.0")
st.sidebar.info("Backend: FastAPI (Port 8000)")

# 1. Fetch Real Live Data
@st.cache_data(ttl=3600)  # Refresh data every hour
def get_live_data():
    try:
        # Get last 15 days to ensure we have enough after cleaning
        df = yf.download("BTC-USD", period="15d", interval="1d")
        
        # Handle yfinance MultiIndex issue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Get the 'Close' prices for the last 10 days
        closing_prices = df['Close'].tail(10).tolist()
        dates = df.index[-10:]
        return closing_prices, dates
    except Exception as e:
        st.error(f"Failed to fetch market data: {e}")
        return None, None

prices, dates = get_live_data()

if prices:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Market Trend (Last 10 Days)")
        # Create a nice Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines+markers', name='BTC Price'))
        fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Inference")
        st.write("Current Sequence:")
        st.dataframe(pd.DataFrame({"Date": dates, "Price": prices}).set_index("Date"))
        
        if st.button("Generate Forecast", use_container_width=True):
            with st.spinner('Calling FastAPI Model...'):
                try:
                    # Send the list of 10 prices to the API
                    response = requests.post("http://127.0.0.1:8000/predict", json=prices)
                    
                    if response.status_code == 200:
                        prediction = response.json()["prediction_usd"]
                        last_price = prices[-1]
                        diff = prediction - last_price
                        
                        st.metric(
                            label="Predicted Price (Next 24h)", 
                            value=f"${prediction:,.2f}", 
                            delta=f"${diff:,.2f}"
                        )
                        
                        if diff > 0:
                            st.success("Analysis: Bullish Trend Predicted ")
                        else:
                            st.warning("Analysis: Bearish Trend Predicted ")
                    else:
                        st.error("API Error: Is the FastAPI server running on port 8000?")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

else:
    st.warning("Waiting for data...")

st.divider()
st.caption("Disclaimer: This is an ML project for educational purposes. Do not use for financial trading.")