# streamlit_app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------ UI Setup ------------------------
st.set_page_config(page_title="📈 Stock Forecast App", layout="centered")
st.title("📈 LSTM Stock Price Forecasting")
st.markdown("Upload a CSV file to forecast the next 7 business days using a pre-trained LSTM model.")

# ------------------------ File Upload ------------------------
uploaded_file = st.file_uploader("📤 Upload stock CSV", type=["csv"])

if uploaded_file:
    st.write("📄 File type detected:", uploaded_file.type)

    # Try to read and normalize CSV
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()  # Normalize column names
    except Exception as e:
        st.error(f"❌ Could not read CSV. Error: {e}")
        st.stop()

    # ------------------------ Preprocessing ------------------------
    if 'date' not in df.columns or 'close' not in df.columns:
        st.error("❌ CSV must contain 'Date' and 'Close' columns (case-insensitive).")
        st.stop()

    st.subheader("🔄 Cleaning & Processing Data...")

    try:
        # Convert date format like '31-Jul-25'
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

        # Clean commas in close column like '4,533.90'
        df['close'] = df['close'].astype(str).str.replace(',', '', regex=False)
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        # Drop bad rows
        df = df.dropna(subset=['date', 'close'])
        df.sort_values('date', inplace=True)

        # Preview cleaned data
        st.write(f"✅ Cleaned rows: {len(df)}")
        st.dataframe(df.tail())

        # Ensure at least 60 rows
        if len(df) < 60:
            st.warning("❗ Not enough data after cleaning. Need at least 60 rows.")
            st.stop()

        # ------------------------ Prepare for Prediction ------------------------
        close_data = df['close'].values.reshape(-1, 1)

        # Scale data
        scaler = MinMaxScaler()
        scaled_close = scaler.fit_transform(close_data)

        # Get last 60 days
        last_60_days = scaled_close[-60:]
        X_input = last_60_days.reshape(1, 60, 1)

        # ------------------------ Load Model ------------------------
        model_path = os.path.join("models", "lstm_msft_model.h5")
        if not os.path.exists(model_path):
            st.error("❌ Pretrained model not found at 'models/lstm_msft_model.h5'")
            st.stop()

        model = load_model(model_path, compile=False)

        # Predict
        predicted_scaled = model.predict(X_input)
        predicted_prices = scaler.inverse_transform(predicted_scaled).flatten()

        # ------------------------ Forecast ------------------------
        last_date = df['date'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date, periods=8, freq='B')
        forecast_prices = np.concatenate([[df['close'].iloc[-1]], predicted_prices])

        st.subheader("📈 Next 7-Day Forecast")
        for date, price in zip(forecast_dates[1:], predicted_prices):
            st.write(f"{date.date()} → ₹{price:.2f}")

        # ------------------------ Plot ------------------------
        st.subheader("📊 Actual vs Forecast")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['date'][-100:], df['close'][-100:], label='Actual Price (Last 100 Days)', color='blue')
        ax.plot(forecast_dates, forecast_prices, label='Forecast (Next 7 Days)', color='orange', linestyle='--', marker='o')

        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.set_title("Stock Price Forecast")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
