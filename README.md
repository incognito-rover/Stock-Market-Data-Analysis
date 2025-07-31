# 📈 Stock Market Forecasting with LSTM

This project predicts future stock prices using an LSTM (Long Short-Term Memory) neural network built with TensorFlow/Keras. A Streamlit web app is included to visualize the 7-day forecast for any uploaded stock CSV file.

---
# 🚀 Project Journey
## ✅ Phase 1: Model Training

### 1. Data Loading
- Load `.csv` file (e.g., `MSFT_data.csv`)
- Format must include: `date`, `open`, `high`, `low`, `close`, `volume`

### 2. Preprocessing
- Convert `date` column to `datetime`
- Focus on `close` prices
- Compute `SMA20` and `SMA50` for visualization

### 3. Exploratory Data Analysis (EDA)
- Plot close price over time
- Overlay SMA20/SMA50

### 4. Feature Engineering
- Extract `close` column
- Scale using `MinMaxScaler`
- Create sequences: 60 previous days to predict next 7 days

### 5. Model Building (LSTM)
- Use TensorFlow/Keras
- Input shape: `(60, 1)` → Output: 7 future values

### 6. Model Training
- Train on historical sequences
- Evaluate using MSE/RMSE

### 7. Offline Prediction
- Use last 60 days to predict next 7 days
- Inverse transform predictions to original price scale

### 8. Plot Forecast vs Actual
- Plot actual prices and future forecast

---

## 🛠️ How to Run from Scratch

### 🔁 Train Your Own Model
1. Open `notebooks/02_LSTM_Model.ipynb`
2. Replace the loaded CSV file with your own data (minimum 60 rows)
3. Run all cells
4. It will generate a model file inside `models/` directory (e.g., `lstm_msft_model.h5`)

### 🧠 Use Better Training
- Tweak model structure, epochs, or sequences in the notebook
- Re-train and re-save the model for better accuracy

---

## 📟 How to Get Stock CSV Data

1. Visit [NSE India](https://www.nseindia.com)
2. Search for your desired stock (e.g., INFY, RELIANCE)
3. Go to **Historical Data** section
4. Set the date range to **at least 300 days**
5. Click **Download CSV**

Upload this CSV in the app to forecast the next 7 days.

---

## 🚀 Phase 2: Streamlit Web App

### Features
- Upload CSV file
- Automatically process and scale the data
- Use pre-trained model to forecast 7 future business days
- Interactive plot (Actual vs Forecast)

---

## 📄 Requirements for CSV File

To ensure compatibility with the model and web app, your uploaded CSV file must follow this format:

- ✅ Must contain a `Date` column  
  (e.g., `31-Jul-25`, accepted in most standard date formats)

- ✅ Must contain a `Close` price column  
  (i.e., the closing stock price for each day)

> 💡 The CSV should have at least **60 rows** of historical data to generate a valid forecast.

---

### Run the App
```bash
cd streamlit_app
streamlit run app.py


📦 Dependencies
Python 3.10+

TensorFlow / Keras

Streamlit

pandas, numpy, matplotlib, scikit-learn

Install all with:
pip install -r requirements.txt





