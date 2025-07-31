import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(file_path, window_size=60, forecast_horizon=7, test_ratio=0.2):
    # Load cleaned dataset
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    close_prices = df[['close']].values

    # 1. Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(close_prices)

    # 2. Create sequences
    X, y = [], []
    for i in range(window_size, len(scaled_close) - forecast_horizon):
        X.append(scaled_close[i - window_size:i])
        y.append(scaled_close[i:i + forecast_horizon].flatten())

    X = np.array(X)
    y = np.array(y)

    # 3. Train/test split
    split_index = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print("✅ Feature prep complete:")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

    return X_train, y_train, X_test, y_test, scaler
