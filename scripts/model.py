from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm_model(input_shape=(60, 1), output_size=7):
    """
    Build and return a compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64))
    model.add(Dense(output_size))
    model.compile(optimizer='adam', loss='mse')
    return model
