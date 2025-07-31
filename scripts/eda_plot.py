import pandas as pd
import matplotlib.pyplot as plt

# Load preprocessed data
df = pd.read_csv("data/MSFT_data.csv", index_col='date', parse_dates=True)

# 1. Calculate moving averages
df['SMA20'] = df['close'].rolling(window=20).mean()
df['SMA50'] = df['close'].rolling(window=50).mean()

# 2. Plot closing price with SMAs
plt.figure(figsize=(14, 6))
plt.plot(df['close'], label='Closing Price', linewidth=1.5)
plt.plot(df['SMA20'], label='SMA 20', linestyle='--')
plt.plot(df['SMA50'], label='SMA 50', linestyle=':')
plt.title('MSFT Closing Price with 20 & 50 Day Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
