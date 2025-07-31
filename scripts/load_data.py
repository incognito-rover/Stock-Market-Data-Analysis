import pandas as pd

# Load dataset
df = pd.read_csv("data/MSFT_data.csv")

# Show shape and columns
print("✅ Shape:", df.shape)
print("✅ Columns:", df.columns.tolist())

# Preview the data
print(df.head())
