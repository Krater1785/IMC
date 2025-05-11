import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = "hi.csv"  # Change this to your actual file path
df = pd.read_csv(file_path, delimiter=';')

# Convert timestamp column to numeric
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df.dropna(subset=["timestamp"], inplace=True)
df["timestamp"] = df["timestamp"].astype(int)

# Convert bid-ask price columns to numeric
price_columns = ["bid_price_1", "ask_price_1"]
df[price_columns] = df[price_columns].apply(pd.to_numeric, errors="coerce")
df.dropna(subset=price_columns, inplace=True)

# Calculate bid-ask spread and mid-price
df["spread"] = df["ask_price_1"] - df["bid_price_1"]
df["mid_price"] = (df["ask_price_1"] + df["bid_price_1"]) / 2

# Separate products
df_resin = df[df["product"] == "RAINFOREST_RESIN"]
df_kelp = df[df["product"] == "KELP"]

# Plot bid-ask spread over time
plt.figure(figsize=(12, 6))
plt.plot(df_resin["timestamp"], df_resin["spread"], label="Rainforest Resin", color="blue")
plt.plot(df_kelp["timestamp"], df_kelp["spread"], label="Kelp", color="red")
plt.title("Bid-Ask Spread Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Spread")
plt.legend()
plt.show()

# Plot spread distribution
plt.figure(figsize=(12, 5))
sns.histplot(df_resin["spread"], bins=30, kde=True, color="blue", label="Rainforest Resin", alpha=0.6)
sns.histplot(df_kelp["spread"], bins=30, kde=True, color="red", label="Kelp", alpha=0.6)
plt.title("Bid-Ask Spread Distribution")
plt.xlabel("Spread")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Simple Mean Reversion Strategy on Spread
rolling_mean_spread = df["spread"].rolling(window=50).mean()
rolling_std_spread = df["spread"].rolling(window=50).std()
upper_threshold = rolling_mean_spread + 2 * rolling_std_spread
lower_threshold = rolling_mean_spread - 2 * rolling_std_spread

# Trading signals
df["buy_signal"] = df["spread"] > upper_threshold
df["sell_signal"] = df["spread"] < lower_threshold

# Plot signals
plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"], df["spread"], label="Spread", color="gray")
plt.plot(df["timestamp"][df["buy_signal"]], df["spread"][df["buy_signal"]], 'go', label="Buy Signal")
plt.plot(df["timestamp"][df["sell_signal"]], df["spread"][df["sell_signal"]], 'ro', label="Sell Signal")
plt.title("Mean Reversion Trading Signals Based on Spread")
plt.xlabel("Timestamp")
plt.ylabel("Spread")
plt.legend()
plt.show()