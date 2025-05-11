import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("hi.csv", delimiter=';')

# Convert timestamps to numerical indices (if needed)
df['timestamp'] = range(len(df))

# Filter data for each product
resin_data = df[df['product'] == 'RAINFOREST_RESIN']
kelp_data = df[df['product'] == 'KELP']

# Define moving average window
SMA_WINDOW = 50
EMA_SHORT = 20
EMA_LONG = 100

# Compute SMA and Bollinger Bands for Rainforest Resin
resin_data = resin_data.sort_values(by='timestamp')
resin_data['SMA'] = resin_data['mid_price'].rolling(window=SMA_WINDOW).mean()
resin_data['STD'] = resin_data['mid_price'].rolling(window=SMA_WINDOW).std()
resin_data['Upper_Band'] = resin_data['SMA'] + (2 * resin_data['STD'])
resin_data['Lower_Band'] = resin_data['SMA'] - (2 * resin_data['STD'])

# Compute EMA(20) and EMA(50) for Kelp
kelp_data = kelp_data.sort_values(by='timestamp')
kelp_data['EMA_20'] = kelp_data['mid_price'].ewm(span=EMA_SHORT, adjust=False).mean()
kelp_data['EMA_50'] = kelp_data['mid_price'].ewm(span=EMA_LONG, adjust=False).mean()

# Plot mid_price with SMA & Bollinger Bands for Rainforest Resin
plt.figure(figsize=(12, 6))

# Rainforest Resin
plt.subplot(2, 1, 1)
plt.plot(resin_data['timestamp'], resin_data['mid_price'], marker='o', linestyle='-', label='Mid Price', color='green')
plt.plot(resin_data['timestamp'], resin_data['SMA'], linestyle='--', label='SMA', color='black')
plt.fill_between(resin_data['timestamp'], resin_data['Upper_Band'], resin_data['Lower_Band'], color='gray', alpha=0.3, label='Bollinger Bands')
plt.xlabel("Timestamp")
plt.ylabel("Mid Price")
plt.title("Rainforest Resin - Mid Price, SMA & Bollinger Bands")
plt.legend()
plt.grid(True)

# Kelp
plt.subplot(2, 1, 2)
plt.plot(kelp_data['timestamp'], kelp_data['mid_price'], marker='s', linestyle='-', label='Mid Price', color='blue')
plt.plot(kelp_data['timestamp'], kelp_data['EMA_20'], linestyle='--', label='EMA 20', color='black')
plt.plot(kelp_data['timestamp'], kelp_data['EMA_50'], linestyle='-.', label='EMA 50', color='red')
plt.xlabel("Timestamp")
plt.ylabel("Mid Price")
plt.title("Kelp - Mid Price, EMA 20 & EMA 50")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
