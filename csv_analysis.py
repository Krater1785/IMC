import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# CONFIGURABLE PARAMETERS
Z_WINDOW = 300        # Rolling window for Z-score
BUY_THRESHOLD = -1.5  # Z-score below which to buy
SELL_THRESHOLD = 1.5  # Z-score above which to sell

days = [-2, -1, 0]
# Load all three days
files = {
    -2:"prices_round_1_day_-2.csv",
    -1:"prices_round_1_day_-1.csv",
    0:"prices_round_1_day_0.csv"
}
# Ensure you’re in the correct directory or update this path

products = ['SQUID_INK']
price_dfs = {product: [] for product in products}
# Filter only SQUID_INK
for day in days:
    df = pd.read_csv(files[day], delimiter=';')
    df['timestamp'] = df['timestamp'] / 1000  # ms to sec
    df['day'] = day

    for product in products:
        prod_df = df[df['product'] == product].copy()
    squid_df = df[df['product'] == 'SQUID_INK'].copy()

    # Compute mid-price
    squid_df['mid_price'] = (squid_df['ask_price_1'] + squid_df['bid_price_1']) / 2

    # Rolling mean and std
    squid_df['rolling_mean'] = squid_df['mid_price'].rolling(window=Z_WINDOW).mean()
    squid_df['rolling_std'] = squid_df['mid_price'].rolling(window=Z_WINDOW).std()

    # Z-score
    squid_df['z_score'] = (squid_df['mid_price'] - squid_df['rolling_mean']) / squid_df['rolling_std']

# Plot mid-price and z-score with signals
fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle("SQUID_INK Mid-Price and Z-score Strategy Signals", fontsize=16)

# 1. Price Chart
axs[0].plot(squid_df['timestamp'], squid_df['mid_price'], label='Mid Price', color='black', linewidth=1)
axs[0].set_ylabel("Mid Price")
axs[0].grid(True)
axs[0].legend()

# 2. Z-score + Thresholds
axs[1].plot(squid_df['timestamp'], squid_df['z_score'], label='Z-Score', color='blue')
axs[1].axhline(BUY_THRESHOLD, color='green', linestyle='--', label='Buy Threshold')
axs[1].axhline(SELL_THRESHOLD, color='red', linestyle='--', label='Sell Threshold')
axs[1].axhline(0, color='gray', linestyle=':')
axs[1].set_ylabel("Z-Score")
axs[1].set_xlabel("Timestamp")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
