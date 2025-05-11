import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# CONFIGURABLE PARAMETERS
Z_WINDOW = 300        # Rolling window for Z-score and NDR
BUY_THRESHOLD = -1.5  # Z-score below which to buy
SELL_THRESHOLD = 1.5  # Z-score above which to sell
PLOT_NDR = True       # Set to False to disable NDR plotting

# Load all three days
df = pd.read_csv('24342628-6a66-4c8d-958f-d4e49de80da1.csv',delimiter=';')

products = ['SQUID_INK']

# Analysis function with NDR
def analyze_with_ndr():
    for product in products:
            df['timestamp'] = df['timestamp'] / 1000  # ms to sec
            squid_df = df[df['product'] == 'SQUID_INK'].copy()

            # Compute mid-price
            squid_df['mid_price'] = (squid_df['ask_price_1'] + squid_df['bid_price_1']) / 2

            # Rolling calculations
            squid_df['rolling_mean'] = squid_df['mid_price'].rolling(window=Z_WINDOW).mean()
            squid_df['rolling_std'] = squid_df['mid_price'].rolling(window=Z_WINDOW).std()
            
            # Calculate Mean Absolute Deviation (MAD)
            squid_df['rolling_mad'] = squid_df['mid_price'].rolling(window=Z_WINDOW).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True
            )
            
            # Handle division by zero in MAD
            squid_df['rolling_mad'] = squid_df['rolling_mad'].replace(0, 1e-6)
            
            # Calculate metrics
            squid_df['z_score'] = (squid_df['mid_price'] - squid_df['rolling_mean']) / squid_df['rolling_std']
            squid_df['ndr'] = (squid_df['mid_price'] - squid_df['rolling_mean']) / squid_df['rolling_mad']

            # Plot configuration
            if PLOT_NDR:
                fig, axs = plt.subplots(3, 1, figsize=(16, 12), 
                                      sharex=True, 
                                      gridspec_kw={'height_ratios': [2, 1, 1]})
                fig.suptitle(f"SQUID_INK Analysis - Day ", fontsize=16)
            else:
                fig, axs = plt.subplots(2, 1, figsize=(16, 10), 
                                      sharex=True, 
                                      gridspec_kw={'height_ratios': [2, 1]})
                fig.suptitle(f"SQUID_INK Analysis - Day ", fontsize=16)

            # Price plot
            axs[0].plot(squid_df['timestamp'], squid_df['mid_price'], 
                      label='Mid Price', color='navy', linewidth=1)
            axs[0].plot(squid_df['timestamp'], squid_df['rolling_mean'],
                      label=f'{Z_WINDOW}-period Mean', color='darkorange', linestyle='--')
            axs[0].set_ylabel("Price Level")
            axs[0].grid(True)
            axs[0].legend()

            # Z-score plot
            axs[1].plot(squid_df['timestamp'], squid_df['z_score'], 
                      label='Z-Score', color='green')
            axs[1].axhline(BUY_THRESHOLD, color='red', linestyle='--', label='Buy Threshold')
            axs[1].axhline(SELL_THRESHOLD, color='blue', linestyle='--', label='Sell Threshold')
            axs[1].axhline(0, color='black', linestyle=':')
            axs[1].set_ylabel("Z-Score")
            axs[1].grid(True)
            axs[1].legend()

            # NDR plot if enabled
            if PLOT_NDR:
                axs[2].plot(squid_df['timestamp'], squid_df['ndr'], 
                          label='NDR', color='purple')
                axs[2].axhline(0, color='black', linestyle=':')
                axs[2].set_ylabel("Normalized Deviation Ratio")
                axs[2].set_xlabel("Timestamp")
                axs[2].grid(True)
                axs[2].legend()

            plt.tight_layout()
            plt.show()

# Statistical analysis functions
def print_statistical_analysis(df):
    print("\nStatistical Analysis:")
    print(f"Average Z-Score: {df['z_score'].mean():.2f}")
    print(f"Average NDR: {df['ndr'].mean():.2f}")
    print(f"Z-Score Volatility: {df['z_score'].std():.2f}")
    print(f"NDR Volatility: {df['ndr'].std():.2f}")
    print(f"Max Z-Score: {df['z_score'].max():.2f}")
    print(f"Min Z-Score: {df['z_score'].min():.2f}")

# Run analysis
if __name__ == "__main__":
    analyze_with_ndr()
    print_statistical_analysis(files)
