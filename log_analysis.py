import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file directly into a DataFrame
df = pd.read_csv('24342628-6a66-4c8d-958f-d4e49de80da1.csv')  # Replace with your actual CSV filename

# Ensure columns match expected structure (adjust if your CSV has different column names)
required_columns = [
    'day', 'timestamp', 'product', 'bid_price_1', 'bid_volume_1',
    'bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3',
    'ask_price_1', 'ask_volume_1', 'ask_price_2', 'ask_volume_2',
    'ask_price_3', 'ask_volume_3', 'mid_price', 'profit_and_loss'
]

# Convert numeric columns (handles commas/strings if present)
numeric_cols = ['day', 'timestamp', 'bid_price_1', 'bid_volume_1', 
               'ask_price_1', 'ask_volume_1', 'mid_price']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# --- Filter SQUID_INK Data ---
squid_df = df[df['product'] == 'SQUID_INK'].copy()

# Convert timestamp to hours (assuming 1M = max timestamp)
squid_df['time_hours'] = squid_df['timestamp'] / (1_000_000 / 3600)

# --- Analysis ---
squid_df['spread'] = squid_df['ask_price_1'] - squid_df['bid_price_1']
squid_df['mid_price_change'] = squid_df['mid_price'].diff()

# Calculate Z-Scores
mean_price = squid_df['mid_price'].mean()
std_price = squid_df['mid_price'].std()
squid_df['z_score'] = (squid_df['mid_price'] - mean_price) / std_price

# --- Save Analysis to New CSV ---
output_filename = 'squid_ink_analysis.csv'
squid_df.to_csv(output_filename, index=False)
print(f"Analysis saved to {output_filename}")

# --- Plotting (Optional) ---
if True:  # Set to True if you want visualizations
    plt.figure(figsize=(14, 10))
    
    # Price Over Time
    plt.subplot(3, 1, 1)
    plt.plot(squid_df['time_hours'], squid_df['mid_price'], color='blue')
    plt.title('SQUID_INK Price Over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Price')
    plt.grid()
    
    # Z-Scores
    plt.subplot(3, 1, 2)
    plt.plot(squid_df['time_hours'], squid_df['z_score'], color='purple')
    plt.axhline(0, color='black', linestyle='-')
    plt.axhline(2, color='orange', linestyle='--')
    plt.axhline(-2, color='orange', linestyle='--')
    plt.title('Z-Scores of Mid Price')
    plt.xlabel('Time (hours)')
    plt.ylabel('Z-Score')
    plt.grid()
    
    # Anomalies
    plt.subplot(3, 1, 3)
    anomalies = squid_df[np.abs(squid_df['z_score']) > 2]
    plt.scatter(anomalies['time_hours'], anomalies['mid_price'], color='red')
    plt.plot(squid_df['time_hours'], squid_df['mid_price'], color='blue', alpha=0.3)
    plt.title('Price Anomalies (|Z|>2)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Price')
    plt.grid()
    
    plt.tight_layout()
    plt.show()

# --- Print Statistics ---
print("\nSQUID_INK Statistics:")
print(f"Time Period: {squid_df['time_hours'].min():.2f} to {squid_df['time_hours'].max():.2f} hours")
print(f"Max Price: {squid_df['mid_price'].max():.2f}")
print(f"Min Price: {squid_df['mid_price'].min():.2f}")
print(f"Average Spread: {squid_df['spread'].mean():.2f}")
print(f"Standard Deviation: {std_price:.2f}")
print(f"Number of Anomalies (|Z|>2): {len(squid_df[np.abs(squid_df['z_score']) > 2])}")