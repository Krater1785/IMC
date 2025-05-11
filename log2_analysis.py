import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

log_lines = []
with open('18be91e1-83ed-4169-8880-b38bd2bff643_final.log', 'r') as f:
    for line in f:
        if line.strip() and not line.startswith(('day', 'timestamp')):  # Skip header/empty lines
            log_lines.append(line.strip().split(';'))

# Define columns (ensure they match your log structure)
columns = [
    'day', 'timestamp', 'product', 'bid_price_1', 'bid_volume_1', 
    'bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3',
    'ask_price_1', 'ask_volume_1', 'ask_price_2', 'ask_volume_2',
    'ask_price_3', 'ask_volume_3', 'mid_price', 'profit_and_loss'
]

# Create DataFrame and convert numeric columns
df = pd.DataFrame(log_lines, columns=columns)
numeric_cols = ['day', 'timestamp', 'bid_price_1', 'bid_volume_1', 'ask_price_1', 'ask_volume_1', 'mid_price']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')  # 'coerce' handles invalid values

# Read and prepare the data (same as before)
# First, let's calculate the synthetic basket value at each timestamp
def calculate_synthetic_basket(row):
    try:
        # Get all required products at this timestamp
        timestamp = row['timestamp']
        current_data = df[df['timestamp'] == timestamp]
        
        # PICNIC_BASKET1 = 1*PICNIC_BASKET2 + 2*CROISSANTS + 1*JAMS + 1*DJEMBES
        pb2 = current_data[current_data['product'] == 'PICNIC_BASKET2']['mid_price'].values[0]
        croissants = current_data[current_data['product'] == 'CROISSANTS']['mid_price'].values[0]
        jams = current_data[current_data['product'] == 'JAMS']['mid_price'].values[0]
        djembes = current_data[current_data['product'] == 'DJEMBES']['mid_price'].values[0]
        
        return pb2 + (2 * croissants) + jams + djembes
    except:
        return np.nan

# Apply to each PICNIC_BASKET1 row
basket_df = df[df['product'] == 'PICNIC_BASKET1'].copy()
basket_df['synthetic_value'] = basket_df.apply(calculate_synthetic_basket, axis=1)
basket_df['time_hours'] = basket_df['timestamp'] / (1_000_000 / 3600)

# Calculate the synthetic spread (arbitrage opportunity)
basket_df['synthetic_spread'] = basket_df['mid_price'] - basket_df['synthetic_value']

# Analysis of synthetic spread
mean_spread = basket_df['synthetic_spread'].mean()
std_spread = basket_df['synthetic_spread'].std()
basket_df['spread_z_score'] = (basket_df['synthetic_spread'] - mean_spread) / std_spread

# Plotting
plt.figure(figsize=(14, 10))

# 1. Synthetic Spread Over Time
plt.subplot(3, 1, 1)
plt.plot(basket_df['time_hours'], basket_df['synthetic_spread'], 
         label='PICNIC_BASKET1 - Synthetic Value', color='royalblue')
plt.axhline(mean_spread, color='red', linestyle='--', label=f'Mean Spread ({mean_spread:.2f})')
plt.title('Synthetic Arbitrage Spread Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('Spread Value')
plt.legend()
plt.grid()

# 2. Z-Scores of Spread
plt.subplot(3, 1, 2)
plt.plot(basket_df['time_hours'], basket_df['spread_z_score'], color='purple')
plt.axhline(0, color='black', linestyle='-', label='Mean (Z=0)')
plt.axhline(2, color='orange', linestyle='--', label='±2 Std Dev')
plt.axhline(-2, color='orange', linestyle='--')
plt.title('Z-Scores of Synthetic Spread')
plt.xlabel('Time (hours)')
plt.ylabel('Z-Score')
plt.legend()
plt.grid()

# 3. Highlight Significant Spreads
plt.subplot(3, 1, 3)
significant_spreads = basket_df[np.abs(basket_df['spread_z_score']) > 2]
plt.scatter(significant_spreads['time_hours'], significant_spreads['synthetic_spread'],
            color='red', label='Significant Spreads (|Z|>2)')
plt.plot(basket_df['time_hours'], basket_df['synthetic_spread'], 
         color='royalblue', alpha=0.3, label='Spread')
plt.title('Significant Arbitrage Opportunities')
plt.xlabel('Time (hours)')
plt.ylabel('Spread Value')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Statistics
print("\nSynthetic Spread Statistics:")
print(f"Maximum Spread (Overpriced): {basket_df['synthetic_spread'].max():.2f}")
print(f"Minimum Spread (Underpriced): {basket_df['synthetic_spread'].min():.2f}")
print(f"Average Spread: {mean_spread:.2f}")
print(f"Standard Deviation: {std_spread:.2f}")
print(f"Number of Significant Opportunities (|Z|>2): {len(significant_spreads)}")
print(f"Percentage of Time with |Z|>2: {len(significant_spreads)/len(basket_df)*100:.1f}%")

# Show some extreme examples
print("\nTop 5 Overpriced Periods:")
print(basket_df.nlargest(5, 'synthetic_spread')[['time_hours', 'mid_price', 'synthetic_value', 'synthetic_spread']])

print("\nTop 5 Underpriced Periods:")
print(basket_df.nsmallest(5, 'synthetic_spread')[['time_hours', 'mid_price', 'synthetic_value', 'synthetic_spread']])