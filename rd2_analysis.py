import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Product list
products = ['CROISSANTS', 'JAMS', 'DJEMBES', 'PICNIC_BASKET1', 'PICNIC_BASKET2']

# File paths and days
days = [-2, -1, 0]
price_files = {
    -2: 'prices_round_2_day_1.csv',
    -1: 'prices_round_2_day_-1.csv',
    0: 'prices_round_2_day_0.csv'
}

# Load data
price_dfs = {product: [] for product in products}
for day in days:
    df = pd.read_csv(price_files[day], delimiter=';')
    df['timestamp'] = df['timestamp'] / 1000
    df['day'] = day

    for product in products:
        prod_df = df[df['product'] == product].copy()
        prod_df['mid_price'] = (prod_df['bid_price_1'] + prod_df['ask_price_1']) / 2
        prod_df = prod_df[['timestamp', 'mid_price', 'day']].copy()
        prod_df['product'] = product
        price_dfs[product].append(prod_df)

# Combine all mid-price data into one DataFrame
all_mid = pd.concat([pd.concat(price_dfs[product]) for product in products])
all_mid = all_mid.sort_values(['timestamp'])

# Pivot to wide format for correlation
wide_df = all_mid.pivot_table(index='timestamp', columns='product', values='mid_price')
wide_df = wide_df.dropna()

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
corr = wide_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation of Product Mid-Prices Across All Days')
plt.tight_layout()
plt.show()

# ==== Spread Code (unchanged) ====

# Combine original full product data
full_data = {product: pd.concat(price_dfs[product]) for product in products}

# Compute synthetic + spread for each basket
spread_dfs = []

for day in days:
    c = full_data['CROISSANTS'][full_data['CROISSANTS']['day'] == day][['timestamp', 'mid_price']]
    j = full_data['JAMS'][full_data['JAMS']['day'] == day][['timestamp', 'mid_price']]
    d = full_data['DJEMBES'][full_data['DJEMBES']['day'] == day][['timestamp', 'mid_price']]
    
    for basket_name, recipe in {
        'PICNIC_BASKET1': {'CROISSANTS': 2, 'JAMS': 3},
        'PICNIC_BASKET2': {'CROISSANTS': 1, 'JAMS': 2, 'DJEMBES': 1}
    }.items():
        basket = full_data[basket_name][full_data[basket_name]['day'] == day][['timestamp', 'mid_price']]
        merged = pd.merge_asof(
            pd.merge_asof(
                pd.merge_asof(basket.sort_values('timestamp'), c.sort_values('timestamp'), on='timestamp', suffixes=('_basket', '_c')),
                j.sort_values('timestamp'), on='timestamp', suffixes=('', '_j')
            ),
            d.sort_values('timestamp'), on='timestamp', suffixes=('', '_d')
        )

        # Replace NaN with 0 if product not in recipe (e.g., DJEMBES not in PICNIC_BASKET1)
        merged['mid_price_djembe'] = merged['mid_price_d'].fillna(0)

        merged['synthetic'] = (
            recipe.get('CROISSANTS', 0) * merged['mid_price_c'] +
            recipe.get('JAMS', 0) * merged['mid_price'] +
            recipe.get('DJEMBES', 0) * merged['mid_price_djembe']
        )
        merged['spread'] = merged['mid_price_basket'] - merged['synthetic']
        merged['basket'] = basket_name
        merged['day'] = day

        spread_dfs.append(merged[['timestamp', 'spread', 'day', 'basket']])

# Combine and plot spread
spread_all = pd.concat(spread_dfs)

fig, axs = plt.subplots(nrows=2, figsize=(15, 8), sharex=True)
for i, basket_name in enumerate(['PICNIC_BASKET1', 'PICNIC_BASKET2']):
    ax = axs[i]
    for day in days:
        df = spread_all[(spread_all['day'] == day) & (spread_all['basket'] == basket_name)]
        ax.plot(df['timestamp'], df['spread'], label=f'Day {day}')
        ax.axhline(df['spread'].mean(), linestyle='--', alpha=0.6)
    
    ax.set_title(f'Spread for {basket_name}')
    ax.set_ylabel('Spread (Basket - Synthetic)')
    ax.legend()
    ax.grid(True)

plt.xlabel('Time (s)')
plt.tight_layout()
plt.suptitle('Spread of Picnic Baskets vs Synthetic Value', fontsize=16, y=1.02)
plt.show()
