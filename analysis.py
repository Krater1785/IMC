import pandas as pd
import matplotlib.pyplot as plt

# Days and filenames
days = [-2, -1, 0]
price_files = {
    -2: 'prices_round_1_day_-2.csv',
    -1: 'prices_round_1_day_-1.csv',
    0: 'prices_round_1_day_0.csv'
}

products = ['SQUID_INK', 'KELP']
price_dfs = {product: [] for product in products}

# Load and compute VWAP
for day in days:
    df = pd.read_csv(price_files[day], delimiter=';')
    df['timestamp'] = df['timestamp'] / 1000  # ms to sec
    df['day'] = day

    for product in products:
        prod_df = df[df['product'] == product].copy()
        
        # Estimate price and volume
        prod_df['mid_price'] = (prod_df['bid_price_1'] + prod_df['ask_price_1']) / 2
        prod_df['volume'] = (prod_df['bid_volume_1'] + prod_df['ask_volume_1']) / 2

        # Compute cumulative VWAP
        prod_df['cum_vol'] = prod_df['volume'].cumsum()
        prod_df['cum_vol_price'] = (prod_df['mid_price'] * prod_df['volume']).cumsum()
        prod_df['vwap'] = prod_df['cum_vol_price'] / prod_df['cum_vol']

        price_dfs[product].append(prod_df)

# Combine by product
full_data = {product: pd.concat(price_dfs[product]) for product in products}

# Plotting
fig, axs = plt.subplots(nrows=3, figsize=(15, 12), sharex=False)

for i, day in enumerate(days):
    ax = axs[i]
    ax.set_title(f'Day {day}')

    for product in products:
        df = full_data[product][full_data[product]['day'] == day]
        ax.plot(df['timestamp'], df['vwap'], label=f'{product} VWAP')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('VWAP')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.suptitle('VWAP of SQUID_INK and KELP for Days -2 to 0', fontsize=16, y=1.02)
plt.show()
