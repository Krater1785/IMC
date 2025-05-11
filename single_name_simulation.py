import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', 1000)  # Set max rows to display
import matplotlib.pyplot as plt

# ——— PARAMETERS ———
round_val = 5
day = 2
prefix = "../data/"

# person & product of interest
person = "Pablo"
# product = "VOLCANIC_ROCK_VOUCHER_10250"
product = "VOLCANIC_ROCK"
if __name__ == "__main__":
    # file paths
    prices_file = f"prices_round_{round_val}_day_{day}.csv"
    trades_file = f"trades_round_{round_val}_day_{day}.csv"

    # ——— DATA LOAD ———
    prices = pd.read_csv(prices_file, sep=';')
    trades = pd.read_csv(trades_file, sep=';')
    mask_adequate_trade = trades["buyer"] != trades["seller"]
    trades = trades[mask_adequate_trade]

    # normalize column names
    prices = prices.rename(columns={'product': 'symbol'})
    # filter prices just for our product
    prices = prices[prices['symbol'] == product]
    trades = trades.rename(columns={'symbol': 'symbol'})

    # filter trades just for our person & product
    mask = (trades['symbol'] == product) & ((trades['buyer'] == person) | (trades['seller'] == person))
    trades_pp = trades[mask].set_index('timestamp')
    print(trades_pp)

    # ——— MERGE TRADES WITH PRICES & MARK TRADE TYPE ———
    # reset index to merge on timestamp
    trades_merged = trades_pp.reset_index().merge(
        prices[["timestamp", 'bid_price_1', 'ask_price_1']],
        left_on='timestamp', right_on = "timestamp", how='left'
    )
    # determine trade type: take, book, or unknown
    def classify_trade(row):
        if (row['buyer'] == person and row['price'] >= row['ask_price_1']) or \
           (row['seller'] == person and row['price'] <= row['bid_price_1']):
            return 'take'
        if (row['seller'] == person and row['price'] >= row['ask_price_1']) or \
           (row['buyer'] == person and row['price'] <= row['bid_price_1']):
            return 'book'
        return 'unknown'
    trades_merged['trade_type'] = trades_merged.apply(classify_trade, axis=1)
    # set index back to timestamp
    trades_pp = trades_merged.set_index('timestamp')

    # make sure prices are sorted
    prices = prices.sort_values('timestamp').set_index('timestamp')

    # prepare time series
    timestamps = prices.index.unique()

    # state variables
    position = 0
    last_mid = None
    cum_pnl = 0.0

    # storage
    ts_list = []
    bid_list = []
    ask_list = []
    pos_list = []
    pnl_list = []

    for ts in timestamps:
        row = prices.loc[ts]
        # extract level‑1 quotes (might need to handle multiple rows per ts but assuming one)
        bid = row['bid_price_1']
        ask = row['ask_price_1']
        mid = row['mid_price']

        # MARK‑TO‑MARKET P&L from mid change
        if last_mid is not None:
            diff_mid = mid - last_mid
            cum_pnl = position * diff_mid +cum_pnl

        # then handle any trades at this ts
        if ts in trades_pp.index:
            # could be multiple trades at same ts
            for _, tr in trades_pp.loc[[ts]].iterrows():
                qty = tr['quantity']
                price = tr['price']
                # trade P&L for this person
                trade_pnl = (mid - price) * qty
                if tr['seller'] == person:
                    trade_pnl *= -1
                    position -= qty
                else:
                    position += qty
                cum_pnl += trade_pnl

        # record
        ts_list.append(ts)
        bid_list.append(bid)
        ask_list.append(ask)
        pos_list.append(position)
        pnl_list.append(cum_pnl)

        last_mid = mid

    # final output
    print(f"== {person} – {product} ==")
    print(f"Final position: {position}")
    print(f"Final cumulative P&L: {cum_pnl:.2f}")

    # ——— PLOTTING ———
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    # 1) Bid & Ask
    axes[0].plot(ts_list, bid_list, label='Bid')
    axes[0].plot(ts_list, ask_list, label='Ask')
    axes[0].set_ylabel('Price')
    axes[0].set_title(f'{product} quotes')
    axes[0].legend()
    # Mark trades by type and role: take (x), book (o), unknown (^); buy-green, sell-red
    take_buy = trades_pp[(trades_pp['trade_type']=='take') & (trades_pp['buyer']==person)]
    take_sell = trades_pp[(trades_pp['trade_type']=='take') & (trades_pp['seller']==person)]
    book_buy = trades_pp[(trades_pp['trade_type']=='book') & (trades_pp['buyer']==person)]
    book_sell = trades_pp[(trades_pp['trade_type']=='book') & (trades_pp['seller']==person)]
    unk_buy = trades_pp[(trades_pp['trade_type']=='unknown') & (trades_pp['buyer']==person)]
    unk_sell = trades_pp[(trades_pp['trade_type']=='unknown') & (trades_pp['seller']==person)]
    axes[0].scatter(take_buy.index, take_buy['price'], marker='x', color='green', label='Take (buy)')
    axes[0].scatter(take_sell.index, take_sell['price'], marker='x', color='red', label='Take (sell)')
    axes[0].scatter(book_buy.index, book_buy['price'], marker='o', color='green', label='Book (buy)')
    axes[0].scatter(book_sell.index, book_sell['price'], marker='o', color='red', label='Book (sell)')
    axes[0].scatter(unk_buy.index, unk_buy['price'], marker='^', color='green', label='Unknown (buy)')
    axes[0].scatter(unk_sell.index, unk_sell['price'], marker='^', color='red', label='Unknown (sell)')
    axes[0].legend()

    # 2) Cumulative P&L
    axes[1].plot(ts_list, pnl_list)
    axes[1].set_ylabel('Cumulative P&L')
    axes[1].set_title(f'{person} P&L over time')

    # 3) Position
    axes[2].step(ts_list, pos_list, where='post')
    axes[2].set_ylabel('Position')
    axes[2].set_xlabel('Timestamp')
    axes[2].set_title(f'{person} Position over time')

    plt.tight_layout()
    plt.show()