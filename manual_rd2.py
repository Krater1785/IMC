import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go

def sea_turtle_profit(bid1, bid2, num_simulations=100000, avg_second_bid=295):
    profits = []
    for _ in range(num_simulations):
        # Generate reserve price from the specified distribution
        if np.random.rand() < 0.5:
            reserve_price = np.random.uniform(160, 200)
        else:
            reserve_price = np.random.uniform(250, 320)
        
        if bid1 >= reserve_price:
            profit = 320 - bid1
        elif bid2 >= reserve_price:
            # Check if our bid is above the archipelago average
            if bid2 >= avg_second_bid:
                profit = 320 - bid2
            else:
                # Apply probability scaling
                p = ((320 - avg_second_bid) / (320 - bid2)) ** 3
                if np.random.rand() < p:
                    profit = 320 - bid2
                else:
                    profit = 0
        else:
            profit = 0
        profits.append(profit)
    return np.mean(profits)

def find_optimal_bids(avg_second_bid=275):
    max_expected_profit = 0
    optimal_bid1 = None
    optimal_bid2 = None
    
    bid1_range = range(160, 321)
    bid2_range = range(160, 321)
    expected_profits = np.zeros((len(bid1_range), len(bid2_range)))
    
    for i, bid1 in enumerate(tqdm(bid1_range)):
        for j, bid2 in enumerate(bid2_range):
            if bid2 >= bid1:  # Assuming second bid should be >= first bid
                expected_profit = sea_turtle_profit(bid1, bid2, avg_second_bid=avg_second_bid)
                expected_profits[i, j] = expected_profit
                if expected_profit > max_expected_profit:
                    max_expected_profit = expected_profit
                    optimal_bid1 = bid1
                    optimal_bid2 = bid2
    
    return optimal_bid1, optimal_bid2, max_expected_profit, expected_profits

# You can adjust the average second bid parameter based on your estimate
optimal_bid1, optimal_bid2, max_expected_profit, expected_profits = find_optimal_bids(avg_second_bid=275)
print(f"Optimal first bid: {optimal_bid1}")
print(f"Optimal second bid: {optimal_bid2}")
print(f"Maximum expected profit per sea turtle: {max_expected_profit:.2f}")

# Create a 3D surface plot
fig = go.Figure(data=[go.Surface(z=expected_profits, x=list(range(160, 321)), y=list(range(160, 321)))])
fig.update_layout(
    title="Expected Profit for Different Bid Combinations",
    scene=dict(
        xaxis_title="Bid 1",
        yaxis_title="Bid 2",
        zaxis_title="Expected Profit"
    )
)
fig.show()