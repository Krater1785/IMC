from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import numpy as np

class Trader:
    POSITION_LIMITS = {"RAINFOREST_RESIN": 50, "KELP": 50}
    WINDOW = 10  # Rolling window for moving averages
    Z_SCORE_THRESHOLD = 1.5
    STOP_LOSS_Z = 2.5
    EMA_SHORT = 5  # Short-term EMA for Kelp
    EMA_LONG = 20  # Long-term EMA for Kelp
    EMA_ALPHA = 0.2  # Smoothing factor for EMA

    def __init__(self):
        self.price_history = {product: [] for product in self.POSITION_LIMITS}
        self.ema_values = {product: None for product in self.POSITION_LIMITS}
        self.ema_short = {"KELP": None}
        self.ema_long = {"KELP": None}

    def compute_metrics(self, product: str):
        prices = np.array(self.price_history[product])
        if len(prices) < self.WINDOW:
            return None, None, None, None

        if product == "KELP":
            if self.ema_short[product] is None:
                self.ema_short[product] = np.mean(prices[-self.EMA_SHORT:])
                self.ema_long[product] = np.mean(prices[-self.EMA_LONG:])
            else:
                self.ema_short[product] = (self.EMA_ALPHA * prices[-1] + (1 - self.EMA_ALPHA) * self.ema_short[product])
                self.ema_long[product] = (self.EMA_ALPHA * prices[-1] + (1 - self.EMA_ALPHA) * self.ema_long[product])
            
            return self.ema_short[product], self.ema_long[product], None, None
        
        # Mean Reversion for Rainforest Resin
        ma = np.mean(prices[-self.WINDOW:])
        std_price = np.std(prices[-self.WINDOW:])
        z_score = (prices[-1] - ma) / std_price if std_price > 0 else 0
        upper_band = ma + 2 * std_price
        lower_band = ma - 2 * std_price

        return ma, std_price, z_score, (upper_band, lower_band)

    def run(self, state: TradingState):
        orders = {}

        for product, depth in state.order_depths.items():
            if product not in self.POSITION_LIMITS:
                continue

            best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
            best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 0
            mid_price = (best_bid + best_ask) / 2
            self.price_history[product].append(mid_price)

            ma, std_price, z_score, bands = self.compute_metrics(product)
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS[product]
            product_orders = []

            if product == "RAINFOREST_RESIN":  # Mean Reversion Strategy
                if bands is None:
                    continue
                upper_band, lower_band = bands
                
                if z_score < -self.Z_SCORE_THRESHOLD and position < limit:  # Buy Signal
                    buy_size = min(limit - position, 10)
                    product_orders.append(Order(product, best_bid, buy_size))
                elif z_score > self.Z_SCORE_THRESHOLD and position > -limit:  # Sell Signal
                    sell_size = min(limit + position, 10)
                    product_orders.append(Order(product, best_ask, -sell_size))
            
            elif product == "KELP":  # Trend Following Strategy
                if ma is None:
                    continue
                ema_short, ema_long = ma, std_price  # Short EMA = ma, Long EMA = std_price
                
                if ema_short > ema_long and position < limit:  # Buy Signal
                    buy_size = min(limit - position, 10)
                    product_orders.append(Order(product, best_bid, buy_size))
                elif ema_short < ema_long and position > -limit:  # Sell Signal
                    sell_size = min(limit + position, 10)
                    product_orders.append(Order(product, best_ask, -sell_size))
            
            orders[product] = product_orders
        
        return orders, None, state.traderData