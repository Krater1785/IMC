import pandas as pd
import numpy as np
from typing import List, Dict
from datamodel import OrderDepth, Order, TradingState
import jsonpickle  # Ensure traderData is serialized correctly

class Trader:
    POSITION_LIMITS = {"RAINFOREST_RESIN": 50, "KELP": 50}
    SMA_RANGE = (20, 50)  # Min/Max SMA windows
    EMA_SHORT = 20
    EMA_LONG = 50
    MAX_TRADE_SIZE = 10
    VOLATILITY_WINDOW = 30  # For SMA adjustment
    MAX_HISTORY = 200  # Limit the number of historical prices

    def __init__(self):
        self.price_history = {product: [] for product in self.POSITION_LIMITS}

    def compute_metrics(self, product: str):
        """Calculate SMA for Resin and EMAs for Kelp"""
        prices = self.price_history[product]
        
        if len(prices) < max(self.SMA_RANGE[1], self.EMA_LONG):
            return None, None, None  # Not enough data yet

        if product == "RAINFOREST_RESIN":
            # Dynamic SMA calculation with better volatility scaling
            changes = np.diff(prices[-self.VOLATILITY_WINDOW:])
            volatility = np.std(changes) if len(changes) > 1 else 0
            
            # Scale volatility effect reasonably (avoid extreme shrinkage)
            window = int(self.SMA_RANGE[1] - (volatility * 10))  
            window = max(self.SMA_RANGE[0], min(self.SMA_RANGE[1], window))

            sma = pd.Series(prices).rolling(window).mean().iloc[-1]
            return sma, None, None

        elif product == "KELP":
            ema_20 = pd.Series(prices).ewm(span=self.EMA_SHORT, adjust=False).mean().iloc[-1]
            ema_50 = pd.Series(prices).ewm(span=self.EMA_LONG, adjust=False).mean().iloc[-1]
            return None, ema_20, ema_50

    def run(self, state: TradingState):
        orders = {}

        for product, depth in state.order_depths.items():
            if product not in self.POSITION_LIMITS:
                continue

            # Get market prices safely
            if not depth.buy_orders or not depth.sell_orders:
                continue  # Skip if no valid order book

            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            self.price_history[product].append(mid_price)

            # Limit price history size
            self.price_history[product] = self.price_history[product][-self.MAX_HISTORY:]

            position = state.position.get(product, 0)
            product_orders = []

            # Get calculated metrics
            sma, ema_20, ema_50 = self.compute_metrics(product)

            # RAINFOREST_RESIN: SMA Strategy
            if product == "RAINFOREST_RESIN" and sma is not None:
                if mid_price < sma and position < self.POSITION_LIMITS[product]:
                    buy_size = min(self.MAX_TRADE_SIZE, self.POSITION_LIMITS[product] - position)
                    product_orders.append(Order(product, best_bid, buy_size))

                elif mid_price > sma and position > -self.POSITION_LIMITS[product]:
                    sell_size = min(self.MAX_TRADE_SIZE, abs(-self.POSITION_LIMITS[product] - position))
                    product_orders.append(Order(product, best_ask, -sell_size))

            # KELP: EMA Crossover Strategy
            elif product == "KELP" and ema_20 is not None and ema_50 is not None:
                if ema_20 > ema_50 and position < self.POSITION_LIMITS[product]:
                    buy_size = min(self.MAX_TRADE_SIZE, self.POSITION_LIMITS[product] - position)
                    product_orders.append(Order(product, best_bid, buy_size))

                elif ema_20 < ema_50 and position > -self.POSITION_LIMITS[product]:
                    sell_size = min(self.MAX_TRADE_SIZE, abs(-self.POSITION_LIMITS[product] - position))
                    product_orders.append(Order(product, best_ask, -sell_size))

            orders[product] = product_orders
        
        # Properly serialize traderData to maintain state
        trader_data = jsonpickle.encode(self.price_history)

        return orders, 0, trader_data  # ✅ Fixed return format
