from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import math
import jsonpickle
import numpy as np

def calculate_sell_quantity(order_depth, target_price):
        bids = order_depth.buy_orders
        q = sum([-y for x, y in bids.items() if x >= target_price])
        return q
    
def calculate_buy_quantity(order_depth, target_price):
        asks = order_depth.sell_orders
        q = sum([-y for x, y in asks.items() if x <= target_price])
        return q

def calculate_deviation_index(prices, lookback):
        """Calculate how many standard deviations current price is from mean"""
        if len(prices) < lookback or np.std(prices[-lookback:]) == 0:
            return 0
        moving_avg = np.mean(prices[-lookback:])
        moving_std = np.std(prices[-lookback:])
        return (prices[-1] - moving_avg) / moving_std

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

class Trader:
    def __init__(self,ink_limit=22,ink_liquidate_quantity=7):
        self.resin_prices = []
        self.resin_vwap = []
        self.kelp_prices = []
        self.kelp_vwap = []
        
        self.squid_ink_prices = []
        self.squid_ink_vwap = []
        self.squid_ink_volatility = []
        self.squid_ink_positions = []  # track position history
        self.ink_liquidate_quantity = ink_liquidate_quantity
        self.ink_limit = ink_limit
        self.target_price = 10000
        self.mean_reversion_lookback = 50  # Period for mean/std calculation
        self.reversion_threshold = 2     # σ threshold for trading
        self.price_improvement = 0.005 
    
    # For drawdown detection on SQUID_INK
        self.squid_ink_recent_losses = []  # record consecutive price declines
        self.squid_ink_drawdown_mode = False

        self.prev_mid_prices = []

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50
        }
        self.volatility_window = 50
        self.vwap_window = 25  # for slope calculation
        self.vwap_threshold = 2
        self.max_trade_size = 50  

    def squid_ink_adaptive_zscore(self, product, order_depth, position, prices, vwap_list, timespan) -> list[Order]:
        orders = []

        # Strategy parameters
        lookback_period = 20
        base_z_buy_threshold = -2.8
        base_z_sell_threshold = 0.9
        base_ndr_buy_threshold = -2.8
        base_ndr_sell_threshold = 0.9
        exit_zone = 0.0
        position_limit = self.ink_limit

        if order_depth.sell_orders and order_depth.buy_orders:
            # Calculate mid-price and update data
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            ask_vol = -order_depth.sell_orders[best_ask]
            bid_vol = order_depth.buy_orders[best_bid]

            mid_price = (best_ask + best_bid) / 2
            prices.append(mid_price)

            # Update VWAP
            volume = ask_vol + bid_vol
            vwap = (best_bid * ask_vol + best_ask * bid_vol) / volume if volume != 0 else mid_price
            vwap_list.append({"vol": volume, "vwap": vwap})

            # Maintain window size
            if len(prices) > timespan:
                prices.pop(0)
            if len(vwap_list) > timespan:
                vwap_list.pop(0)

            # Calculate Z-score and NDR
            if len(prices) >= lookback_period:
                recent_prices = prices[-lookback_period:]
                mean_price = sum(recent_prices) / len(recent_prices)
                std_dev = (sum((price - mean_price) ** 2 for price in recent_prices) / len(recent_prices)) ** 0.5
                mad = sum(abs(price - mean_price) for price in recent_prices) / len(recent_prices)

                z_score = (mid_price - mean_price) / std_dev if std_dev > 0 else 0
                ndr = (mid_price - mean_price) / mad if mad > 0 else 0

                # Calculate trend strength using VWAP slope
                trend_strength = 0
                if len(vwap_list) >= 10:
                    trend_strength = (vwap_list[-1]["vwap"] - vwap_list[-10]["vwap"]) / (vwap_list[-1]["vwap"])

                # Adjust thresholds based on trend
                trend_adjustment = abs(trend_strength) * 2
                z_buy_threshold = base_z_buy_threshold - trend_adjustment
                z_sell_threshold = base_z_sell_threshold + trend_adjustment
                ndr_buy_threshold = base_ndr_buy_threshold - trend_adjustment
                ndr_sell_threshold = base_ndr_sell_threshold + trend_adjustment

                # Trading logic
                limit_buy = position_limit - position
                limit_sell = -position_limit - position

                # Filtered order book
                filtered_asks = [p for p in order_depth.sell_orders if abs(order_depth.sell_orders[p]) >= 15]
                filtered_bids = [p for p in order_depth.buy_orders if abs(order_depth.buy_orders[p]) >= 15]

                mm_ask = min(filtered_asks) if filtered_asks else best_ask
                mm_bid = max(filtered_bids) if filtered_bids else best_bid

                # Position sizing based on signal extremity
                intensity = min(1.0, max(abs(z_score), abs(ndr)) / max(abs(z_buy_threshold), abs(z_sell_threshold), abs(ndr_buy_threshold), abs(ndr_sell_threshold)))
                max_position_size = int(position_limit * 0.8 * intensity)

                # --- Buy Signal: Use BOTH Z-score and NDR for confirmation ---
                if z_score <= z_buy_threshold and ndr <= ndr_buy_threshold and limit_buy > 0:
                    buy_price = mm_bid + (2 if z_score < z_buy_threshold * 1.5 or ndr < ndr_buy_threshold * 1.5 else 1)
                    quantity = min(max_position_size, limit_buy, calculate_buy_quantity(order_depth, buy_price))
                    if quantity > 0:
                        orders.append(Order(product, buy_price, quantity))

                # --- Sell Signal: Use BOTH Z-score and NDR for confirmation ---
                if z_score >= z_sell_threshold and ndr >= ndr_sell_threshold and limit_sell < 0:
                    sell_price = mm_ask - (2 if z_score > z_sell_threshold * 1.5 or ndr > ndr_sell_threshold * 1.5 else 1)
                    quantity = max(-max_position_size, limit_sell, calculate_sell_quantity(order_depth, sell_price))
                    if quantity < 0:
                        orders.append(Order(product, sell_price, quantity))

                # Take profit when Z-score or NDR approaches mean
                if position > 0 and (z_score >= -exit_zone or ndr >= -exit_zone):
                    exit_price = best_ask
                    orders.append(Order(product, exit_price, -position))

                elif position < 0 and (z_score <= exit_zone or ndr <= exit_zone):
                    exit_price = best_bid
                    orders.append(Order(product, exit_price, -position))

                # Stop loss for risk management
                if (position > 0 and (z_score > z_sell_threshold * 1.5 or ndr > ndr_sell_threshold * 1.5)) or \
                (position < 0 and (z_score < z_buy_threshold * 1.5 or ndr < ndr_buy_threshold * 1.5)):
                    stop_price = mm_bid if position > 0 else mm_ask
                    orders.append(Order(product, stop_price, -position))

        return orders

        
    def run(self, state: TradingState):
        result = {}
        order_depths = state.order_depths
        positions = state.position
        current_positions = {product: positions.get(product, 0) for product in self.LIMIT.keys()}
        resin_pos = state.position.get(Product.RAINFOREST_RESIN, 0)
        kelp_pos = state.position.get(Product.KELP, 0)

        if Product.SQUID_INK in order_depths:
            squid_ink_pos = current_positions.get(Product.SQUID_INK, 0)
            squid_orders = self.squid_ink_adaptive_zscore(
                product=Product.SQUID_INK,
                order_depth=state.order_depths[Product.SQUID_INK],
                position=squid_ink_pos,
                prices=self.squid_ink_prices,
                vwap_list=self.squid_ink_vwap,
                timespan=20
            )
            result[Product.SQUID_INK] = squid_orders

        traderData = jsonpickle.encode({
            "squid_ink_prices": self.squid_ink_prices,
            "squid_ink_vwap": self.squid_ink_vwap,
            "squid_ink_volatility": self.squid_ink_volatility})
       

        conversions = 1
        return result, conversions, traderData