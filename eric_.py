from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import math
import jsonpickle
import numpy as np
from collections import deque

# Enhanced Z-score calculation with exponential weighting
def calculate_z_score(prices, lookback, alpha=0.1):
    """Calculate exponentially weighted Z-score with volatility adjustment"""
    if len(prices) < lookback or len(prices) < 2:
        return 0, 0  # Z-score, volatility
    
    # Use exponential weighting for more recent prices
    weights = np.array([(1 - alpha)**i for i in reversed(range(lookback))])
    weights /= weights.sum()
    
    weighted_prices = np.array(prices[-lookback:]) * weights
    weighted_mean = np.sum(weighted_prices)
    weighted_std = np.sqrt(np.sum(weights * (prices[-lookback:] - weighted_mean)**2))
    
    current_price = prices[-1]
    z_score = (current_price - weighted_mean) / (weighted_std + 1e-6)  # Prevent division by zero
    return z_score, weighted_std

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

class Trader:
    def __init__(self, 
                ink_limit=22,
                ink_liquidate_quantity=7,
                z_lookbacks=[20, 50, 100],  # Multiple lookback periods
                dynamic_thresholds=True):
        
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
        
        # Enhanced Z-score parameters
        self.z_lookbacks = z_lookbacks  # Multiple lookback periods for confirmation
        self.base_threshold = 1.5
        self.dynamic_thresholds = dynamic_thresholds
        self.volatility_window = deque(maxlen=50)  # Track recent volatility
        self.adaptive_multiplier = 1.0
        
        # Trend detection
        self.trend_filter_window = 50
        self.trend_slope = 0
        
        # Position sizing parameters
        self.max_z_position_ratio = 0.8  # Max position per Z-score signal
        self.position_decay = 0.95  # Position size decay for consecutive signals

    def calculate_dynamic_threshold(self):
        """Adjust thresholds based on recent volatility"""
        if len(self.volatility_window) < 10:
            return self.base_threshold
        
        recent_vol = np.mean(list(self.volatility_window)[-10:])
        long_term_vol = np.mean(self.volatility_window)
        
        # Increase thresholds during high volatility regimes
        if recent_vol > 1.2 * long_term_vol:
            return self.base_threshold * 1.3
        # Decrease during low volatility
        elif recent_vol < 0.8 * long_term_vol:
            return self.base_threshold * 0.7
        return self.base_threshold

    def calculate_trend_slope(self, prices):
        """Calculate price trend using linear regression"""
        if len(prices) < self.trend_filter_window:
            return 0
        
        x = np.arange(len(prices[-self.trend_filter_window:]))
        y = np.array(prices[-self.trend_filter_window:])
        A = np.vstack([x, np.ones(len(x))]).T
        self.trend_slope = np.linalg.lstsq(A, y, rcond=None)[0][0]
        return self.trend_slope

    def get_mean_reversion_orders(self, product, order_depth, position, prices) -> list[Order]:
        """Enhanced Z-score strategy with multiple confirmations"""
        orders = []
        
        # Calculate trend filter
        trend_slope = self.calculate_trend_slope(prices)
        trend_filter = 1 - min(abs(trend_slope)/0.5, 1)  # Scale from 0-1
        
        # Calculate multiple Z-scores
        z_scores = []
        volatilities = []
        for lookback in self.z_lookbacks:
            z, vol = calculate_z_score(prices, lookback)
            z_scores.append(z)
            volatilities.append(vol)
        
        # Update volatility window
        self.volatility_window.append(np.mean(volatilities))
        
        # Composite Z-score (weighted average)
        weights = [1/(lb**0.5) for lb in self.z_lookbacks]  # Favor shorter lookbacks
        composite_z = np.average(z_scores, weights=weights)
        
        # Dynamic threshold adjustment
        threshold = self.calculate_dynamic_threshold() if self.dynamic_thresholds else self.base_threshold
        
        # Calculate position sizing
        position_scale = min(
            abs(composite_z)/3.0,  # Scale with Z-score extremity
            self.max_z_position_ratio
        ) * (1 - abs(position)/self.LIMIT[product])  # Account for current position
        
        # Generate signals with trend filter
        # Inside the buy signal block (composite_z < -threshold)
        best_bid = max(order_depth.buy_orders.keys(), default=0)
        best_ask = min(order_depth.sell_orders.keys(), default=0)
        spread = best_ask - best_bid if best_ask and best_bid else 0

        if composite_z < -threshold and trend_filter < 0.7:
            # Aggressive buying in deep deviations
            buy_price = best_bid + 1 if spread > 2 else best_bid
            max_long = int(self.LIMIT[product] * position_scale * trend_filter)
            buy_quantity = min(max_long, self.LIMIT[product] - position)
            
            if buy_quantity > 0:
                orders.append(Order(product, int(buy_price), buy_quantity))
                # Add secondary order at better price for steeper discounts
                if composite_z < -threshold*1.5:
                    secondary_price = buy_price - max(1, int(spread*0.3))
                    secondary_qty = int(buy_quantity * 0.3)
                    orders.append(Order(product, secondary_price, secondary_qty))

        # Inside the sell signal block (composite_z > threshold)
        elif composite_z > threshold and trend_filter < 0.7:
            sell_price = best_ask - 1 if spread > 2 else best_ask
            max_short = -int(self.LIMIT[product] * position_scale * trend_filter)
            sell_quantity = max(-max_short, -self.LIMIT[product] - position)
            
            if sell_quantity < 0:
                orders.append(Order(product, int(sell_price), sell_quantity))
                # Add secondary order for steeper premiums
                if composite_z > threshold*1.5:
                    secondary_price = sell_price + max(1, int(spread*0.3))
                    secondary_qty = int(abs(sell_quantity) * 0.3)
                    orders.append(Order(product, secondary_price, -secondary_qty))
        return orders

    def squid_ink_hybrid_strategy(self, product, order_depth, position, prices, vwap_list, timespan) -> list[Order]:
        """Enhanced with Z-score liquidation logic"""
        orders = []
        
        if len(self.squid_ink_prices) >= 2:
            if prices[-1] < prices[-2]:
                self.squid_ink_recent_losses.append(1)
            else:
                self.squid_ink_recent_losses.clear()

            # Trigger drawdown mode if more than 3 consecutive losses
            if len(self.squid_ink_recent_losses) >= 3:
                self.squid_ink_drawdown_mode = True
        else:
            self.squid_ink_recent_losses.clear()
        
        # New Z-score based liquidation
        z_score, _ = calculate_z_score(prices, 50)
        # Replace the liquidation block with:
        if abs(z_score) < 0.5 and position != 0:
            best_bid = max(order_depth.buy_orders.keys(), default=0)
            best_ask = min(order_depth.sell_orders.keys(), default=0)
            liquidation_orders = []
            
            if position > 0:  # Long position
                liquidation_price = best_bid - 1 if best_bid else 0
                liquidation_quantity = min(position, self.ink_liquidate_quantity * 2)
                if liquidation_quantity > 0:
                    liquidation_orders.append(Order(product, liquidation_price, -liquidation_quantity))
            
            elif position < 0:  # Short position
                liquidation_price = best_ask + 1 if best_ask else 0
                liquidation_quantity = min(abs(position), self.ink_liquidate_quantity * 2)
                if liquidation_quantity > 0:
                    liquidation_orders.append(Order(product, liquidation_price, liquidation_quantity))
            
            orders.extend(liquidation_orders)

        
        # Enhanced mean reversion with multiple Z-scores
        mr_orders = self.get_mean_reversion_orders(product, order_depth, position, prices)
        orders.extend(mr_orders)
        
        # Replace the market making block with:
        if not mr_orders:
            best_bid = max(order_depth.buy_orders.keys(), default=0)
            best_ask = min(order_depth.sell_orders.keys(), default=0)
            
            # Calculate tight spread market making
            spread = best_ask - best_bid if best_ask and best_bid else 2
            mid_price = (best_bid + best_ask) / 2 if best_ask and best_bid else prices[-1]
            
            # Dynamic spread adjustment based on volatility
            vol_adjusted_spread = max(1, int(spread * (1 + np.mean(self.volatility_window)/100)))
            
            # Calculate inventory skew
            position_ratio = position / self.LIMIT[product]
            skew = 1 - abs(position_ratio)**0.5
            
            # Bid/ask prices with inventory adjustment
            bid_price = int(mid_price - vol_adjusted_spread/2 * (1 + skew))
            ask_price = int(mid_price + vol_adjusted_spread/2 * (1 + skew))
            
            # Order quantities with anti-snipe protection
            bid_quantity = min(5 + int(10 * (1 - position_ratio)), 15)
            ask_quantity = min(5 + int(10 * (1 + position_ratio)), 15)
            
            # Place orders with price improvement
            if bid_price > 0:
                orders.append(Order(product, bid_price, bid_quantity))
            if ask_price > 0:
                orders.append(Order(product, ask_price, -ask_quantity))

        return orders

    def run(self, state: TradingState):
        result = {}
        order_depths = state.order_depths
        positions = state.position
        current_positions = {product: positions.get(product, 0) for product in self.LIMIT.keys()}
        
        # Update all product positions
        for product in self.LIMIT:
            if product not in current_positions:
                current_positions[product] = 0

        # Squid Ink Strategy Execution
        if Product.SQUID_INK in order_depths:
            squid_order_depth = order_depths[Product.SQUID_INK]
            squid_ink_pos = current_positions[Product.SQUID_INK]
            
            # Calculate market metrics
            best_bid = max(squid_order_depth.buy_orders.keys(), default=None)
            best_ask = min(squid_order_depth.sell_orders.keys(), default=None)
            
            # Price tracking with error handling
            if best_bid and best_ask:
                mid_price = (best_bid + best_ask) / 2
                self.squid_ink_prices.append(mid_price)
                
                # Calculate volume-weighted metrics
                bid_volume = sum(squid_order_depth.buy_orders.values())
                ask_volume = sum(squid_order_depth.sell_orders.values())
                total_volume = bid_volume + ask_volume
                
                if total_volume > 0:
                    vwap = (
                        sum(p * q for p, q in squid_order_depth.buy_orders.items()) +
                        sum(p * q for p, q in squid_order_depth.sell_orders.items())
                    ) / total_volume
                    self.squid_ink_vwap.append(vwap)
            else:
                # Fallback to last known values if no orders
                mid_price = self.squid_ink_prices[-1] if self.squid_ink_prices else None
                vwap = self.squid_ink_vwap[-1] if self.squid_ink_vwap else None
            
            # Volatility calculation (20-period rolling)
            if len(self.squid_ink_prices) >= 20:
                recent_prices = self.squid_ink_prices[-20:]
                self.squid_ink_volatility.append(np.std(recent_prices))
            
            # Position tracking
            self.squid_ink_positions.append(squid_ink_pos)
            
            # Execute hybrid strategy
            squid_orders = self.squid_ink_hybrid_strategy(
                product=Product.SQUID_INK,
                order_depth=squid_order_depth,
                position=squid_ink_pos,
                prices=self.squid_ink_prices,
                vwap_list=self.squid_ink_vwap,
                timespan=20
            )
            result[Product.SQUID_INK] = squid_orders

        # Encode persistent data
        traderData = jsonpickle.encode({
            "squid_ink_prices": self.squid_ink_prices,
            "squid_ink_vwap": self.squid_ink_vwap,
            "squid_ink_volatility": self.squid_ink_volatility,
            "squid_ink_positions": self.squid_ink_positions,
            "prev_mid_prices": self.prev_mid_prices
        })
        
        conversions = 1
        return result, conversions, traderData