from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import numpy as np
import jsonpickle

def calculate_sell_quantity(order_depth, target_price):
    return sum(-y for x, y in order_depth.buy_orders.items() if x >= target_price)
    
def calculate_buy_quantity(order_depth, target_price):
    return sum(-y for x, y in order_depth.sell_orders.items() if x <= target_price)

class Product:
    SQUID_INK = "SQUID_INK"

class Trader:
    def __init__(self):
        self.LIMIT = {Product.SQUID_INK: 50}
        self.current_position = 0
        self.squid_ink_prices = []
        self.squid_ink_vwap = []
        self.last_price = None
        self._price_cache = None  # Tuple of (mean, std)
        self._vwap_cache = {}    # Cache for VWAP calculations
        self.performance_tracker = {
            'realized_pnl': 0.0,  # Ensure float type
            'unrealized_pnl': 0.0,
            'position_cost': 0.0,
            'peak_pnl': 0.0,
            'trade_history': [],
            'suspended': False
        }

    def check_component_limits(self, product: str, order_quantity: int, current_position: int) -> int:
        if product not in self.LIMIT:
            return order_quantity
        max_position = self.LIMIT[product]
        return min(order_quantity, max_position - current_position) if order_quantity > 0 else max(order_quantity, -max_position - current_position)

    def _calculate_vwap(self, bid, ask, bid_vol, ask_vol):
        key = (bid, ask, bid_vol, ask_vol)
        if key not in self._vwap_cache:
            total_vol = bid_vol + ask_vol
            self._vwap_cache[key] = (bid * ask_vol + ask * bid_vol) / total_vol if total_vol else (bid + ask)/2
        return float(self._vwap_cache[key])  # Convert to native float

    def squid_ink_adaptive_zscore(self, product, order_depth, position, prices, vwap_list, timespan) -> Tuple[List[Order], float]:
        if self.performance_tracker['suspended']:
            return [], None
        
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return [], None

        # Efficient min/max calculation
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        ask_vol = -order_depth.sell_orders[best_ask]
        bid_vol = order_depth.buy_orders[best_bid]
        current_price = float((best_ask + best_bid) / 2)  # Convert to native float
        
        # Update price history with fixed-size window
        if len(prices) >= timespan:
            prices.pop(0)
        prices.append(current_price)
        
        # Calculate VWAP using caching
        vwap = self._calculate_vwap(best_bid, best_ask, bid_vol, ask_vol)
        if len(vwap_list) >= timespan:
            vwap_list.pop(0)
        vwap_list.append({"vol": int(bid_vol + ask_vol), "vwap": float(vwap)})  # Ensure native types
        
        # Z-score calculation with caching
        if len(prices) >= 160:  # lookback_period
            if self._price_cache is None or len(prices) != len(self._price_cache[2]):
                price_window = np.array(prices[-200:], dtype=np.float64)
                mean_price = float(np.mean(price_window))
                std_dev = float(np.std(price_window))
                self._price_cache = (mean_price, std_dev, price_window.tolist())
            
            mean_price, std_dev, _ = self._price_cache
            if std_dev > 1e-6:  # Avoid division by tiny numbers
                z_score = float((current_price - mean_price) / std_dev)
                
                # Trend strength calculation
                trend_strength = 0.0
                if len(vwap_list) >= 10:
                    trend_strength = float((vwap_list[-1]["vwap"] - vwap_list[-10]["vwap"]) / vwap_list[-1]["vwap"])
                
                # Dynamic thresholds
                trend_adjustment = abs(trend_strength) * 0.5
                z_buy_threshold = -1.5 - trend_adjustment
                z_sell_threshold = 1.5 + trend_adjustment
                
                # Order generation
                orders = []
                position_limit = self.LIMIT[product]
                
                # Efficient order book processing
                mm_ask = min((p for p in order_depth.sell_orders if abs(order_depth.sell_orders[p]) >= 15), default=best_ask)
                mm_bid = max((p for p in order_depth.buy_orders if abs(order_depth.buy_orders[p]) >= 15), default=best_bid)
                
                # Position sizing
                intensity = min(1.0, abs(z_score) / max(abs(z_buy_threshold), abs(z_sell_threshold)))
                max_position_size = int(position_limit * 0.8 * intensity)
                
                # Trading signals
                if z_score <= z_buy_threshold and (limit_buy := position_limit - position) > 0:
                    buy_price = mm_bid + (2 if z_score < z_buy_threshold * 1.5 else 1)
                    quantity = min(max_position_size, limit_buy, calculate_buy_quantity(order_depth, buy_price))
                    if (quantity := self.check_component_limits(product, quantity, position)) > 0:
                        orders.append(Order(product, int(buy_price), quantity))
                
                if z_score >= z_sell_threshold and (limit_sell := -position_limit - position) < 0:
                    sell_price = mm_ask - (2 if z_score > z_sell_threshold * 1.5 else 1)
                    quantity = max(-max_position_size, limit_sell, calculate_sell_quantity(order_depth, sell_price))
                    if (quantity := self.check_component_limits(product, quantity, position)) < 0:
                        orders.append(Order(product, int(sell_price), quantity))
                
                # Exit and stop logic
                if position > 0 and z_score >= -0.0:
                    orders.append(Order(product, int(best_ask), -position))
                elif position < 0 and z_score <= 0.0:
                    orders.append(Order(product, int(best_bid), -position))
                
                if (position > 0 and z_score > z_sell_threshold * 1.5) or (position < 0 and z_score < z_buy_threshold * 1.5):
                    stop_price = float(mm_bid if position > 0 else mm_ask)
                    orders.append(Order(product, int(stop_price), -position))
                
                # Batch update trade history
                if orders and hasattr(self, 'last_timestamp'):
                    self.performance_tracker['trade_history'].extend({
                        'timestamp': self.last_timestamp,
                        'product': product,
                        'price': float(order.price),
                        'quantity': order.quantity,
                        'type': 'BUY' if order.quantity > 0 else 'SELL'
                    } for order in orders)
                
                return orders, current_price
        
        return [], current_price

    def run(self, state: TradingState):
        result = {}
        conversions = 1
        trader_data = ""
        
        if not hasattr(self, 'last_timestamp'):
            self.last_timestamp = state.timestamp
        
        current_positions = getattr(state, 'position', {})
        squid_ink_pos = current_positions.get(Product.SQUID_INK, 0)
        
        if not self.performance_tracker['suspended'] and Product.SQUID_INK in getattr(state, 'order_depths', {}):
            squid_orders, current_price = self.squid_ink_adaptive_zscore(
                product=Product.SQUID_INK,
                order_depth=state.order_depths[Product.SQUID_INK],
                position=squid_ink_pos,
                prices=self.squid_ink_prices,
                vwap_list=self.squid_ink_vwap,
                timespan=500
            )
            result[Product.SQUID_INK] = squid_orders
            
            # Optimized PnL calculation
            if current_price is not None:
                own_trades = getattr(state, 'own_trades', {}).get(Product.SQUID_INK, [])
                new_trades = [t for t in own_trades if getattr(t, 'timestamp', 0) == state.timestamp]
                
                if new_trades:
                    # Convert to native Python types immediately
                    quantities = [float(t.quantity) for t in new_trades]
                    prices = [float(t.price) for t in new_trades]
                    trade_pnls = [(p - self.performance_tracker['position_cost']) * q for p, q in zip(prices, quantities)]
                    self.performance_tracker['realized_pnl'] += sum(trade_pnls)
                    
                    total_cost = sum(p * q for p, q in zip(prices, quantities))
                    total_qty = sum(quantities)
                    current_qty = float(self.current_position)
                    if current_qty + total_qty != 0:
                        self.performance_tracker['position_cost'] = float(
                            (self.performance_tracker['position_cost'] * current_qty + total_cost) / (current_qty + total_qty))
                
                if squid_ink_pos != 0 and hasattr(self.performance_tracker, 'position_cost'):
                    if current_price is not None and self.performance_tracker['position_cost'] is not None:
                            try:
                                self.performance_tracker['unrealized_pnl'] = float(
                                    (current_price - self.performance_tracker['position_cost']) * squid_ink_pos
                                )
                            except (TypeError, ValueError) as e:
                                print(f"Error calculating unrealized PnL: {e}")
                                self.performance_tracker['unrealized_pnl'] = 0.0
                else:
                        self.performance_tracker['unrealized_pnl'] = 0.0
                
                self.current_position = int(squid_ink_pos)
                total_pnl = float(self.performance_tracker['realized_pnl'] + self.performance_tracker['unrealized_pnl'])
                self.performance_tracker['peak_pnl'] = max(float(self.performance_tracker['peak_pnl']), total_pnl)
                
                if self.performance_tracker['peak_pnl'] - total_pnl > 1000:
                    self.performance_tracker['suspended'] = True
        
        self.last_timestamp = state.timestamp
        
        # Prepare trader data with native Python types
        trader_data = jsonpickle.encode({
            'prices': [float(p) for p in self.squid_ink_prices[-100:]],
            'vwap': [{'vol': int(x['vol']), 'vwap': float(x['vwap'])} for x in self.squid_ink_vwap[-100:]],
            'performance': {
                'realized': float(self.performance_tracker['realized_pnl']),
                'unrealized': float(self.performance_tracker['unrealized_pnl']),
                'position': int(self.current_position),
                'cost': float(self.performance_tracker['position_cost']),
                'peak': float(self.performance_tracker['peak_pnl']),
                'suspended': bool(self.performance_tracker['suspended'])
            },
            'market': {
                'last_price': float(self.last_price) if self.last_price is not None else None,
                'spread': float(
                    min(state.order_depths[Product.SQUID_INK].sell_orders.keys()) - 
                    max(state.order_depths[Product.SQUID_INK].buy_orders.keys())
                ) if Product.SQUID_INK in getattr(state, 'order_depths', {}) else None
            }
        }, unpicklable=False)
        
        return result, conversions, trader_data