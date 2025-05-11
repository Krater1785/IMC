from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Tuple, Any
import math
import json
import numpy as np
from statistics import NormalDist
from abc import abstractmethod

def calculate_sell_quantity(order_depth, target_price):
        bids = order_depth.buy_orders
        q = sum([-y for x, y in bids.items() if x >= target_price])
        return q
    
def calculate_buy_quantity(order_depth, target_price):
        asks = order_depth.sell_orders
        q = sum([-y for x, y in asks.items() if x <= target_price])
        return q

def get_lowest_ask_with_min_qty(order_depth, product, min_qty):
    if not hasattr(order_depth, 'sell_orders'):
        return None

    # sort sell orders by price (lowest first)
    sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
    for price, volume in sorted_asks:
        # Convention: sell order volumes might be negative so take absolute value.
        available = -volume if volume < 0 else volume
        if available >= min_qty:
            return price
    return None


def get_highest_bid_with_min_qty(order_depth, product, min_qty):
    if not hasattr(order_depth, 'buy_orders'):
        return None

    # sort bid orders by price in descending order (highest first)
    sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
    for price, volume in sorted_bids:
        if volume >= min_qty:
            return price
    return None

def calculate_artificial_bid_ask(state, artificial_product, min_qty = 10):
    # Initialize aggregated prices
    isWrong = False
    artificial_bid = 0.0
    artificial_ask = 0.0

    product_prices = {}
    for product, coef in artificial_product.items():
        # Retrieve order depth for this product (assumes state.order_depths is a dict).
        order_depth = state.order_depths.get(product)
        if not order_depth:
            isWrong = True
            continue  # Skip if there is no order book for the product

        basic_bid = get_highest_bid_with_min_qty(order_depth, product, min_qty*abs(coef))
        basic_ask = get_lowest_ask_with_min_qty(order_depth, product, min_qty*abs(coef))
        product_prices[product] = {
            'min_qty': min_qty,
            'basic_bid': basic_bid,
            'basic_ask': basic_ask
        }
        if basic_bid is None or basic_ask is None:
            isWrong = True
            continue
        if coef >= 0:
            artificial_bid += coef * basic_bid
            artificial_ask += coef * basic_ask
        else:
            artificial_bid += coef * basic_ask  # Use ask price for bid aggregation.
            artificial_ask += coef * basic_bid  # Use bid price for ask aggregation.

    return artificial_bid, artificial_ask, product_prices, isWrong

def clip(value, min_value, max_value):
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    else:
        return value

def price_to_position(price, max_position, max_spread):
    score = price / max_spread
    score = clip(score, -1.0, 1.0)
    position = int(round(score * max_position))
    return position


def fill_the_order(state:TradingState, product:str, quantity:int) -> Optional[Order]:
    order_depth = state.order_depths[product]
    if quantity > 0:
        # Buy
        price = max(order_depth.sell_orders.keys())
        if price is not None:
            return Order(product, price, quantity)
    else:
        # Sell
        price = min(order_depth.buy_orders.keys())
        if price is not None:
            return Order(product, price, quantity)
    return None

def add_order_for_proportions(state: TradingState, artificial_product: Dict[str, int], orders: Dict[str, list[Order]], main_product, additional_products):
    base = state.position.get(main_product,0)
    for prod in additional_products:
        expected_q = artificial_product[prod] * base
        actual_q = state.position.get(prod, 0)
        filled_order = fill_the_order(state, prod, expected_q-actual_q)
        orders[prod] = orders.get(prod, []) + [filled_order]
    return orders

def extract_strike(product_name):
    # Expected format: "VOLCANIC_ROCK_VOUCHER_XXXX"
    parts = product_name.split('_')
    if "VOUCHER" in parts:
        try:
            return float(parts[-1])
        except ValueError:
            return None
    return None

# Normal cumulative distribution function (CDF)
def norm_cdf(x):
    return NormalDist().cdf(x)

# Normal probability density function (PDF)
def norm_pdf(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x ** 2)

# Black-Scholes price for a European call option
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    return price

# Black-Scholes price for a European put option (using put-call parity)
def black_scholes_put_price(S, K, T, r, sigma):
    call_price = black_scholes_call_price(S, K, T, r, sigma)
    put_price = call_price - S + K * np.exp(-r * T)
    return put_price

# Compute Vega (sensitivity of option price with respect to sigma)
def black_scholes_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm_pdf(d1)

# Compute Black-Scholes Delta for calls (and for puts if needed)
def black_scholes_delta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm_cdf(d1)
    else:  # For a put option
        return norm_cdf(d1) - 1

# Implied volatility using the Newton–Raphson method
def implied_vol(market_price, S, K, T, r, option_type='call', tol=1e-5, max_iter=1000):
    # Set initial bounds for volatility
    low_sigma = 0.01
    high_sigma = 0.2

    for i in range(max_iter):
        mid_sigma = (low_sigma + high_sigma) / 2.0

        # Compute option price using the mid-point volatility estimate
        if option_type == 'call':
            price = black_scholes_call_price(S, K, T, r, mid_sigma)
        else:
            price = black_scholes_put_price(S, K, T, r, mid_sigma)

        diff = price - market_price

        # Check if the price difference is within tolerance
        if abs(diff) < tol:
            return mid_sigma
        # Adjust the bounds based on whether the computed price is lower or higher than the market price.
        # For call options, the price increases with higher sigma.
        if diff < 0:
            # If the computed price is too low, we need a higher sigma:
            low_sigma = mid_sigma
        else:
            # If the computed price is too high, lower the volatility:
            high_sigma = mid_sigma

    # If convergence was not reached within the maximum iterations,
    # return the mid-point of the final interval as the estimated implied volatility.
    return (low_sigma + high_sigma) / 2.0

def get_significant_ask(order_depth, volume_threshold: int = 15) -> int:
    lowest_ask = None
    for price, volume in order_depth.sell_orders.items():
        if abs(volume) > volume_threshold and (lowest_ask is None or price < lowest_ask):
            lowest_ask = price
    return lowest_ask

def get_significant_bid(order_depth, volume_threshold: int = 15) -> int:
    highest_bid = None
    for price, volume in order_depth.buy_orders.items():
        if abs(volume) > volume_threshold and (highest_bid is None or price > highest_bid):
            highest_bid = price
    return highest_bid

def probability_of_overlap(ask_iv, bid_iv)->float:
    m_ask = np.quantile(ask_iv,0.2)
    m_bid = np.quantile(bid_iv, 0.8)
    return np.mean(bid_iv>m_ask)+np.mean(ask_iv<m_bid)

def add_order_for_proportions(state: TradingState, artificial_product: Dict[str, int], orders: Dict[str, list[Order]], main_product, additional_products):
    base = state.position.get(main_product,0)
    for prod in additional_products:
        expected_q = artificial_product[prod] * base
        actual_q = state.position.get(prod, 0)
        filled_order = fill_the_order(state, prod, expected_q-actual_q)
        orders[prod] = orders.get(prod, []) + [filled_order]
    return orders

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    DJEMBES = "DJEMBES"
    JAMS = "JAMS"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

class Strategy:
    def __init__(self, symbol: str, limit: int):
        self.symbol = symbol
        self.limit = limit
        self._logs = {}
        
    def log(self, key: str, value: Any):
        self._logs[key] = value
        
    @abstractmethod
    def run(self, state: TradingState) -> Tuple[List[Order], int]:
        raise NotImplementedError()
        
    def save(self) -> Dict:
        return {}
        
    def load(self, data: Dict) -> None:
        pass
        
    def get_logs(self) -> Dict:
        return self._logs
        
    def clean_logs(self):
        self._logs = {}

class ResinStrategy(Strategy):
    def __init__(self):
        super().__init__(Product.RAINFOREST_RESIN, 50)
        self.prices = []
        self.vwap = []
        
    def run(self, state: TradingState) -> Tuple[List[Order], int]:
        position = state.position.get(self.symbol, 0)
        orders = self.hybrid_orders(state.order_depths[self.symbol], position)
        return orders, 0
        
    def hybrid_orders(self, order_depth, position):
        orders = []
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            filtered_ask = [p for p in order_depth.sell_orders if abs(order_depth.sell_orders[p]) >= 15]
            filtered_bid = [p for p in order_depth.buy_orders if abs(order_depth.buy_orders[p]) >= 15]
            mm_ask = min(filtered_ask) if filtered_ask else best_ask
            mm_bid = max(filtered_bid) if filtered_bid else best_bid
            
            mmmid_price = (mm_ask + mm_bid) / 2
            self.prices.append(mmmid_price)
            
            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * -order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume if volume != 0 else mmmid_price
            self.vwap.append({"vol": volume, "vwap": vwap})
            
            if len(self.vwap) > 10:
                self.vwap.pop(0)
            if len(self.prices) > 10:
                self.prices.pop(0)
                
            fair_value = mmmid_price
            buy_order_volume, sell_order_volume = self.take_best_orders(fair_value, 1, orders, order_depth, position, 0, 0, True, 20)
            buy_order_volume, sell_order_volume = self.clear_position_order(fair_value, orders, order_depth, position, buy_order_volume, sell_order_volume)
            
            aaf = [p for p in order_depth.sell_orders if p > fair_value + 1]
            bbf = [p for p in order_depth.buy_orders if p < fair_value - 1]
            baaf = min(aaf) if aaf else fair_value + 2
            bbbf = max(bbf) if bbf else fair_value - 2
            
            self.market_make(orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)
            
        return orders
        
    def take_best_orders(self,product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,) :
        position_limit = self.LIMIT[product]

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if (not prevent_adverse and best_ask <= fair_value - take_width) or (prevent_adverse and best_ask_amount <= adverse_volume and best_ask <= fair_value - take_width):
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, round(best_ask), quantity))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if (not prevent_adverse and best_bid >= fair_value + take_width) or (prevent_adverse and best_bid_amount <= adverse_volume and best_bid >= fair_value + take_width):
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, round(best_bid), -quantity))
                    sell_order_volume += quantity

        return buy_order_volume, sell_order_volume

    def clear_position_order(self,
        product: str,
        fair_value: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,):

        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value)
        fair_for_ask = round(fair_value)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0 and fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(-order_depth.sell_orders[fair_for_bid], -position_after_take)
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def market_make(self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,)  :

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, int(bid), buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, int(ask), -sell_quantity))

        return buy_order_volume, sell_order_volume
    
    def save(self):
        return {
            'prices': self.prices,
            'vwap': self.vwap
        }
        
    def load(self, data):
        self.prices = data.get('prices', [])
        self.vwap = data.get('vwap', [])

class KelpStrategy(Strategy):
    def __init__(self):
        super().__init__(Product.KELP, 50)
        self.prices = []
        self.vwap = []
        
    def run(self, state: TradingState) -> Tuple[List[Order], int]:
        position = state.position.get(self.symbol, 0)
        orders = self.hybrid_orders(state.order_depths[self.symbol], position)
        return orders, 0
        
    def take_best_orders(self,product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,) :
        position_limit = self.LIMIT[product]

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if (not prevent_adverse and best_ask <= fair_value - take_width) or (prevent_adverse and best_ask_amount <= adverse_volume and best_ask <= fair_value - take_width):
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, round(best_ask), quantity))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if (not prevent_adverse and best_bid >= fair_value + take_width) or (prevent_adverse and best_bid_amount <= adverse_volume and best_bid >= fair_value + take_width):
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, round(best_bid), -quantity))
                    sell_order_volume += quantity

        return buy_order_volume, sell_order_volume

    def clear_position_order(self,
        product: str,
        fair_value: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,):

        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value)
        fair_for_ask = round(fair_value)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0 and fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(-order_depth.sell_orders[fair_for_bid], -position_after_take)
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def market_make(self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,)  :

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, int(bid), buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, int(ask), -sell_quantity))

        return buy_order_volume, sell_order_volume
    
    def save(self):
        return {
            'prices': self.prices,
            'vwap': self.vwap
        }
        
    def load(self, data):
        self.prices = data.get('prices', [])
        self.vwap = data.get('vwap', [])

class SquidInkStrategy(Strategy):
    def __init__(self):
        super().__init__(Product.SQUID_INK, 50)
        self.prices = []
        self.vwap = []
        self._price_cache = None
        self._vwap_cache = {}
        self.performance_tracker = {
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'position_cost': 0.0,
            'peak_pnl': 0.0,
            'trade_history': [],
            'suspended': False
        }
        
    def run(self, state: TradingState) -> Tuple[List[Order], int]:
        if self.performance_tracker['suspended']:
            return [], 0
            
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        orders, current_price = self.squid_ink_adaptive_zscore(order_depth, position)
        
        # Update performance tracker
        if current_price is not None:
            own_trades = state.own_trades.get(self.symbol, [])
            new_trades = [t for t in own_trades if t.timestamp == state.timestamp]
            
            if new_trades:
                quantities = [t.quantity for t in new_trades]
                prices = [t.price for t in new_trades]
                trade_pnls = [(p - self.performance_tracker['position_cost']) * q for p, q in zip(prices, quantities)]
                self.performance_tracker['realized_pnl'] += sum(trade_pnls)
                
                total_cost = sum(p * q for p, q in zip(prices, quantities))
                total_qty = sum(quantities)
                current_qty = position - total_qty
                if current_qty != 0:
                    self.performance_tracker['position_cost'] = (
                        self.performance_tracker['position_cost'] * current_qty + total_cost) / position
                
            if position != 0:
                self.performance_tracker['unrealized_pnl'] = (
                    current_price - self.performance_tracker['position_cost']) * position
            else:
                self.performance_tracker['unrealized_pnl'] = 0.0
                
            total_pnl = self.performance_tracker['realized_pnl'] + self.performance_tracker['unrealized_pnl']
            self.performance_tracker['peak_pnl'] = max(self.performance_tracker['peak_pnl'], total_pnl)
            
            if self.performance_tracker['peak_pnl'] - total_pnl > 1000:
                self.performance_tracker['suspended'] = True
                
        return orders, 0
        
    def check_component_limits(self, product: str, order_quantity: int, current_position: int) -> int:
        if product not in self.LIMIT:
            return order_quantity  # if No limit defined for the product
        
        max_position = self.LIMIT[product]
        if order_quantity > 0:  # Buying
            adjusted_quantity = min(order_quantity, max_position - current_position)
        else:  # Selling
            adjusted_quantity = max(order_quantity, -max_position - current_position)
        
        return adjusted_quantity if adjusted_quantity != 0 else 0
    
    def _calculate_vwap(self, bid, ask, bid_vol, ask_vol):
        key = (bid, ask, bid_vol, ask_vol)
        if key not in self._vwap_cache:
            total_vol = bid_vol + ask_vol
            self._vwap_cache[key] = (bid * ask_vol + ask * bid_vol) / total_vol if total_vol else (bid + ask)/2
        return float(self._vwap_cache[key]) 

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
            if std_dev > 1e-6:
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
                    if len(self.performance_tracker['trade_history']) > 500:
                        self.performance_tracker['trade_history'] = self.performance_tracker['trade_history'][-500:]
                    self.log("len_history", len(self.performance_tracker['trade_history']))
                
                return orders, current_price
        
        return [], current_price
    
    def save(self):
        return {
            'prices': self.prices,
            'vwap': self.vwap,
            'performance': self.performance_tracker,
            'price_cache': self._price_cache,
            'vwap_cache': list(self._vwap_cache.items())
        }
        
    def load(self, data):
        self.prices = data.get('prices', [])
        self.vwap = data.get('vwap', [])
        self.performance_tracker = data.get('performance', self.performance_tracker)
        self._price_cache = data.get('price_cache')
        self._vwap_cache = dict(data.get('vwap_cache', []))

class BasketStrategy(Strategy):
    def __init__(self):
        super().__init__(Product.PICNIC_BASKET1, 60)
        self.max_box_diff = 60
        self.max_spread = 200
        self.min_box_diff_order = 15
        
    def run(self, state: TradingState) -> Tuple[List[Order], int]:
        orders_dict = self.order_box_diff(state)
        return orders_dict.get(self.symbol, []), 0
        
    def order_box_diff(self, state: TradingState):
        orders = {}
        artificial_product = {
            Product.PICNIC_BASKET1: 1, 
            Product.PICNIC_BASKET2: -1,
            Product.CROISSANTS: -2, 
            Product.JAMS: -1, 
            Product.DJEMBES: -1
        }
        
        artificial_bid, artificial_ask, products_info, isWrong = calculate_artificial_bid_ask(
            state, artificial_product, min_qty=self.min_box_diff_order)
            
        if isWrong:
            self.log("isWrong", True)
            return orders
            
        curr_position = state.position.get(self.symbol, 0)
        mid_price = (artificial_bid + artificial_ask) / 2
        
        up_position = price_to_position(-artificial_bid, self.max_box_diff, self.max_spread)
        down_position = price_to_position(-artificial_ask, self.max_box_diff, self.max_spread)
        
        if curr_position > up_position:
            additional_order = up_position - curr_position
        elif curr_position < down_position:
            additional_order = down_position - curr_position
        else:
            additional_order = 0

        if abs(additional_order) < self.min_box_diff_order:
            return orders
            
        if additional_order > 0:
            buy_price = products_info[self.symbol]['basic_ask']
            available_trades = calculate_buy_quantity(state.order_depths[self.symbol], buy_price)
            quantity = min(available_trades, additional_order)
            orders[self.symbol] = [Order(self.symbol, buy_price, quantity)]
        else:
            sell_price = products_info[self.symbol]['basic_bid']
            available_trades = calculate_sell_quantity(state.order_depths[self.symbol], sell_price)
            quantity = max(available_trades, additional_order)
            orders[self.symbol] = [Order(self.symbol, sell_price, quantity)]

        # Add component orders
        traded_products = [Product.CROISSANTS, Product.JAMS, Product.DJEMBES, Product.PICNIC_BASKET2]
        for constituent in traded_products:
            qty = artificial_product[constituent] * quantity
            if qty > 0:
                price = products_info[constituent]['basic_ask']
                orders[constituent] = [Order(constituent, price, qty)]
            else:
                price = products_info[constituent]['basic_bid']
                orders[constituent] = [Order(constituent, price, qty)]
                
        return orders

class VolcanicVoucherStrategy(Strategy):
    def __init__(self, symbol: str):
        super().__init__(symbol, 200)
        self.ask_ivol_history = []
        self.bid_ivol_history = []
        self.volcano_history_len = 50
        self.short_history_len = 10
        self.prob_lim = 0.05
        self.block_prob = True
        
    def run(self, state: TradingState) -> Tuple[List[Order], int]:
        orders = []
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            return [], 0
            
        pos = state.position.get(self.symbol, 0)
        limit_buy = self.limit - pos
        limit_sell = -self.limit - pos
        
        underlying_price = self.get_underlying_price(state)
        if underlying_price is None:
            return [], 0
            
        strike = extract_strike(self.symbol)
        
        # Compute IVs
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        
        if best_ask:
            iv_ask = implied_vol(best_ask, underlying_price, strike, 1, 0, 'call')
            self.ask_ivol_history.append(iv_ask)
            if len(self.ask_ivol_history) > self.volcano_history_len:
                self.ask_ivol_history.pop(0)
                
        if best_bid:
            iv_bid = implied_vol(best_bid, underlying_price, strike, 1, 0, 'call')
            self.bid_ivol_history.append(iv_bid)
            if len(self.bid_ivol_history) > self.volcano_history_len:
                self.bid_ivol_history.pop(0)
                
        # Check probability condition
        if (probability_of_overlap(self.ask_ivol_history, self.bid_ivol_history) < self.prob_lim 
            and self.block_prob):
            return [], 0
            
        # Compute target IVs
        if len(self.bid_ivol_history) >= self.short_history_len:
            target_iv_ask = np.quantile(self.bid_ivol_history[-self.short_history_len:], 0.7)
        else:
            return []
            
        if len(self.ask_ivol_history) >= self.short_history_len:
            target_iv_bid = np.quantile(self.ask_ivol_history[-self.short_history_len:], 0.3)
        else:
            return []
            
        if target_iv_bid > target_iv_ask:
            target_iv_bid = target_iv_ask = (target_iv_bid + target_iv_ask) / 2
            
        # Generate orders
        if target_iv_ask is not None and best_ask:
            bid_price = int(math.floor(black_scholes_call_price(underlying_price, strike, 1, 0, target_iv_ask)))
            q = calculate_buy_quantity(order_depth, bid_price)
            q = min(q, limit_buy)
            if q > 0:
                orders.append(Order(self.symbol, best_ask, q))
                
        if target_iv_bid is not None and best_bid:
            ask_price = int(math.ceil(black_scholes_call_price(underlying_price, strike, 1, 0, target_iv_bid)))
            q = calculate_sell_quantity(order_depth, ask_price)
            q = max(q, limit_sell)
            if q < 0:
                orders.append(Order(self.symbol, best_bid, q))
                
        return orders, 0
        
    def get_underlying_price(self, state: TradingState) -> Optional[float]:
        underlying_name = "VOLCANIC_ROCK"
        underlying_prices = []
        
        if underlying_name in state.order_depths:
            order_depth = state.order_depths[underlying_name]
            bid = get_significant_bid(order_depth, 1)
            ask = get_significant_ask(order_depth, 1)
            if bid: underlying_prices.append(bid)
            if ask: underlying_prices.append(ask)
            
        lower_strike = "VOLCANIC_ROCK_VOUCHER_9500"
        strike = extract_strike(lower_strike)
        if lower_strike in state.order_depths:
            order_depth = state.order_depths[lower_strike]
            coupon_ask = get_significant_ask(order_depth, 1)
            coupon_bid = get_significant_bid(order_depth, 1)
            if coupon_ask: underlying_prices.append(coupon_ask + strike)
            if coupon_bid: underlying_prices.append(coupon_bid + strike)
            
        return np.mean(underlying_prices) if underlying_prices else None
        
    def save(self):
        return {
            'ask_ivol': self.ask_ivol_history,
            'bid_ivol': self.bid_ivol_history
        }
        
    def load(self, data):
        self.ask_ivol_history = data.get('ask_ivol', [])
        self.bid_ivol_history = data.get('bid_ivol', [])

class MacaronStrategy(Strategy):
    def __init__(self):
        super().__init__("MAGNIFICENT_MACARONS", 30)
        
    def run(self, state: TradingState) -> Tuple[List[Order], int]:
        orders = []
        position = state.position.get(self.symbol, 0)
        
        observations = state.observations.conversionObservations.get(self.symbol)
        if not observations:
            return [], 0
            
        island_price = observations.askPrice + observations.transportFees + observations.importTariff
        order_depth = state.order_depths[self.symbol]
        
        if not order_depth.buy_orders:
            return [], 0
            
        best_bid = max(order_depth.buy_orders.keys())
        total_qty = -position
        
        if best_bid > island_price:
            for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                potential_profit = price - island_price
                if potential_profit < 0.5:
                    break
                    
                if total_qty < 20:
                    take = min(50 - total_qty, order_depth.buy_orders[price])
                    if take > 0:
                        orders.append(Order(self.symbol, price, -take))
                        total_qty += take
                else:
                    if potential_profit >= 2.0:
                        take = min(40 - total_qty, order_depth.buy_orders[price])
                        if take > 0:
                            orders.append(Order(self.symbol, price, -take))
                            total_qty += take
                    else:
                        break
                        
            if total_qty > 10:
                buy_qty = total_qty - 10
                orders.append(Order(self.symbol, round(island_price), buy_qty))
                
        # Market making orders
        min_price = int(round(island_price) + 1)
        available = 20 - total_qty
        
        if available > 10:
            mm_orders = [
                Order(self.symbol, min_price, -10),
                Order(self.symbol, min_price + 1, -(available - 10))
            ]
        elif available > 0:
            mm_orders = [Order(self.symbol, min_price, -available)]
        else:
            mm_orders = []
            
        orders.extend(mm_orders)
        
        conversions = self.determine_conversion_size(position)
        return orders, conversions
        
    def determine_conversion_size(self, position: int) -> int:
        island_lim = 10
        if position > 0:
            return 0
        elif position < -island_lim:
            return island_lim
        else:
            return -position

class Trader:
    def __init__(self, verbose=True):
        self.verbose = verbose

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.DJEMBES: 60,
            Product.JAMS: 350,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            "MAGNIFICENT_MACARONS": 30,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }

        self.strategies = {
            Product.RAINFOREST_RESIN: ResinStrategy(),
            Product.KELP: KelpStrategy(),
            Product.SQUID_INK: SquidInkStrategy(),
            Product.PICNIC_BASKET1: BasketStrategy(),
            Product.VOLCANIC_ROCK_VOUCHER_9750: VolcanicVoucherStrategy(Product.VOLCANIC_ROCK_VOUCHER_9750),
            Product.VOLCANIC_ROCK_VOUCHER_10000: VolcanicVoucherStrategy(Product.VOLCANIC_ROCK_VOUCHER_10000),
            Product.VOLCANIC_ROCK_VOUCHER_10250: VolcanicVoucherStrategy(Product.VOLCANIC_ROCK_VOUCHER_10250),
            Product.VOLCANIC_ROCK_VOUCHER_10500: VolcanicVoucherStrategy(Product.VOLCANIC_ROCK_VOUCHER_10500),
            "MAGNIFICENT_MACARONS": MacaronStrategy()
        }
        
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        orders = {}
        conversions = 0
        new_trader_data = {}

        # Load previous state
        if state.traderData:
            try:
                old_data = json.loads(state.traderData)
                for symbol, strategy in self.strategies.items():
                    if symbol in old_data:
                        strategy.load(old_data[symbol])
            except json.JSONDecodeError:
                pass

        # Process strategies
        for symbol, strategy in self.strategies.items():
            if symbol in state.order_depths:
                strategy_orders, strategy_conversions = strategy.run(state)
                if strategy_orders:
                    orders[symbol] = strategy_orders
                conversions += strategy_conversions
                new_trader_data[symbol] = strategy.save()
                
                if self.verbose:
                    logs = strategy.get_logs()
                    if logs:
                        print(f"{symbol} logs:", logs)
                    strategy.clean_logs()

        return orders, conversions, json.dumps(new_trader_data)
