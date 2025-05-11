from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict,Optional
import math
import jsonpickle
import numpy as np
from statistics import NormalDist

def calculate_sell_quantity(order_depth, target_price):
        bids = order_depth.buy_orders
        q = sum([-y for x, y in bids.items() if x >= target_price])
        return q
    
def calculate_buy_quantity(order_depth, target_price):
        asks = order_depth.sell_orders
        q = sum([-y for x, y in asks.items() if x <= target_price])
        return q

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

def calc_traded_price(history: Dict[int, TradingState], product: str, n_back: int = -1) -> float:
    if len(history) == 0:
        return -1

    all_trades = []
    if n_back == -1 or len(history) < n_back:
        last_timestamp = min(history.keys())
    else:
        last_timestamp = sorted(history.keys())[-n_back]

    for timestamp, state in history.items():
        if timestamp >= last_timestamp and product in state.market_trades:
            all_trades.extend(state.market_trades[product])

    prices = [trade.price for trade in all_trades]
    abs_quantities = [abs(trade.quantity) for trade in all_trades]
    if sum(abs_quantities) == 0:
        return -1
    weighted_price = np.average(prices, weights=abs_quantities)
    return weighted_price

def get_lowest_ask_with_min_qty(order_depth, product, min_qty):
    """
    Returns the lowest ask price for the specified product in the order_depth
    that has an available volume of at least min_qty.

    Parameters:
      order_depth: An object with a 'sell_orders' attribute (a dictionary mapping price to volume).
      product: The product symbol (for informational consistency).
      min_qty: Minimal required available quantity.

    Returns:
      The lowest ask price meeting the minimal quantity requirement, or None if no such ask exists.
    """
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
    """
    Returns the highest bid price for the specified product in the order_depth
    that has an available volume of at least min_qty.

    Parameters:
      order_depth: An object with a 'buy_orders' attribute (a dictionary mapping price to volume).
      product: The product symbol (for informational consistency).
      min_qty: Minimal required available quantity.

    Returns:
      The highest bid price meeting the minimal quantity requirement, or None if no such bid exists.
    """
    if not hasattr(order_depth, 'buy_orders'):
        return None

    # sort bid orders by price in descending order (highest first)
    sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
    for price, volume in sorted_bids:
        if volume >= min_qty:
            return price
    return None


def calculate_artificial_bid_ask(state, artificial_product, min_qty = 10):
    """
    Calculate the aggregated artificial bid and ask prices for a basket of products
    by determining for each product the minimum quantity prices and then combining
    them. For products with a negative coefficient, the bid and ask prices are swapped.

    Args:
        state: The current trading state, expected to have an order_depths attribute
               (a dict mapping product symbols to their order books).
        artificial_product: A dict mapping product names to their coefficients.
                            For example:
                            {"PICNIC_BASKET1": 1, "PICNIC_BASKET2": -1,
                             "CROISSANTS": -2, "JAMS": -1, "DJEMBES": -1}

    Returns:
        A dict with the aggregated artificial bid and ask prices, and also details
        per product.
    """
    # Initialize aggregated prices
    isWrong = False
    artificial_bid = 0.0
    artificial_ask = 0.0

    # For debugging or further insight, store each product's computed prices.
    product_prices = {}

    for product, coef in artificial_product.items():
        # Retrieve order depth for this product (assumes state.order_depths is a dict).
        order_depth = state.order_depths.get(product)
        if not order_depth:
            isWrong = True
            continue  # Skip if there is no order book for the product

        # Use the absolute value of the coefficient as the min_qty parameter.
        # Calculate minimum quantity bid and ask prices using helper functions.
        basic_bid = get_highest_bid_with_min_qty(order_depth, product, min_qty*abs(coef))
        basic_ask = get_lowest_ask_with_min_qty(order_depth, product, min_qty*abs(coef))

        # Save the details of this product's pricing for inspection.
        product_prices[product] = {
            'min_qty': min_qty,
            'basic_bid': basic_bid,
            'basic_ask': basic_ask
        }

        # Skip this product if any of the prices couldn't be determined.
        if basic_bid is None or basic_ask is None:
            isWrong = True
            continue

        # For positive coefficients, use the standard pricing:
        #   - Basket bid: product's bid price (as you would sell a long position).
        #   - Basket ask: product's ask price (as you would buy a long position).
        if coef >= 0:
            artificial_bid += coef * basic_bid
            artificial_ask += coef * basic_ask
        else:
            # For negative coefficients (short positions) we "switch"
            # bid and ask:
            #   - When selling the basket, you will close the short by buying at the ask.
            #   - When buying the basket, you cover the short by selling at the bid.
            # Thus, for aggregation:
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

BASKET_COMPOSITION = {
    Product.PICNIC_BASKET2: {
        Product.CROISSANTS: 4,
        Product.JAMS: 2
    },
    Product.PICNIC_BASKET1: {
        Product.PICNIC_BASKET2: 1,
        Product.CROISSANTS: 2,
        Product.JAMS: 1,
        Product.DJEMBES: 1
    },
}

class Trader:
    def __init__(self, volcano_history_len = 50, block_prob = True, prob_lim = 0.05,verbose=True,min_box_diff_order=15):
        self.verbose = verbose
        self.min_box_diff_order = min_box_diff_order

        self.products = ['RAINFOREST_RESIN', 'KELP', 'SQUID_INK']
        self.resin_prices = []
        self.resin_vwap = []
        self.kelp_prices = []
        self.kelp_vwap = []
        
        self.squid_ink_prices = []
        self.squid_ink_vwap = []
        self.squid_ink_volatility = []
        self.target_price = 10000

        self.prev_mid_prices = []

        self.history_size = 1
        self.history = {}
        self.max_box_diff = 60
        self.max_spread = 200

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.DJEMBES: 60,
            Product.JAMS: 350,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }

        self.performance_tracker = {
            'starting_capital': 1000000,  # Example starting value
            'current_pnl': 0,
            'peak_pnl': 0,
            'max_drawdown_pct': 0.05,  # 5% max drawdown threshold
            'trade_history': [],
            'suspended': False
        }

        self.prob_lim = prob_lim
        self.block_prob = block_prob
        self.volcano_history_len = volcano_history_len
        self.short_history_len = volcano_history_len//5
        self.history_size = 10
        self.history = {}
        self.required_hedge_position = 0
        self.ask_ivol_history = {
            "VOLCANIC_ROCK_VOUCHER_9500": [],
            "VOLCANIC_ROCK_VOUCHER_9750": [],
            "VOLCANIC_ROCK_VOUCHER_10000": [],
            "VOLCANIC_ROCK_VOUCHER_10250": [],
            "VOLCANIC_ROCK_VOUCHER_10500": [],
        }
        self.bid_ivol_history = {
            "VOLCANIC_ROCK_VOUCHER_9500": [],
            "VOLCANIC_ROCK_VOUCHER_9750": [],
            "VOLCANIC_ROCK_VOUCHER_10000": [],
            "VOLCANIC_ROCK_VOUCHER_10250": [],
            "VOLCANIC_ROCK_VOUCHER_10500": [],
        }
        self.trading_coupons = ["VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
                                "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250",
                                "VOLCANIC_ROCK_VOUCHER_10500"]


        self.volatility_window = 20
        self.vwap_window = 5  # for slope calculation
        self.vwap_threshold = 1.2

        self.spread_data = {
            Product.PICNIC_BASKET1: {"spread_history": []},
            Product.PICNIC_BASKET2: {"spread_history": []},
        }

        self.params = {
            Product.SPREAD: {
                "spread_std_window": 135,
                "zscore_threshold": 5.2,
                "default_spread_mean": 0,
                "target_position": 60,
            }

        }
        self.position = {
            'RAINFOREST_RESIN': 0,
            'KELP': 0,
            'SQUID_INK': 0,
            'CROISSANTS': 0,
            'JAMS': 0,
            'DJEMBES': 0,
            'PICNIC_BASKET1': 0,
            'PICNIC_BASKET2': 0,
            'VOLCANIC_ROCK': 0,
            "VOLCANIC_ROCK_VOUCHER_9500": 0,
            "VOLCANIC_ROCK_VOUCHER_9750": 0,
            "VOLCANIC_ROCK_VOUCHER_10000": 0,
            "VOLCANIC_ROCK_VOUCHER_10250": 0,
            "VOLCANIC_ROCK_VOUCHER_10500": 0,
        }
        self._logs = {}
        # this is how to log:
        self.log("Initial params", self.__dict__)

    def update_performance(self, state: TradingState, current_positions: dict):
        """Track PnL and check drawdown limits"""
        if not hasattr(self, 'last_timestamp'):
            self.last_timestamp = state.timestamp
            return
            
        # Calculate current portfolio value
        portfolio_value = self.performance_tracker['starting_capital'] + self.performance_tracker['current_pnl']
        
        # Update peak PnL
        self.performance_tracker['peak_pnl'] = max(
            self.performance_tracker['peak_pnl'],
            self.performance_tracker['current_pnl']
        )
        
        # Calculate drawdown
        current_drawdown = (self.performance_tracker['peak_pnl'] - self.performance_tracker['current_pnl']) / portfolio_value
        
        # Check if we should suspend trading
        if current_drawdown > self.performance_tracker['max_drawdown_pct']:
            self.performance_tracker['suspended'] = True
            print(f"⚠️ TRADING SUSPENDED - Drawdown {current_drawdown*100:.2f}% exceeds threshold")
        
        # Update last timestamp
        self.last_timestamp = state.timestamp

    def log(self, key: str, value: any):
        self._logs[key] = value

    def check_component_limits(self, product: str, order_quantity: int, current_position: int) -> int:
        if product not in self.LIMIT:
            return order_quantity  # No limit defined for this product
        
        max_position = self.LIMIT[product]
        
        if order_quantity > 0:  # Buying
            adjusted_quantity = min(order_quantity, max_position - current_position)
        else:  # Selling
            adjusted_quantity = max(order_quantity, -max_position - current_position)
        
        return adjusted_quantity if adjusted_quantity != 0 else 0

    def vwap_slope(self, vwap_list):
        if len(vwap_list) < self.vwap_window:
            return 0
        slope = (vwap_list[-1]["vwap"] - vwap_list[-self.vwap_window]["vwap"]) / self.vwap_window
        return slope

    def order_box_diff(self, state: TradingState) -> dict[str, Order]:
        # self.log("timestamp", state.timestamp)
        # self.log("position", {k:state.position.get(k,0) for k in self.products})

        orders = {}
        artificial_product = {"PICNIC_BASKET1": 1, "PICNIC_BASKET2": -1,
                 "CROISSANTS": -2, "JAMS": -1, "DJEMBES": -1}
        traded_products = ['CROISSANTS','JAMS','DJEMBES','PICNIC_BASKET2']
        artificial_bid, artificial_ask, products_info, isWrong= calculate_artificial_bid_ask(state, artificial_product, min_qty=self.min_box_diff_order)
        if isWrong:
            self.log("isWrong", True)
            self.log("artificial_info", products_info)
            return orders
        self.log("artificial_bid_ask", (artificial_bid, artificial_ask))
        curr_position = state.position.get("PICNIC_BASKET1", 0)
        mid_price = (artificial_bid + artificial_ask) / 2
        if abs(mid_price)>self.max_spread:
            self.max_spread = abs(mid_price)
            self.log("max_spread", self.max_spread)
        up_position = price_to_position(-artificial_bid, self.max_box_diff, self.max_spread)
        down_position = price_to_position(-artificial_ask, self.max_box_diff, self.max_spread)
        if curr_position>up_position:
            additional_order = up_position - curr_position
        elif curr_position<down_position:
            additional_order = down_position - curr_position
        else:
            additional_order = 0
        # self.log("additional_order", additional_order)
        # self.log("up_position", up_position)
        # self.log("down_position", down_position)
        # self.log("curr_position", curr_position)


        if abs(additional_order) < self.min_box_diff_order:
            return orders
        elif additional_order>0:
            buy_price = products_info["PICNIC_BASKET1"]['basic_ask']
            available_trades = calculate_buy_quantity(state.order_depths["PICNIC_BASKET1"], buy_price)
            quantity_pb1 = min(available_trades, additional_order)
            order = Order("PICNIC_BASKET1", buy_price, quantity_pb1)
            orders["PICNIC_BASKET1"] = [order]
        elif additional_order<0:
            sell_price = products_info["PICNIC_BASKET1"]['basic_bid']
            available_trades = calculate_sell_quantity(state.order_depths["PICNIC_BASKET1"], sell_price)
            quantity_pb1 = max(available_trades, additional_order)
            order = Order("PICNIC_BASKET1", sell_price, quantity_pb1)
            orders["PICNIC_BASKET1"] = [order]

        self.log("quantity_pb1", quantity_pb1)


        for constituent in traded_products:
            quantity = artificial_product[constituent]*quantity_pb1
            if quantity>0:
                price = products_info[constituent]['basic_ask']
                orders[constituent] = [Order(constituent, price, quantity)]
            else:
                price = products_info[constituent]['basic_bid']
                orders[constituent] = [Order(constituent, price, quantity)]

        #check proper proportions
        artificial_sum =0
        base = orders["PICNIC_BASKET1"][0].quantity
        for prod in traded_products:
            expected_q = artificial_product[prod]*base
            actual_q =orders[prod][0].quantity
            artificial_sum+= np.abs(expected_q-actual_q)
        if artificial_sum!=0:
            self.log("artificial_sum", artificial_sum)
            self.log("position", state.position)
        orders = add_order_for_proportions(state, artificial_product, orders, "PICNIC_BASKET1", traded_products)
        spent = 0
        for order in orders.values():
            spent += order[0].price * order[0].quantity
        if not (artificial_bid<=(spent/base)<=artificial_ask):
            self.log("spent_per_unit", spent/(base))
            self.log("quantity", base)
            self.log("bidask", (artificial_bid, artificial_ask))
        return orders

    def squid_ink_adaptive_zscore(self, product, order_depth, position, prices, vwap_list, timespan) -> list[Order]:
        orders = []

        if self.performance_tracker['suspended']:
            print("🚫 Trading suspended - not generating new orders")
            return []
        
        # Strategy parameters
        lookback_period = 20
        base_z_buy_threshold = -2.7
        base_z_sell_threshold = 0.9
        exit_zone = 0.0
        position_limit = self.LIMIT[product]
        
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
            
            # Calculate Z-score
            if len(prices) >= lookback_period:
                recent_prices = prices[-lookback_period:]
                mean_price = sum(recent_prices) / len(recent_prices)
                std_dev = (sum((price - mean_price) ** 2 for price in recent_prices) / len(recent_prices)) ** 0.5
                
                if std_dev > 0:
                    z_score = (mid_price - mean_price) / std_dev
                    
                    # Calculate trend strength using VWAP slope
                    trend_strength = 0
                    if len(vwap_list) >= 10:
                        trend_strength = (vwap_list[-1]["vwap"] - vwap_list[-10]["vwap"]) / (vwap_list[-1]["vwap"])
                    
                    # Adjust Z-score thresholds based on trend strength
                    # In stronger trends, we want more extreme thresholds to avoid fighting the trend
                    trend_adjustment = abs(trend_strength) * 2
                    z_buy_threshold = base_z_buy_threshold - trend_adjustment
                    z_sell_threshold = base_z_sell_threshold + trend_adjustment
                    
                    # Trading logic
                    limit_buy = position_limit - position
                    limit_sell = -position_limit - position
                    
                    # Filtered order book
                    filtered_asks = [p for p in order_depth.sell_orders if abs(order_depth.sell_orders[p]) >= 15]
                    filtered_bids = [p for p in order_depth.buy_orders if abs(order_depth.buy_orders[p]) >= 15]
                    
                    mm_ask = min(filtered_asks) if filtered_asks else best_ask
                    mm_bid = max(filtered_bids) if filtered_bids else best_bid
                    
                    # Position sizing based on Z-score extremity
                    intensity = min(1.0, abs(z_score) / max(abs(z_buy_threshold), abs(z_sell_threshold)))
                    max_position_size = int(position_limit * 0.8 * intensity)  # 80% of limit max, scaled by intensity
                    
                    # Buy signal with Z-score below threshold (oversold)
                    if z_score <= z_buy_threshold and limit_buy > 0:
                        buy_price = mm_bid + (2 if z_score < z_buy_threshold * 1.5 else 1)  # More aggressive for extreme Z
                        quantity = min(max_position_size, limit_buy, calculate_buy_quantity(order_depth, buy_price))
                        quantity = self.check_component_limits(product, quantity, position)
                        if quantity > 0:
                            orders.append(Order(product, buy_price, quantity))
                    
                    # Sell signal with Z-score above threshold (overbought)
                    if z_score >= z_sell_threshold and limit_sell < 0:
                        sell_price = mm_ask - (2 if z_score > z_sell_threshold * 1.5 else 1)  # More aggressive for extreme Z
                        quantity = max(-max_position_size, limit_sell, calculate_sell_quantity(order_depth, sell_price))
                        quantity = self.check_component_limits(product, quantity, position)
                        if quantity < 0:
                            orders.append(Order(product, sell_price, quantity))
                    
                    # Take profit when Z-score approaches mean
                    if position > 0 and z_score >= -exit_zone:  # Exit long positions
                        exit_price = best_ask
                        quantity = self.check_component_limits(product, -position, position)
                        orders.append(Order(product, exit_price, -position))
                    
                    elif position < 0 and z_score <= exit_zone:  # Exit short positions
                        exit_price = best_bid
                        quantity = self.check_component_limits(product, -position, position)
                        orders.append(Order(product, exit_price, -position))
                    
                    # Stop loss for risk management
                    if (position > 0 and z_score > z_sell_threshold * 1.5) or \
                    (position < 0 and z_score < z_buy_threshold * 1.5):
                        # Emergency exit if Z-score moves strongly against position
                        stop_price = mm_bid if position > 0 else mm_ask
                        orders.append(Order(product, stop_price, -position))
        
        return orders

    def hybrid_orders(self, product, order_depth, position, prices, vwap_list, timespan, take_width, make_width):
        orders = []
        buy_order_volume = 0
        sell_order_volume = 0

        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            filtered_ask = [p for p in order_depth.sell_orders if abs(order_depth.sell_orders[p]) >= 15]
            filtered_bid = [p for p in order_depth.buy_orders if abs(order_depth.buy_orders[p]) >= 15]
            mm_ask = min(filtered_ask) if filtered_ask else best_ask
            mm_bid = max(filtered_bid) if filtered_bid else best_bid

            mmmid_price = (mm_ask + mm_bid) / 2
            prices.append(mmmid_price)

            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * -order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume if volume != 0 else mmmid_price
            vwap_list.append({"vol": volume, "vwap": vwap})

            if len(vwap_list) > timespan:
                vwap_list.pop(0)
            if len(prices) > timespan:
                prices.pop(0)

            fair_value = mmmid_price

            buy_order_volume, sell_order_volume = self.take_best_orders(product, fair_value, take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, True, 20)
            buy_order_volume, sell_order_volume = self.clear_position_order(product, fair_value, orders, order_depth, position, buy_order_volume, sell_order_volume)

            aaf = [p for p in order_depth.sell_orders if p > fair_value + 1]
            bbf = [p for p in order_depth.buy_orders if p < fair_value - 1]
            baaf = min(aaf) if aaf else fair_value + 2
            bbbf = max(bbf) if bbf else fair_value - 2

            buy_order_volume, sell_order_volume = self.market_make(product, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)

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
    
    def get_underlying_price(self, state: TradingState) -> float:
        underlying_name = "VOLCANIC_ROCK"
        underlying_prices_available = []
        if state.order_depths[underlying_name] is not None:
            underlying_bid = get_significant_bid(state.order_depths[underlying_name], 1)
            underlying_ask = get_significant_ask(state.order_depths[underlying_name], 1)
            if underlying_bid is not None:
                underlying_prices_available.append(underlying_bid)
            if underlying_ask is not None:
                underlying_prices_available.append(underlying_ask)

        lower_strike_coupon = "VOLCANIC_ROCK_VOUCHER_9500"
        strike = extract_strike(lower_strike_coupon)
        hedge_coupon_ask = get_significant_ask(state.order_depths[lower_strike_coupon], 1)
        hedge_coupon_bid = get_significant_bid(state.order_depths[lower_strike_coupon], 1)
        if hedge_coupon_ask is not None:
            underlying_prices_available.append(hedge_coupon_ask+strike)
        if hedge_coupon_bid is not None:
            underlying_prices_available.append(hedge_coupon_bid+strike)
        return np.mean(underlying_prices_available)

    def order_volcanic_rock_voucher(self, state: TradingState, voucher_name: str) -> list[Order]:
        orders = []
        # Choose a voucher product (this example uses one voucher; you can iterate over several if desired)

        # Obtain order depth and current position for the voucher
        order_depth = state.order_depths[voucher_name]
        pos = state.position.get(voucher_name, 0)
        self.log(f"{voucher_name}_position", pos)

        # Limits on position (assumed to be set in self.limits)
        limit_buy = self.LIMIT[voucher_name] - pos
        limit_sell = -self.LIMIT[voucher_name] - pos

        #underlying price
        underlying_name = "VOLCANIC_ROCK"
        underlying_price = self.get_underlying_price(state)
        if underlying_price is None:
            return []


        # Extract the strike from the voucher's voucher_name name
        strike = extract_strike(voucher_name)


        # Determine best available prices from the order book:
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            # Compute IV based on the ask price (buy side)
            iv_ask = implied_vol(best_ask, underlying_price, strike, 1, 0, option_type='call')
        else:
            best_ask = None
            iv_ask = None

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            # Compute IV based on the bid price (sell side)
            iv_bid = implied_vol(best_bid, underlying_price, strike, 1, 0, option_type='call')
        else:
            best_bid = None
            iv_bid = None

        # Update IV history for the voucher—only add if we have a valid value.
        if iv_ask is not None:
            self.ask_ivol_history[voucher_name].append(iv_ask)
            if len(self.ask_ivol_history[voucher_name]) > self.volcano_history_len:
                self.ask_ivol_history[voucher_name].pop(0)

        if iv_bid is not None:
            self.bid_ivol_history[voucher_name].append(iv_bid)
            if len(self.bid_ivol_history[voucher_name]) > self.volcano_history_len:
                self.bid_ivol_history[voucher_name].pop(0)

        # Compute dynamic target IV thresholds:
        # For buying, we set target by the 0.3 quantile of the sell-side IV history.

        if len(self.bid_ivol_history[voucher_name]) >= self.short_history_len:
            target_iv_ask = np.quantile(self.bid_ivol_history[voucher_name][-self.short_history_len:], 0.7)
        else:
            return []

        # For selling, we set target by the 0.7 quantile of the buy-side IV history.
        if len(self.ask_ivol_history[voucher_name]) >= self.short_history_len:
            target_iv_bid = np.quantile(self.ask_ivol_history[voucher_name][-self.short_history_len:], 0.3)
        else:
            return []

        if probability_of_overlap(self.ask_ivol_history[voucher_name],
                                  self.bid_ivol_history[voucher_name]) < self.prob_lim and self.block_prob:
            self.log(f"blocked_{strike}", True)
            self.log("blocked_underlying_price", underlying_price)
            return orders

        # self.log(f"{voucher_name}_iv_buy", iv_ask)
        # self.log(f"{voucher_name}_iv_sell", iv_bid)
        # self.log("target_iv_ask", target_iv_ask)
        # self.log("target_iv_bid", target_iv_bid)

        # ----- Trading decision based on dynamically computed IV thresholds -----
        # If the voucher's buy-side implied volatility (from best ask) is below target_iv_ask,
        # then the option appears undervalued in volatility terms and we consider buying.
        if target_iv_ask is not None:
            bid_price = int(math.floor(black_scholes_call_price(underlying_price, strike, 1, 0, target_iv_ask)))
            q = calculate_buy_quantity(order_depth, bid_price)
            q = min(q, limit_buy)
            if q != 0:
                orders.append(Order(voucher_name, best_ask, q))
                pos += q
                limit_buy -= q

        # Conversely, if the voucher's sell-side implied volatility (from best bid) is above target_iv_bid,
        # the option seems expensive in volatility terms, so we consider selling.
        if target_iv_bid is not None:
            ask_price = int(math.ceil(black_scholes_call_price(underlying_price, strike, 1, 0, target_iv_bid)))
            q = calculate_sell_quantity(order_depth, ask_price)
            # Note: Here, using max(q, limit_sell) is kept from your original logic.
            q = max(q, limit_sell)
            if q != 0:
                orders.append(Order(voucher_name, best_bid, q))
                pos += q
                limit_sell -= q
        return orders
    
    def update(self, state: TradingState):
        if state and state.__dict__.get("traderData") is not None and state.traderData != "":
            try:
                # self.__dict__.update(pickle.loads(base64.b64decode(state.traderData.encode("utf-8"))))
                state.traderData = ""
                self._logs = {}
            except Exception as e:
                self.log("template_error", str(e))

        self.history[state.timestamp] = state
        if len(self.history) > self.history_size:
            self.history.pop(min(self.history.keys()))

        for product in self.products:
            if product in state.position:
                self.position[product] = state.position[product]
            else:
                self.position[product] = 0

            calc_traded_price(self.history, product, n_back=5)

    def run(self, state: TradingState):
        result = {}
        order_depths = state.order_depths
        positions = state.position
        self._logs = {}
        self.update(state)
        self.log("timestamp", state.timestamp)

        current_positions = {product: positions.get(product, 0) for product in self.LIMIT.keys()}
        resin_pos = state.position.get(Product.RAINFOREST_RESIN, 0)
        kelp_pos = state.position.get(Product.KELP, 0)

        resin_orders = self.hybrid_orders(Product.RAINFOREST_RESIN, state.order_depths[Product.RAINFOREST_RESIN], resin_pos, self.resin_prices, self.resin_vwap, 10, 1, 3.5)
        kelp_orders = self.hybrid_orders(Product.KELP, state.order_depths[Product.KELP], kelp_pos, self.kelp_prices, self.kelp_vwap, 10, 1, 3.5)

        self.update_performance(state, current_positions)
        
        # Run strategies only if not suspended
        if not self.performance_tracker['suspended']:
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


        result[Product.RAINFOREST_RESIN] = resin_orders
        result[Product.KELP] = kelp_orders

        box_diff_orders = self.order_box_diff(state)
        for product, orders in box_diff_orders.items():
            if product in result:
                # Append new orders to existing ones
                result[product].extend(orders)
            else:
                # Create new entry
                result[product] = orders
        if not hasattr(self, 'volcanic_voucher_prices'):
            self.volcanic_voucher_prices = {
                "VOLCANIC_ROCK_VOUCHER_9500": [],
                "VOLCANIC_ROCK_VOUCHER_9750": [],
                "VOLCANIC_ROCK_VOUCHER_10000": [],
                "VOLCANIC_ROCK_VOUCHER_10250": [],
                "VOLCANIC_ROCK_VOUCHER_10500": [],
            }
            self.volcanic_voucher_vwap = {
                "VOLCANIC_ROCK_VOUCHER_9500": [],
                "VOLCANIC_ROCK_VOUCHER_9750": [],
                "VOLCANIC_ROCK_VOUCHER_10000": [],
                "VOLCANIC_ROCK_VOUCHER_10250": [],
                "VOLCANIC_ROCK_VOUCHER_10500": [],
            }

        # Update price data for volcanic vouchers
        for voucher in self.trading_coupons:
            if voucher in order_depths:
                order_depth = order_depths[voucher]
                if order_depth.sell_orders and order_depth.buy_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_bid = max(order_depth.buy_orders.keys())
                    mid_price = (best_ask + best_bid) / 2
                    if voucher not in self.volcanic_voucher_prices:
                        self.volcanic_voucher_prices[voucher] = []
                    self.volcanic_voucher_prices[voucher].append(mid_price)

                    # Update VWAP
                    ask_vol = -order_depth.sell_orders[best_ask]
                    bid_vol = order_depth.buy_orders[best_bid]
                    volume = ask_vol + bid_vol
                    vwap = (best_bid * ask_vol + best_ask * bid_vol) / volume if volume != 0 else mid_price
                    self.volcanic_voucher_vwap[voucher].append({"vol": volume, "vwap": vwap})

                    # Maintain window size
                    if len(self.volcanic_voucher_prices[voucher]) > 20:
                        self.volcanic_voucher_prices[voucher].pop(0)
                    if len(self.volcanic_voucher_vwap[voucher]) > 20:
                        self.volcanic_voucher_vwap[voucher].pop(0)

            self.required_hedge_position = 0
        volcanic_callers = {
            "VOLCANIC_ROCK_VOUCHER_9750": lambda state: self.order_volcanic_rock_voucher(state, "VOLCANIC_ROCK_VOUCHER_9750"),
            "VOLCANIC_ROCK_VOUCHER_10000": lambda state: self.order_volcanic_rock_voucher(state, "VOLCANIC_ROCK_VOUCHER_10000"),
            "VOLCANIC_ROCK_VOUCHER_10250": lambda state: self.order_volcanic_rock_voucher(state, "VOLCANIC_ROCK_VOUCHER_10250"),
            "VOLCANIC_ROCK_VOUCHER_10500": lambda state: self.order_volcanic_rock_voucher(state, "VOLCANIC_ROCK_VOUCHER_10500"),
    }
        volcanic_orders = {x: caller(state) for x, caller in volcanic_callers.items()}
        for product_name, orders in volcanic_orders.items():
            result[product_name] = orders

        traderData = jsonpickle.encode({
            "resin_prices": self.resin_prices if hasattr(self, 'resin_prices') else [],
            "resin_vwap": self.resin_vwap if hasattr(self, 'resin_vwap') else [],
            "kelp_prices": self.kelp_prices if hasattr(self, 'kelp_prices') else [],
            "kelp_vwap": self.kelp_vwap if hasattr(self, 'kelp_vwap') else [],
            "squid_ink_prices": self.squid_ink_prices if hasattr(self, 'squid_ink_prices') else [],
            "squid_ink_vwap": self.squid_ink_vwap if hasattr(self, 'squid_ink_vwap') else [],
            "volcanic_voucher_prices": self.volcanic_voucher_prices if hasattr(self, 'volcanic_voucher_prices') else {},
            "volcanic_voucher_vwap": self.volcanic_voucher_vwap if hasattr(self, 'volcanic_voucher_vwap') else {},
            "ask_ivol_history": self.ask_ivol_history,
            "bid_ivol_history": self.bid_ivol_history,
        })

        conversions = 1
        return result, conversions, traderData