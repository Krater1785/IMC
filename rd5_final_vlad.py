from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Tuple
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


def calculate_artificial_bid_ask(state, artificial_product, min_qty=10):
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

        basic_bid = get_highest_bid_with_min_qty(order_depth, product, min_qty * abs(coef))
        basic_ask = get_lowest_ask_with_min_qty(order_depth, product, min_qty * abs(coef))
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


def fill_the_order(state: TradingState, product: str, quantity: int) -> Optional[Order]:
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


def add_order_for_proportions(state: TradingState, artificial_product: Dict[str, int], orders: Dict[str, list[Order]],
                              main_product, additional_products):
    base = state.position.get(main_product, 0)
    for prod in additional_products:
        expected_q = artificial_product[prod] * base
        actual_q = state.position.get(prod, 0)
        filled_order = fill_the_order(state, prod, expected_q - actual_q)
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


def probability_of_overlap(ask_iv, bid_iv) -> float:
    m_ask = np.quantile(ask_iv, 0.2)
    m_bid = np.quantile(bid_iv, 0.8)
    return np.mean(bid_iv > m_ask) + np.mean(ask_iv < m_bid)


def add_order_for_proportions(state: TradingState, artificial_product: Dict[str, int], orders: Dict[str, list[Order]],
                              main_product, additional_products):
    base = state.position.get(main_product, 0)
    for prod in additional_products:
        expected_q = artificial_product[prod] * base
        actual_q = state.position.get(prod, 0)
        filled_order = fill_the_order(state, prod, expected_q - actual_q)
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
    def __init__(self, volcano_history_len=50, block_prob=True, prob_lim=0.05, verbose=True, min_box_diff_order=15):
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
            "MAGNIFICENT_MACARONS": 40,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
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
        olivia_products = ['SQUID_INK','CROISSANTS','PICNIC_BASKET1','PICNIC_BASKET2','JAMS','DJEMBES']
        self.olivia_based_target = {p: 0 for p in olivia_products}

        self.resin_prices = []
        self.resin_vwap = []
        self.kelp_prices = []
        self.kelp_vwap = []

        self.current_position = 0
        self.squid_ink_prices = []
        self.squid_ink_vwap = []
        self.last_price = None
        self._price_cache = None  # Tuple of (mean, std)
        self._vwap_cache = {}

        self.performance_tracker = {
            'realized_pnl': 0.0,  # Ensure float type
            'unrealized_pnl': 0.0,
            'position_cost': 0.0,
            'peak_pnl': 0.0,
            'trade_history': [],
            'suspended': False
        }

        self.max_box_diff = 60
        self.max_spread = 200
        self.min_box_diff_order = min_box_diff_order

        self.performance_baskets = {
            'realized_pnl_bs': 0.0,
            'unrealized_pnl_bs': 0.0,
            'position_cost': 0.0,
            'peak_pnl': 0.0,
            'suspended': False,
            'last_timestamp': None
        }

        self.prob_lim = prob_lim
        self.block_prob = block_prob
        self.volcano_history_len = volcano_history_len
        self.short_history_len = volcano_history_len // 5
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

        self._logs = {}
        # this is how to log:
        self.log("Initial params", self.__dict__)

    def log(self, key: str, value: any):
        self._logs[key] = value

    ############################################################################################################
    # Blocks for RESIN and KELP
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
            vwap = (best_bid * -order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[
                best_bid]) / volume if volume != 0 else mmmid_price
            vwap_list.append({"vol": volume, "vwap": vwap})

            if len(vwap_list) > timespan:
                vwap_list.pop(0)
            if len(prices) > timespan:
                prices.pop(0)

            fair_value = mmmid_price

            buy_order_volume, sell_order_volume = self.take_best_orders(product, fair_value, take_width, orders,
                                                                        order_depth, position, buy_order_volume,
                                                                        sell_order_volume, True, 20)
            buy_order_volume, sell_order_volume = self.clear_position_order(product, fair_value, orders, order_depth,
                                                                            position, buy_order_volume,
                                                                            sell_order_volume)

            aaf = [p for p in order_depth.sell_orders if p > fair_value + 1]
            bbf = [p for p in order_depth.buy_orders if p < fair_value - 1]
            baaf = min(aaf) if aaf else fair_value + 2
            bbbf = max(bbf) if bbf else fair_value - 2

            buy_order_volume, sell_order_volume = self.market_make(product, orders, bbbf + 1, baaf - 1, position,
                                                                   buy_order_volume, sell_order_volume)

        return orders

    def take_best_orders(self, product: str,
                         fair_value: int,
                         take_width: float,
                         orders: List[Order],
                         order_depth: OrderDepth,
                         position: int,
                         buy_order_volume: int,
                         sell_order_volume: int,
                         prevent_adverse: bool = False,
                         adverse_volume: int = 0, ):
        position_limit = self.LIMIT[product]

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if (not prevent_adverse and best_ask <= fair_value - take_width) or (
                    prevent_adverse and best_ask_amount <= adverse_volume and best_ask <= fair_value - take_width):
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, round(best_ask), quantity))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if (not prevent_adverse and best_bid >= fair_value + take_width) or (
                    prevent_adverse and best_bid_amount <= adverse_volume and best_bid >= fair_value + take_width):
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
                             sell_order_volume: int, ):

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
                    sell_order_volume: int, ):

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, int(bid), buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, int(ask), -sell_quantity))

        return buy_order_volume, sell_order_volume

    ###########################################################################
    # Blocks for SQUID INK
    def order_squid_ink(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        product = 'SQUID_INK'
        trades = state.market_trades.get(product, [])
        for trade in trades:
            if trade.buyer == 'Olivia':
                self.olivia_based_target[product] = self.LIMIT[product]
                break
            elif trade.seller == 'Olivia':
                self.olivia_based_target[product] = -self.LIMIT[product]
                break

        current = state.position.get(product, 0)
        delta = self.olivia_based_target[product] - current
        if delta == 0:
            return orders

        od = state.order_depths.get(product)
        if od is None:
            return orders
        if delta > 0 and od.sell_orders:
            ask_price = max(od.sell_orders.keys())
            orders.append(Order(product, ask_price, delta))
        elif delta < 0 and od.buy_orders:
            bid_price = min(od.buy_orders.keys())
            orders.append(Order(product, bid_price, delta))
        return orders

    #############################################################################################################################
    # Blocks for BASKETS and COMPONENTS

    def order_croissants(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        product = 'CROISSANTS'
        trades = state.market_trades.get(product, [])
        for trade in trades:
            if trade.buyer == 'Olivia':
                self.olivia_based_target[product] = self.LIMIT[product]
                break
            elif trade.seller == 'Olivia':
                self.olivia_based_target[product] = -self.LIMIT[product]
                break

        current = state.position.get(product, 0)
        delta = self.olivia_based_target[product] - current
        if delta == 0:
            return orders

        od = state.order_depths.get(product)
        if od is None:
            return orders
        if delta > 0 and od.sell_orders:
            ask_price = max(od.sell_orders.keys())
            orders.append(Order(product, ask_price, delta))
        elif delta < 0 and od.buy_orders:
            bid_price = min(od.buy_orders.keys())
            orders.append(Order(product, bid_price, delta))
        return orders

    def order_picnic_basket_1(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        product = 'PICNIC_BASKET1'
        cb_target = self.olivia_based_target['CROISSANTS']
        if cb_target > 0:
            self.olivia_based_target[product] = self.LIMIT[product]
        elif cb_target < 0:
            self.olivia_based_target[product] = -self.LIMIT[product]
        else:
            self.olivia_based_target[product] = 0

        current = state.position.get(product, 0)
        delta = self.olivia_based_target[product] - current
        if delta == 0:
            return orders

        od = state.order_depths.get(product)
        if od is None:
            return orders
        if delta > 0 and od.sell_orders:
            ask_price = max(od.sell_orders.keys())
            orders.append(Order(product, ask_price, delta))
        elif delta < 0 and od.buy_orders:
            bid_price = min(od.buy_orders.keys())
            orders.append(Order(product, bid_price, delta))
        return orders

    def order_picnic_basket_2(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        product = 'PICNIC_BASKET2'
        cb_target = self.olivia_based_target['CROISSANTS']
        if cb_target > 0:
            self.olivia_based_target[product] = self.LIMIT[product]
        elif cb_target < 0:
            self.olivia_based_target[product] = -self.LIMIT[product]
        else:
            self.olivia_based_target[product] = 0

        current = state.position.get(product, 0)
        delta = self.olivia_based_target[product] - current
        if delta == 0:
            return orders

        od = state.order_depths.get(product)
        if od is None:
            return orders
        if delta > 0 and od.sell_orders:
            ask_price = max(od.sell_orders.keys())
            orders.append(Order(product, ask_price, delta))
        elif delta < 0 and od.buy_orders:
            bid_price = min(od.buy_orders.keys())
            orders.append(Order(product, bid_price, delta))
        if delta!=0:
            self.log("pbs2_orders", orders)
            self.log("pb2_order_book", od)
        return orders

    def order_jams(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        product = 'JAMS'
        cb_target = self.olivia_based_target['CROISSANTS']
        # opposite of PICNIC_BASKET signals
        if cb_target > 0:
            self.olivia_based_target[product] = -self.LIMIT[product]
        elif cb_target < 0:
            self.olivia_based_target[product] = self.LIMIT[product]
        else:
            self.olivia_based_target[product] = 0

        current = state.position.get(product, 0)
        delta = self.olivia_based_target[product] - current
        if delta == 0:
            return orders

        od = state.order_depths.get(product)
        if od is None:
            return orders
        if delta > 0 and od.sell_orders:
            ask_price = max(od.sell_orders.keys())
            orders.append(Order(product, ask_price, delta))
        elif delta < 0 and od.buy_orders:
            bid_price = min(od.buy_orders.keys())
            orders.append(Order(product, bid_price, delta))
        return orders

    def order_djembes(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        product = 'DJEMBES'
        cb_target = self.olivia_based_target['CROISSANTS']
        if cb_target > 0:
            self.olivia_based_target[product] = -self.LIMIT[product]
        elif cb_target < 0:
            self.olivia_based_target[product] = self.LIMIT[product]
        else:
            self.olivia_based_target[product] = 0

        current = state.position.get(product, 0)
        delta = self.olivia_based_target[product] - current
        if delta == 0:
            return orders

        od = state.order_depths.get(product)
        if od is None:
            return orders
        if delta > 0 and od.sell_orders:
            ask_price = max(od.sell_orders.keys())
            orders.append(Order(product, ask_price, delta))
        elif delta < 0 and od.buy_orders:
            bid_price = min(od.buy_orders.keys())
            orders.append(Order(product, bid_price, delta))
        return orders

    ############################################################################################################
    # Blocks for VOUCHERS

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
            underlying_prices_available.append(hedge_coupon_ask + strike)
        if hedge_coupon_bid is not None:
            underlying_prices_available.append(hedge_coupon_bid + strike)
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

        # underlying price
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

        if target_iv_bid > target_iv_ask:
            target_iv_bid = (target_iv_bid + target_iv_ask) / 2
            target_iv_ask = target_iv_bid

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

    def clean_logs(self):
        """Reset the internal logs."""
        self._logs = {}

    ############################################################################################################
    # Blocks for MACARONS
    def trade_magnificent_macarons(self, state: TradingState):
        orders = []
        """
        Logs important information from the state:
        1. For the product 'Magnificent Macarons', it logs the best bid (max bid)
           and best ask (min ask) prices.
        2. It logs the conversion observation details if available.
        """
        # Log best bid and ask for "Magnificent Macarons"
        product = "MAGNIFICENT_MACARONS"
        position = state.position.get(product, 0)
        lim1 = 20
        lim2 = 40
        self.log("position", position)
        if product in state.order_depths:
            observations = state.observations.conversionObservations.get(product, None)
            if observations:
                island_price = observations.askPrice + observations.transportFees + observations.importTariff
            else:
                self.log("no_observations", True)
                return orders
            self.log("island_price", island_price)

            order_depth = state.order_depths[product]
            # Calculate best bid as the highest available bid
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            total_qty = -state.position.get(product, 0)
            if best_bid is not None:
                self.log("min_profit", best_bid - island_price)

            if best_bid is not None and best_bid > island_price:

                for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    potential_profit = price - island_price
                    if potential_profit < 0.5:
                        break
                    if total_qty < lim1:
                        if potential_profit > 1.0:
                            available = order_depth.buy_orders[price]
                            take = min(50 - total_qty, available)
                            if take > 0:
                                orders.append(Order(product, round(price), -take))
                                total_qty += take
                        else:
                            available = order_depth.buy_orders[price]
                            take = min(lim1 - total_qty, available)
                            if take > 0:
                                orders.append(Order(product, round(price), -take))
                                total_qty += take
                    else:
                        if price - island_price >= 2.0:
                            available = order_depth.buy_orders[price]
                            take = min(lim2 - total_qty, available)
                            if take > 0:
                                orders.append(Order(product, round(price), -take))
                                total_qty += take
                        else:
                            break
                self.log("total_qty", total_qty)
                # self.log("lowest_price", min([x.price for x in orders]))
                if total_qty > 10:
                    buy_qty = total_qty - 10
                    orders.append(Order(product, int(island_price), buy_qty))
                    self.log("buy_island_qty", buy_qty)
            else:
                self.log("no_arbitrage", True)
            self.log(f"best_bid", best_bid)
            self.log(f"best_ask", best_ask)
            self.log(f"book_orders", orders)

            ##mm orders
            min_price = int(round(island_price) + 1)
            available = 20 - total_qty
            mm_orders = []
            if available > 10:
                take_high = available - 10
                take_low = 10
                mm_orders.append(Order(product, round(min_price), -take_low))
                mm_orders.append(Order(product, round(min_price + 1), -take_high))
            elif available > 0:
                take_low = available
                mm_orders.append(Order(product, round(min_price), -take_low))
            if mm_orders:
                self.log("mm_orders", mm_orders)
        return orders

    def determine_conversion_size(self, potential_position):
        island_lim = 10
        product = "MAGNIFICENT_MACARONS"
        position = potential_position
        result = 0
        if position > 0:
            result = 0
        elif position < -island_lim:
            result = island_lim
        else:
            result = -position
        return result

    def run(self, state: TradingState):
        result = {}
        self._logs = {}
        self.log("timestamp", state.timestamp)

        # Load previous state
        if state.traderData:
            old_data = jsonpickle.decode(state.traderData)
            self.resin_prices = old_data.get("resin_prices", [])
            self.resin_vwap = old_data.get("resin_vwap", [])
            self.kelp_prices = old_data.get("kelp_prices", [])
            self.kelp_vwap = old_data.get("kelp_vwap", [])
            self.volcanic_voucher_prices = old_data.get("volcanic_voucher_prices", {})
            self.volcanic_voucher_vwap = old_data.get("volcanic_voucher_vwap", {})
            self.ask_ivol_history = old_data.get("ask_ivol_history", self.ask_ivol_history)
            self.bid_ivol_history = old_data.get("bid_ivol_history", self.bid_ivol_history)

        if not hasattr(self, 'last_timestamp'):
            self.last_timestamp = state.timestamp

        # RESIN and KELP
        resin_pos = state.position.get(Product.RAINFOREST_RESIN, 0)
        kelp_pos = state.position.get(Product.KELP, 0)

        resin_orders = self.hybrid_orders(Product.RAINFOREST_RESIN, state.order_depths[Product.RAINFOREST_RESIN],
                                          resin_pos, self.resin_prices, self.resin_vwap, 10, 1, 3.5)
        kelp_orders = self.hybrid_orders(Product.KELP, state.order_depths[Product.KELP], kelp_pos, self.kelp_prices,
                                         self.kelp_vwap, 10, 1, 3.5)

        result[Product.RAINFOREST_RESIN] = resin_orders
        result[Product.KELP] = kelp_orders

        # SQUID INK
        result[Product.SQUID_INK] = self.order_squid_ink(state)

        self.last_timestamp = state.timestamp

        # PICNIC BASKETS and COMPONENTS
        result[Product.CROISSANTS] = self.order_croissants(state)
        result[Product.PICNIC_BASKET1] = self.order_picnic_basket_1(state)
        result[Product.PICNIC_BASKET2] = self.order_picnic_basket_2(state)
        result[Product.JAMS] = self.order_jams(state)
        result[Product.DJEMBES] = self.order_djembes(state)

        # VOUCHERS
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

        for voucher in self.trading_coupons:
            if voucher in state.order_depths:
                order_depth = state.order_depths[voucher]
                if order_depth.sell_orders and order_depth.buy_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_bid = max(order_depth.buy_orders.keys())
                    mid_price = (best_ask + best_bid) / 2
                    if voucher not in self.volcanic_voucher_prices:
                        self.volcanic_voucher_prices[voucher] = []
                    self.volcanic_voucher_prices[voucher].append(mid_price)

                    ask_vol = -order_depth.sell_orders[best_ask]
                    bid_vol = order_depth.buy_orders[best_bid]
                    volume = ask_vol + bid_vol
                    vwap = (best_bid * ask_vol + best_ask * bid_vol) / volume if volume != 0 else mid_price
                    self.volcanic_voucher_vwap[voucher].append({"vol": volume, "vwap": vwap})

                    if len(self.volcanic_voucher_prices[voucher]) > 20:
                        self.volcanic_voucher_prices[voucher].pop(0)
                    if len(self.volcanic_voucher_vwap[voucher]) > 20:
                        self.volcanic_voucher_vwap[voucher].pop(0)

        volcanic_callers = {
            "VOLCANIC_ROCK_VOUCHER_9750": lambda state: self.order_volcanic_rock_voucher(state,
                                                                                         "VOLCANIC_ROCK_VOUCHER_9750"),
            "VOLCANIC_ROCK_VOUCHER_10000": lambda state: self.order_volcanic_rock_voucher(state,
                                                                                          "VOLCANIC_ROCK_VOUCHER_10000"),
            "VOLCANIC_ROCK_VOUCHER_10250": lambda state: self.order_volcanic_rock_voucher(state,
                                                                                          "VOLCANIC_ROCK_VOUCHER_10250"),
            "VOLCANIC_ROCK_VOUCHER_10500": lambda state: self.order_volcanic_rock_voucher(state,
                                                                                          "VOLCANIC_ROCK_VOUCHER_10500"),
        }

        volcanic_orders = {x: caller(state) for x, caller in volcanic_callers.items()}
        for product_name, orders in volcanic_orders.items():
            result[product_name] = orders

        # MACARONS
        try:
            macaron_orders = self.trade_magnificent_macarons(state)
            if macaron_orders:
                if "MAGNIFICENT_MACARONS" in result:
                    result["MAGNIFICENT_MACARONS"].extend(macaron_orders)
                else:
                    result["MAGNIFICENT_MACARONS"] = macaron_orders
        except Exception as e:
            self.log("macarons_error", str(e))
            result["MAGNIFICENT_MACARONS"] = []

        traderData = jsonpickle.encode({
            "resin_prices": self.resin_prices,
            "resin_vwap": self.resin_vwap,
            "kelp_prices": self.kelp_prices,
            "kelp_vwap": self.kelp_vwap,
            "volcanic_voucher_prices": self.volcanic_voucher_prices,
            "volcanic_voucher_vwap": self.volcanic_voucher_vwap,
            "ask_ivol_history": self.ask_ivol_history,
            "bid_ivol_history": self.bid_ivol_history,
        })

        if self.verbose:
            print(self._logs)
            self.clean_logs()

        conversions = self.determine_conversion_size(state.position.get("MAGNIFICENT_MACARONS", 0))

        return result, conversions, traderData