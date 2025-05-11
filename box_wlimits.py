import base64
import pickle
from typing import Optional, Dict, List
import numpy as np
from datamodel import TradingState, Order
import jsonpickle

def calculate_buy_quantity(order_depth, target_price):
    asks = order_depth.sell_orders
    q = sum([-y for x, y in asks.items() if x <= target_price])
    return q

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

def calculate_sell_quantity(order_depth, target_price):
    bids = order_depth.buy_orders
    q = sum([-y for x, y in bids.items() if x >= target_price])
    return q

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
    if (sum(abs_quantities) == 0):
        return -1
    weighted_price = np.average(prices, weights=abs_quantities)
    return weighted_price

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
        if filled_order:
            orders[prod] = orders.get(prod, []) + [filled_order]
    return orders

def get_lowest_ask_with_min_qty(order_depth, product, min_qty):
    if not hasattr(order_depth, 'sell_orders'):
        return None
    sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
    for price, volume in sorted_asks:
        available = -volume if volume < 0 else volume
        if available >= min_qty:
            return price
    return None

def get_highest_bid_with_min_qty(order_depth, product, min_qty):
    if not hasattr(order_depth, 'buy_orders'):
        return None
    sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
    for price, volume in sorted_bids:
        if volume >= min_qty:
            return price
    return None

def calculate_artificial_bid_ask(state, artificial_product, min_qty = 10):
    isWrong = False
    artificial_bid = 0.0
    artificial_ask = 0.0
    product_prices = {}

    for product, coef in artificial_product.items():
        order_depth = state.order_depths.get(product)
        if not order_depth:
            isWrong = True
            continue

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
            artificial_bid += coef * basic_ask
            artificial_ask += coef * basic_bid

    return artificial_bid, artificial_ask, product_prices, isWrong

class Trader:
    def __init__(self, verbose=True, min_box_diff_order=15):
        self.verbose = verbose
        self.min_box_diff_order = min_box_diff_order
        self.products = ['CROISSANTS','JAMS','DJEMBES','PICNIC_BASKET1','PICNIC_BASKET2']
        self.history_size = 1
        self.history = {}
        self.max_box_diff = 60
        self.max_spread = 200

        self.limits = {
            'RAINFOREST_RESIN': 50,
            'KELP': 50,
            'SQUID_INK': 50,
            'CROISSANTS': 250,
            'JAMS': 350,
            'DJEMBES': 60,
            'PICNIC_BASKET1': 60,
            'PICNIC_BASKET2': 100,
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
        }
        
        self.spread_data = []
        self.params_spread = {
            "spread_std_window": 135,
            "zscore_threshold": 5.2,
            "default_spread_mean": 0,
            "target_position": 60,
            "spread_mean2":0,
            "spread_mean_count":0,
        }
        
        # Performance tracker similar to just_squid_ink.py
        self.performance_baskets = {
            'realized_pnl_bs': 0.0,
            'unrealized_pnl_bs': 0.0,
            'position_cost': 0.0,
            'peak_pnl': 0.0,
            'trade_history': [],
            'suspended': False,
            'last_timestamp': None
        }
        
        self._logs = {}

    def log(self, key: str, value: any):
        self._logs[key] = value

    def update_performance(self, state: TradingState, product: str):
        if product not in state.order_depths:
            return
            
        # Get current position and orders
        current_pos = state.position.get(product, 0)
        orders = state.own_trades.get(product, [])
        new_trades = [t for t in orders if t.timestamp == state.timestamp]
        
        # Calculate mid price for PnL calculation
        order_depth = state.order_depths[product]
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        current_price = (best_ask + best_bid) / 2 if best_ask and best_bid else None
        
        # Update realized PnL for new trades
        if new_trades:
            quantities = [float(t.quantity) for t in new_trades]
            prices = [float(t.price) for t in new_trades]
            trade_pnls = [(p - self.performance_baskets['position_cost']) * q 
                          for p, q in zip(prices, quantities)]
            self.performance_baskets['realized_pnl_bs'] += sum(trade_pnls)
            
            total_cost = sum(p * q for p, q in zip(prices, quantities))
            total_qty = sum(quantities)
            current_qty = float(self.position[product])
            
            if current_qty + total_qty != 0:
                self.performance_baskets['position_cost'] = float(
                    (self.performance_baskets['position_cost'] * current_qty + total_cost) / 
                    (current_qty + total_qty))
        
        # Update unrealized PnL
        if current_pos != 0 and current_price is not None:
            try:
                self.performance_baskets['unrealized_pnl_bs'] = float(
                    (current_price - self.performance_baskets['position_cost']) * current_pos
                )
            except (TypeError, ValueError) as e:
                print(f"Error calculating unrealized PnL: {e}")
                self.performance_baskets['unrealized_pnl_bs'] = 0.0
        else:
            self.performance_baskets['unrealized_pnl_bs'] = 0.0
        
        # Update peak PnL
        total_pnl = float(self.performance_baskets['realized_pnl_bs'] + self.performance_baskets['unrealized_pnl_bs'])
        self.performance_baskets['peak_pnl'] = max(float(self.performance_baskets['peak_pnl']), total_pnl)
        
        # Suspension logic
        if self.performance_baskets['peak_pnl'] - total_pnl > 1000:
            self.performance_baskets['suspended'] = True
        
        # Update trade history
        if new_trades:
            self.performance_baskets['trade_history'].extend({
                'timestamp': state.timestamp,
                'product': product,
                'price': float(trade.price),
                'quantity': trade.quantity,
                'type': 'BUY' if trade.quantity > 0 else 'SELL'
            } for trade in new_trades)

    def order_rainforest_resin(self, state: TradingState) -> list[Order]:
        return []

    def order_kelp(self, state: TradingState) -> list[Order]:
        return []

    def order_box_diff(self, state: TradingState) -> dict[str, Order]:
        orders = {}
        if self.performance_baskets['suspended']:
            return orders
            
        artificial_product = {"PICNIC_BASKET1": 1, "PICNIC_BASKET2": -1,
                 "CROISSANTS": -2, "JAMS": -1, "DJEMBES": -1}
        traded_products = ['CROISSANTS','JAMS','DJEMBES','PICNIC_BASKET2']
        artificial_bid, artificial_ask, products_info, isWrong = calculate_artificial_bid_ask(
            state, artificial_product, min_qty=self.min_box_diff_order)
            
        if isWrong:
            self.log("isWrong", True)
            self.log("artificial_info", products_info)
            return orders
            
        self.log("artificial_bid_ask", (artificial_bid, artificial_ask))
        curr_position = state.position.get("PICNIC_BASKET1", 0)
        mid_price = (artificial_bid + artificial_ask) / 2
        
        if abs(mid_price) > self.max_spread:
            self.max_spread = abs(mid_price)
            self.log("max_spread", self.max_spread)
            
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
        elif additional_order > 0:
            buy_price = products_info["PICNIC_BASKET1"]['basic_ask']
            available_trades = calculate_buy_quantity(state.order_depths["PICNIC_BASKET1"], buy_price)
            quantity_pb1 = min(available_trades, additional_order)
            order = Order("PICNIC_BASKET1", buy_price, quantity_pb1)
            orders["PICNIC_BASKET1"] = [order]
        elif additional_order < 0:
            sell_price = products_info["PICNIC_BASKET1"]['basic_bid']
            available_trades = calculate_sell_quantity(state.order_depths["PICNIC_BASKET1"], sell_price)
            quantity_pb1 = max(available_trades, additional_order)
            order = Order("PICNIC_BASKET1", sell_price, quantity_pb1)
            orders["PICNIC_BASKET1"] = [order]

        self.log("quantity_pb1", quantity_pb1)

        for constituent in traded_products:
            quantity = artificial_product[constituent]*quantity_pb1
            if quantity > 0:
                price = products_info[constituent]['basic_ask']
                orders[constituent] = [Order(constituent, price, quantity)]
            else:
                price = products_info[constituent]['basic_bid']
                orders[constituent] = [Order(constituent, price, quantity)]

        # Update performance for all traded products
        for product in orders.keys():
            self.update_performance(state, product)
            
        return orders

    def update(self, state: TradingState):
        if state and state.__dict__.get("traderData") is not None and state.traderData != "":
            try:
                self.__dict__.update(pickle.loads(base64.b64decode(state.traderData.encode("utf-8"))))
                state.traderData = ""
                self.logs = {}
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
            
        # Update last timestamp in performance tracker
        self.performance_baskets['last_timestamp'] = state.timestamp

    def run(self, state: TradingState):
        self.logs = {}
        self.update(state)
        result = self.order_box_diff(state)

        # Prepare trader data with performance information
        trader_data = jsonpickle.encode({
            'performance': {
                'realized': float(self.performance_baskets['realized_pnl_bs']),
                'unrealized': float(self.performance_baskets['unrealized_pnl_bs']),
                'position_cost': float(self.performance_baskets['position_cost']),
                'peak': float(self.performance_baskets['peak_pnl']),
                'suspended': bool(self.performance_baskets['suspended']),
                'trade_history': self.performance_baskets['trade_history'][-100:]  # Last 100 trades
            },
            'state': {
                'timestamp': state.timestamp,
                'positions': {k: state.position.get(k, 0) for k in self.products}
            }
        }, unpicklable=False)

        try:
            trader_data_binary = base64.b64encode(pickle.dumps(self.__dict__)).decode("utf-8")
        except Exception as e:
            self.log("template_error", str(e))
            trader_data_binary = ""

        if self._logs and self.verbose:
            print(self._logs)
        self._logs = {}
        
        return result, 0, trader_data_binary