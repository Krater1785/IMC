from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict , Any
import math
import jsonpickle
import numpy as np
import json


def calculate_sell_quantity(order_depth, target_price):
    bids = order_depth.buy_orders
    q = sum([-y for x, y in bids.items() if x >= target_price])
    return q


def calculate_buy_quantity(order_depth, target_price):
    asks = order_depth.sell_orders
    q = sum([-y for x, y in asks.items() if x <= target_price])
    return q

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3
        state.traderData = ""

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
    
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()  # Initialize logger (defined elsewhere)

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
    SPREAD2 = "SPREAD2"

BASKET_COMPOSITION = {
    Product.PICNIC_BASKET1: {
        Product.CROISSANTS: 6,
        Product.JAMS: 3,
        Product.DJEMBES: 1
    },
    Product.PICNIC_BASKET2: {
        Product.CROISSANTS: 4,
        Product.JAMS: 2
    }
}

class Trader:
    def __init__(self, ink_limit=22, ink_liquidate_quantity=7):
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.CROISSANTS: 250,
            Product.DJEMBES: 60,
            Product.JAMS: 350,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100
        }

        self.spread_data = {
            Product.PICNIC_BASKET1: {"spread_history": []},
            Product.PICNIC_BASKET2: {"spread_history": []},
        }

        self.params = {
            Product.SPREAD: {
                "spread_std_window": 70,
                "zscore_threshold": 2,
                "default_spread_mean": 0,
                "target_position": 60,
            },
            Product.SPREAD2: {
                "spread_std_window": 130,
                "zscore_threshold": 2,
                "default_spread_mean": 0,
                "target_position": 100,
            }
        }

    def get_max_basket_quantity(self, basket_product, desired_quantity, current_positions):
        """Returns the maximum allowed quantity considering ALL limits"""
        basket_weights = BASKET_COMPOSITION[basket_product]
        max_quantities = []
        
        for product, weight in basket_weights.items():
            current_pos = current_positions.get(product, 0)
            limit = self.LIMIT[product]
            
            if desired_quantity > 0:  # Buying basket = buying components
                max_q = (limit - current_pos) // weight
            else:  # Selling basket = selling components
                max_q = (limit + current_pos) // weight
            
            max_quantities.append(max_q)
        
        # Check basket's own limit
        current_basket_pos = current_positions.get(basket_product, 0)
        if desired_quantity > 0:
            max_quantities.append(self.LIMIT[basket_product] - current_basket_pos)
        else:
            max_quantities.append(self.LIMIT[basket_product] + current_basket_pos)
        
        return min(max_quantities, key=abs) if max_quantities else 0

    def get_swmid(self, order_depth: OrderDepth):
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float("inf")
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        return (best_ask + best_bid) / 2 if best_ask < float("inf") and best_bid > 0 else 0

    def get_synthetic_basket_order_depth(self, order_depths, basket_product):
        basket_weights = BASKET_COMPOSITION[basket_product]
        synthetic_order_depth = OrderDepth()

        best_bids = []
        best_asks = []
        bid_volumes = []
        ask_volumes = []

        for product, weight in basket_weights.items():
            buy_orders = order_depths[product].buy_orders
            sell_orders = order_depths[product].sell_orders

            best_bid = max(buy_orders.keys()) if buy_orders else 0
            best_ask = min(sell_orders.keys()) if sell_orders else float('inf')

            best_bids.append(best_bid * weight)
            best_asks.append(best_ask * weight)

            bid_volumes.append(buy_orders.get(best_bid, 0) // weight if best_bid in buy_orders else 0)
            ask_volumes.append(-sell_orders.get(best_ask, 0) // weight if best_ask in sell_orders else 0)

        implied_bid = sum(best_bids)
        implied_ask = sum(best_asks)

        if implied_bid > 0:
            synthetic_order_depth.buy_orders[implied_bid] = min(bid_volumes)
        if implied_ask < float("inf"):
            synthetic_order_depth.sell_orders[implied_ask] = -min(ask_volumes)

        return synthetic_order_depth

    def execute_spread_orders(self, basket_product, target_position, current_position, order_depths, state):
        if target_position == current_position:
            return None

        desired_quantity = target_position - current_position
        max_quantity = self.get_max_basket_quantity(basket_product, desired_quantity, state.position)
        
        if (desired_quantity > 0 and max_quantity <= 0) or (desired_quantity < 0 and max_quantity >= 0):
            return None

        basket_depth = order_depths[basket_product]
        synthetic_depth = self.get_synthetic_basket_order_depth(order_depths, basket_product)

        if desired_quantity > 0:  # Buying
            basket_price = min(basket_depth.sell_orders.keys())
            basket_volume = abs(basket_depth.sell_orders[basket_price])
            synthetic_price = max(synthetic_depth.buy_orders.keys())
            synthetic_volume = synthetic_depth.buy_orders[synthetic_price]
            
            actual_quantity = min(
                max_quantity,
                basket_volume,
                synthetic_volume
            )
            
            if actual_quantity <= 0:
                return None
                
            basket_orders = [Order(basket_product, basket_price, actual_quantity)]
            synthetic_orders = [Order("SYNTHETIC", synthetic_price, -actual_quantity)]
        else:  # Selling
            basket_price = max(basket_depth.buy_orders.keys())
            basket_volume = basket_depth.buy_orders[basket_price]
            synthetic_price = min(synthetic_depth.sell_orders.keys())
            synthetic_volume = -synthetic_depth.sell_orders[synthetic_price]
            
            actual_quantity = min(
                abs(max_quantity),
                basket_volume,
                synthetic_volume
            )
            
            if actual_quantity <= 0:
                return None
                
            basket_orders = [Order(basket_product, basket_price, -actual_quantity)]
            synthetic_orders = [Order("SYNTHETIC", synthetic_price, actual_quantity)]

        component_orders = self.convert_to_component_orders(synthetic_orders, order_depths, basket_product)
        component_orders[basket_product] = basket_orders
        
        return self.adjust_order_quantities(component_orders, state)

    def convert_to_component_orders(self, synthetic_orders, order_depths, basket_product):
        basket_weights = BASKET_COMPOSITION[basket_product]
        component_orders = {product: [] for product in basket_weights}

        for order in synthetic_orders:
            quantity = order.quantity
            for product, weight in basket_weights.items():
                if quantity > 0:
                    price = min(order_depths[product].sell_orders.keys())
                else:
                    price = max(order_depths[product].buy_orders.keys())
                component_orders[product].append(Order(product, price, quantity * weight))

        return component_orders

    def adjust_order_quantities(self, orders, state):
        adjusted = {}
        for product, order_list in orders.items():
            new_orders = []
            current_pos = state.position.get(product, 0)
            for order in order_list:
                limit = self.LIMIT.get(product, float('inf'))
                
                if order.quantity > 0:  # Buying
                    adjusted_qty = min(order.quantity, limit - current_pos)
                else:  # Selling
                    adjusted_qty = max(order.quantity, -limit - current_pos)
                
                if adjusted_qty != 0:
                    new_orders.append(Order(product, order.price, adjusted_qty))
                    current_pos += adjusted_qty
            adjusted[product] = new_orders
        return adjusted

    def spread_orders(self, order_depths, basket_product, basket_position, spread_data, state):
        if basket_product not in order_depths:
            return None

        basket_depth = order_depths[basket_product]
        synthetic_depth = self.get_synthetic_basket_order_depth(order_depths, basket_product)

        spread = self.get_swmid(basket_depth) - self.get_swmid(synthetic_depth)
        spread_data["spread_history"].append(spread)

        params = self.params[Product.SPREAD if basket_product == Product.PICNIC_BASKET1 else Product.SPREAD2]
        
        if len(spread_data["spread_history"]) < params["spread_std_window"]:
            return None
            
        spread_data["spread_history"] = spread_data["spread_history"][-params["spread_std_window"]:]
        
        std = np.std(spread_data["spread_history"])
        zscore = (spread - params["default_spread_mean"]) / std

        if zscore > params["zscore_threshold"] and basket_position != -params["target_position"]:
            return self.execute_spread_orders(
                basket_product,
                -params["target_position"],
                basket_position,
                order_depths,
                state
            )
        elif zscore < -params["zscore_threshold"] and basket_position != params["target_position"]:
            return self.execute_spread_orders(
                basket_product,
                params["target_position"],
                basket_position,
                order_depths,
                state
            )
        return None

    def run(self, state: TradingState):
        result = {}
        positions = state.position.copy()  # Create a working copy
        
        # Execute strategies in defined priority order
        for basket_product in [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2]:
            if basket_product in state.order_depths:
                strategy = self.spread_orders
                orders = strategy(
                    state.order_depths,
                    basket_product,
                    positions.get(basket_product, 0),
                    self.spread_data[basket_product],
                    state
                )
                
                if orders:
                    for product, order_list in orders.items():
                        result.setdefault(product, []).extend(order_list)
                        # Update our position tracking
                        for order in order_list:
                            positions[product] = positions.get(product, 0) + order.quantity

        # Validate no limits were exceeded
        for product, pos in positions.items():
            if abs(pos) > self.LIMIT.get(product, float('inf')):
                logger.print(f"WARNING: Potential limit breach for {product} at {pos}")

        traderData = jsonpickle.encode({
            "spread_data": self.spread_data
        })

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        
        return result,conversions, traderData