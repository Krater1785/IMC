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


logger = Logger()

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
        self.resin_prices = []
        self.resin_vwap = []
        self.kelp_prices = []
        self.kelp_vwap = []

        self.squid_ink_prices = []
        self.squid_ink_vwap = []
        self.squid_ink_volatility = []
        self.ink_liquidate_quantity = ink_liquidate_quantity
        self.ink_limit = ink_limit
        self.target_price = 10000

        self.prev_mid_prices = []

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.CROISSANTS: 250,
            Product.DJEMBES: 60,
            Product.JAMS: 350,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100
        }

        self.volatility_window = 20
        self.vwap_window = 5  # for slope calculation
        self.vwap_threshold = 1.2

        self.spread_data = {
            Product.PICNIC_BASKET1: {"spread_history": []},
            Product.PICNIC_BASKET2: {"spread_history": []},
        }

        self.params = {
            Product.SPREAD: {
                "spread_std_window": 70,
                "zscore_threshold": 2.5,
                "default_spread_mean": 0,
                "target_position": 60,
            },
            Product.SPREAD2: {
                "spread_std_window": 130,
                "zscore_threshold": 2.5,
                "default_spread_mean": 0,
                "target_position": 100,
            }
        }

    def vwap_slope(self, vwap_list):
        if len(vwap_list) < self.vwap_window:
            return 0
        slope = (vwap_list[-1]["vwap"] - vwap_list[-self.vwap_window]["vwap"]) / self.vwap_window
        return slope
    
    def check_component_limits(self, product: str, order_quantity: int, current_position: int) -> int:
        if product not in self.LIMIT:
            return order_quantity  # No limit defined for this product
        
        max_position = self.LIMIT[product]
        
        if order_quantity > 0:  # Buying
            adjusted_quantity = min(order_quantity, max_position - current_position)
        else:  # Selling
            adjusted_quantity = max(order_quantity, -max_position - current_position)
        
        return adjusted_quantity if adjusted_quantity != 0 else 0

    def get_swmid(self, order_depth: OrderDepth):
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float("inf")
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        return (best_ask + best_bid) / 2 if best_ask < float("inf") and best_bid > 0 else 0

    def get_swvmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
                best_bid_vol + best_ask_vol
        )

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

    def convert_synthetic_basket_orders(self, synthetic_orders, order_depths, basket_product):
        basket_weights = BASKET_COMPOSITION[basket_product]
        component_orders = {product: [] for product in basket_weights}

        for order in synthetic_orders:
            quantity = order.quantity
            price = order.price

            if quantity > 0:
                price_func = lambda p: min(order_depths[p].sell_orders.keys())
            else:
                price_func = lambda p: max(order_depths[p].buy_orders.keys())

            for product, weight in basket_weights.items():
                component_price = price_func(product)
                component_orders[product].append(Order(product, component_price, quantity * weight))

        return component_orders

    def execute_spread_orders(self, basket_product, target_position, current_position, order_depths, state):
        if target_position == current_position:
            return None

        basket_depth = order_depths[basket_product]
        synthetic_depth = self.get_synthetic_basket_order_depth(order_depths, basket_product)

        buy_side = target_position > current_position
        synthetic_orders = []
        basket_orders = []

        if buy_side:
            basket_price = min(basket_depth.sell_orders.keys())
            basket_volume = abs(basket_depth.sell_orders[basket_price])
            synthetic_price = max(synthetic_depth.buy_orders.keys())
            synthetic_volume = synthetic_depth.buy_orders[synthetic_price]
            quantity = min(basket_volume, synthetic_volume, abs(target_position - current_position))

            # Adjust quantity to respect basket limit
            adjusted_quantity = self.check_component_limits(
                basket_product, quantity, current_position
            )
            if adjusted_quantity <= 0:
                return None

            basket_orders = [Order(basket_product, basket_price, adjusted_quantity)]
            synthetic_orders = [Order("SYNTHETIC", synthetic_price, -adjusted_quantity)]
        else:
            basket_price = max(basket_depth.buy_orders.keys())
            basket_volume = basket_depth.buy_orders[basket_price]
            synthetic_price = min(synthetic_depth.sell_orders.keys())
            synthetic_volume = -synthetic_depth.sell_orders[synthetic_price]
            quantity = min(basket_volume, synthetic_volume, abs(current_position - target_position))

            # Adjust quantity to respect basket limit
            adjusted_quantity = self.check_component_limits(
                basket_product, -quantity, current_position
            )
            if adjusted_quantity >= 0:
                return None

            basket_orders = [Order(basket_product, basket_price, adjusted_quantity)]
            synthetic_orders = [Order("SYNTHETIC", synthetic_price, -adjusted_quantity)]

        component_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths, basket_product)
        
        # Adjust component orders to respect their limits
        for product, orders in component_orders.items():
            if not orders:
                continue
            current_pos = state.position.get(product, 0)
            adjusted_quantity = self.check_component_limits(
                product, orders[0].quantity, current_pos
            )
            if adjusted_quantity == 0:
                component_orders[product] = []
            else:
                component_orders[product] = [Order(product, orders[0].price, adjusted_quantity)]

        component_orders[basket_product] = basket_orders
        return component_orders

    def spread_orders(self, order_depths, basket_product, basket_position, spread_data, state):
        if basket_product not in order_depths:
            return None

        basket_depth = order_depths[basket_product]
        synthetic_depth = self.get_synthetic_basket_order_depth(order_depths, basket_product)

        basket_mid = self.get_swmid(basket_depth)
        synthetic_mid = self.get_swmid(synthetic_depth)
        spread = basket_mid - synthetic_mid
        spread_data["spread_history"].append(spread)

        window = self.params[Product.SPREAD]["spread_std_window"]
        threshold = self.params[Product.SPREAD]["zscore_threshold"]
        target_pos = self.params[Product.SPREAD]["target_position"]
        mean = self.params[Product.SPREAD]["default_spread_mean"]

        if len(spread_data["spread_history"]) < window:
            return None
        elif len(spread_data["spread_history"]) > window:
            spread_data["spread_history"].pop(0)

        std = np.std(spread_data["spread_history"])
        zscore = (spread - mean) / std

        if zscore > threshold and basket_position != -target_pos:
            return self.execute_spread_orders(basket_product, -target_pos, basket_position, order_depths, state)

        if zscore < -threshold and basket_position != target_pos:
            return self.execute_spread_orders(basket_product, target_pos, basket_position, order_depths, state)

        return None
    
    def spread2_orders(self, order_depths, basket_product, basket_position, spread_data, state):
        orders = None
        if basket_product not in order_depths:
            return orders
        basket_depth = order_depths[basket_product]
        synthetic_depth = self.get_synthetic_basket_order_depth(order_depths, basket_product)

        basket_swmid = self.get_swvmid(basket_depth)
        synthetic_swmid = self.get_swvmid(synthetic_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)
        if (
                len(spread_data["spread_history"])
                < self.params[Product.SPREAD2]["spread_std_window"]
        ):
            return orders
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD2]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
                         spread - self.params[Product.SPREAD2]["default_spread_mean"]
                 ) / spread_std

        if zscore >= self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD2]["target_position"]:
                orders = self.execute_spread_orders(
                    basket_product,
                    -self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                    state
                )

        elif zscore <= -self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD2]["target_position"]:
                orders = self.execute_spread_orders(
                    basket_product,
                    self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                    state
                )

        spread_data["prev_zscore"] = zscore
        # print("orders:", orders)
        return orders

    def run(self, state: TradingState):
        result = {}
        state
        bs1 = state.position.get(Product.PICNIC_BASKET1, 0)
        bs2 = state.position.get(Product.PICNIC_BASKET2, 0)
        for prod in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]:
            expected_cr = -bs1*BASKET_COMPOSITION[Product.PICNIC_BASKET1][prod]+bs2*BASKET_COMPOSITION[Product.PICNIC_BASKET2].get(prod, 0)
            actual_cr = state.position.get(prod, 0)
            if expected_cr != actual_cr:
                print()
        order_depths = state.order_depths
        positions = state.position
        self.spread_data_picnic_basket_1 = {
        "spread_history": [],
        "prev_zscore": 0.0
        }
        self.spread_data_picnic_basket_2 = {
        "spread_history": [],
        "prev_zscore": 0.0
        }
        order_depths = state.order_depths
        positions = state.position
        self.spread_data_picnic_basket_1 = {
            "spread_history": [],
            "prev_zscore": 0.0
        }
        self.spread_data_picnic_basket_2 = {
            "spread_history": [],
            "prev_zscore": 0.0
        }

        for basket in [Product.PICNIC_BASKET1]:
            component_orders = self.spread_orders(
                order_depths,
                basket,
                positions.get(basket, 0),
                self.spread_data[basket],
                state  
            )
            if component_orders:
                for product, orders in component_orders.items():
                    if product not in result:
                        result[product] = []
                    result[product].extend(orders)

        for basket in [Product.PICNIC_BASKET2]:
            component_orders = self.spread2_orders(
                order_depths,
                basket,
                positions.get(basket, 0),
                self.spread_data[basket],
                state
            )
            if component_orders:
                for product, orders in component_orders.items():
                    if product not in result:
                        result[product] = []
                    result[product].extend(orders)

        traderData = jsonpickle.encode({
            "resin_prices": self.resin_prices,
            "resin_vwap": self.resin_vwap,
            "kelp_prices": self.kelp_prices,
            "kelp_vwap": self.kelp_vwap,
            "squid_ink_prices": self.squid_ink_prices,
            "squid_ink_vwap": self.squid_ink_vwap,
            "squid_ink_volatility": self.squid_ink_volatility,
            "spread_data_picnic_basket_1": self.spread_data_picnic_basket_1,
            "spread_data_picnic_basket_2": self.spread_data_picnic_basket_2
        })

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData