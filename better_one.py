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
    def __init__(self,ink_limit=22,ink_liquidate_quantity=7):
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
        self.high_djembes_ask = 13509.0
        self.low_djembes_ask = 13420.0
        self.high_djembes_bid = 13509.0
        self.low_djembes_bid = 13420.0
        self.high_croissant_ask = 4341.0
        self.low_croissants_ask = 4303.5
        self.high_croissant_bid = 4317.0
        self.low_croissants_bid = 4302.5
        self.high_jams_ask = 6703.5
        self.low_jams_ask=6616.5
        self.high_jams_bid = 6701.5
        self.low_jams_bid = 6614.5

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
                "spread_std_window": 40,
                "zscore_threshold": 1.3,
                "default_spread_mean": 0,
                "target_position": 3,
            },
            Product.SPREAD2:{
                "spread_std_window": 20,
                "zscore_threshold": 1.1,
                "default_spread_mean": 0,
                "target_position": 3,
            }
        }

    def adaptive_threshold(self, spread_history, base_threshold=1.1):
        if len(spread_history) < 2:
            return base_threshold
        volatility = np.std(spread_history)
        mean_spread = np.mean(spread_history)
        return base_threshold + 0.5 * (volatility / mean_spread) if mean_spread != 0 else base_threshold


    def vwap_slope(self, vwap_list):
        if len(vwap_list) < self.vwap_window:
            return 0
        slope = (vwap_list[-1]["vwap"] - vwap_list[-self.vwap_window]["vwap"]) / self.vwap_window
        return slope

    def squid_ink_trend_follower(self, product, order_depth, position, prices, vwap_list, timespan) -> list[Order]:
        orders = []
        limit = self.ink_limit
        liquidate_threshold = self.ink_liquidate_quantity
        limit_buy, limit_sell = self.ink_limit - position, -self.ink_limit - position

        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            ask_vol = -order_depth.sell_orders[best_ask]  # Fix: negate sell order volume
            bid_vol = order_depth.buy_orders[best_bid]

            volume = ask_vol + bid_vol
            mid_price = (best_ask + best_bid) / 2
            prices.append(mid_price)

            vwap = (best_bid * ask_vol + best_ask * bid_vol) / volume if volume != 0 else mid_price
            vwap_list.append({"vol": volume, "vwap": vwap})

            if len(prices) > timespan:
                prices.pop(0)
            if len(vwap_list) > timespan:
                vwap_list.pop(0)

            slope = self.vwap_slope(vwap_list)
            diff = vwap - mid_price

            # --- Liquidation logic ---
            filtered_asks = [p for p in order_depth.sell_orders if abs(order_depth.sell_orders[p]) >= 15]
            filtered_bids = [p for p in order_depth.buy_orders if abs(order_depth.buy_orders[p]) >= 15]

            mm_ask = min(filtered_asks) if filtered_asks else best_ask
            mm_bid = max(filtered_bids) if filtered_bids else best_bid

            if position >= liquidate_threshold and filtered_asks:
                sell_price = mm_ask-2
                quantity = -position
                orders.append(Order(product, sell_price, quantity))
                limit_sell -= quantity

            elif position <= -liquidate_threshold and filtered_bids:
                buy_price = mm_bid+2
                quantity = -position
                limit_buy -= quantity
                orders.append(Order(product, buy_price, quantity))
               
            if mm_bid is not None and limit_buy > 0 and position <liquidate_threshold:
                bid_price = mm_bid + 1
                available_trades = calculate_buy_quantity(order_depth, bid_price)
                quantity = min(limit_buy, available_trades)
                orders.append(Order(product, bid_price, quantity))

            if mm_ask is not None and limit_sell < 0 and position > -liquidate_threshold:
                ask_price = mm_ask - 1
                available_trades = calculate_sell_quantity(order_depth, ask_price)
                quantity = max(limit_sell, available_trades)
                orders.append(Order(product, ask_price, quantity))
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

    def take_best_orders(self, product, fair_value, take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, prevent_adverse=False, adverse_volume=0):
        position_limit = self.LIMIT[product]

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if (not prevent_adverse and best_ask <= fair_value - take_width) or (prevent_adverse and best_ask_amount <= adverse_volume and best_ask <= fair_value - take_width):
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if (not prevent_adverse and best_bid >= fair_value + take_width) or (prevent_adverse and best_bid_amount <= adverse_volume and best_bid >= fair_value + take_width):
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity

        return buy_order_volume, sell_order_volume

    def clear_position_order(self, product, fair_value, orders, order_depth, position, buy_order_volume, sell_order_volume):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
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

    def market_make(self, product, orders, bid, ask, position, buy_order_volume, sell_order_volume):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bid, buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))

        return buy_order_volume, sell_order_volume
    
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

    def execute_spread_orders(self, basket_product, target_position, current_position, order_depths):
        if target_position == current_position:
            return None

        basket_depth = order_depths[basket_product]
        synthetic_depth = self.get_synthetic_basket_order_depth(order_depths, basket_product)

        buy_side = target_position > current_position
        synthetic_orders = []
        basket_orders = [ ]

        if buy_side:
            basket_price = min(basket_depth.sell_orders.keys())
            basket_volume = abs(basket_depth.sell_orders[basket_price])
            synthetic_price = max(synthetic_depth.buy_orders.keys())
            synthetic_volume = synthetic_depth.buy_orders[synthetic_price]
            quantity = min(basket_volume, synthetic_volume, abs(target_position - current_position))
            
            if(basket_product=="PICNIC_BASKET1"):
                if(basket_price>self.high_croissant_bid*  0.5742696414340938*6+0.7921546943766852*self.high_jams_bid*3+0.5199028111233017*self.high_djembes_bid):
                    basket_orders = [Order(basket_product, basket_price, quantity)]
                    synthetic_orders = [Order("SYNTHETIC", synthetic_price, -quantity)]
            elif(basket_product=="PICNIC_BASKET2"):
                if(basket_price>self.high_croissant_bid*0.3989135923827439*4+0.7705782181434639*self.high_jams_bid*2):
                    basket_orders = [Order(basket_product, basket_price, quantity)]
                    synthetic_orders = [Order("SYNTHETIC", synthetic_price, -quantity)]
        else:
            basket_price = max(basket_depth.buy_orders.keys())
            basket_volume = basket_depth.buy_orders[basket_price]
            synthetic_price = min(synthetic_depth.sell_orders.keys())
            synthetic_volume = -synthetic_depth.sell_orders[synthetic_price]
            quantity = min(basket_volume, synthetic_volume, abs(target_position - current_position))
            if(basket_product=="PICNIC_BASKET1"):
                if(basket_price<self.high_croissant_ask* 0.5756399965417093*6+0.7919562851494966*self.high_jams_ask*3+0.5199208434198621*self.high_djembes_ask):
                    basket_orders = [Order(basket_product, basket_price, quantity)]
                    synthetic_orders = [Order("SYNTHETIC", synthetic_price, -quantity)]
            elif(basket_product=="PICNIC_BASKET2"):
                if(basket_price<self.high_croissant_ask*0.3995876009876715*4+0.770233216802427*self.high_jams_ask*2):
                    basket_orders = [Order(basket_product, basket_price, quantity)]
                    synthetic_orders = [Order("SYNTHETIC", synthetic_price, -quantity)]

        component_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths, basket_product)
        component_orders[basket_product] = basket_orders
        return component_orders

    def spread_orders(self, order_depths, basket_product, basket_position, spread_data):
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
            return self.execute_spread_orders(basket_product, -target_pos, basket_position, order_depths)

        if zscore < -threshold and basket_position != target_pos:
            return self.execute_spread_orders(basket_product, target_pos, basket_position, order_depths)

        return None
    
    def spread2_orders(self, order_depths, basket_product, basket_position, spread_data):
        if basket_product not in order_depths:
            return None
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
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD2]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
            spread - self.params[Product.SPREAD2]["default_spread_mean"]
        ) / spread_std

        if zscore >= self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread_orders(
                    basket_product,
                    -self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread_orders(
                    basket_product,
                    self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None


    def run(self, state: TradingState):
        result = {}
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


        resin_pos = state.position.get(Product.RAINFOREST_RESIN, 0)
        kelp_pos = state.position.get(Product.KELP, 0)

        resin_orders = self.hybrid_orders(Product.RAINFOREST_RESIN, state.order_depths[Product.RAINFOREST_RESIN], resin_pos, self.resin_prices, self.resin_vwap, 10, 1, 3.5)
        kelp_orders = self.hybrid_orders(Product.KELP, state.order_depths[Product.KELP], kelp_pos, self.kelp_prices, self.kelp_vwap, 10, 1, 3.5)

        if Product.SQUID_INK in state.order_depths:
            squid_ink_pos = state.position.get(Product.SQUID_INK, 0)
            squid_ink_orders = self.squid_ink_trend_follower(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                squid_ink_pos,
                self.squid_ink_prices,
                self.squid_ink_vwap,
                20
            )
            result[Product.SQUID_INK] = squid_ink_orders
        
        for basket in [Product.PICNIC_BASKET1]:
            component_orders = self.spread_orders(
                order_depths,
                basket,
                positions.get(basket, 0),
                self.spread_data[basket]
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
                self.spread_data[basket]
            )
            if component_orders:
                for product, orders in component_orders.items():
                    if product not in result:
                        result[product] = []
                    result[product].extend(orders)


        result[Product.RAINFOREST_RESIN] = resin_orders
        result[Product.KELP] = kelp_orders

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
        return result, conversions, traderData