from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import math
import jsonpickle

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

class Trader:
    def __init__(self):
        self.resin_prices = []
        self.resin_vwap = []
        self.kelp_prices = []
        self.kelp_vwap = []
        
        self.squid_ink_prices = []
        self.squid_ink_vwap = []
        self.squid_ink_volatility = []

        self.prev_mid_prices = []

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50
        }
        self.volatility_window = 20
        self.vwap_window = 5  # for slope calculation
        self.vwap_threshold = 1.2

    def vwap_slope(self, vwap_list):
        if len(vwap_list) < self.vwap_window:
            return 0
        slope = (vwap_list[-1]["vwap"] - vwap_list[-self.vwap_window]["vwap"]) / self.vwap_window
        return slope

    def squid_ink_trend_follower(self, product, order_depth, position, prices, vwap_list, timespan):
        orders = []
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]

            mid_price = (best_ask + best_bid) / 2
            prices.append(mid_price)

            vwap = (best_bid * -order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume if volume != 0 else mid_price
            vwap_list.append({"vol": volume, "vwap": vwap})

            if len(prices) > timespan:
                prices.pop(0)
            if len(vwap_list) > timespan:
                vwap_list.pop(0)

            slope = self.vwap_slope(vwap_list)
            diff = vwap - mid_price
            pos_limit = self.LIMIT[product]

            recent_dip = len(self.prev_mid_prices) > 2 and self.prev_mid_prices[-1] < self.prev_mid_prices[-2] < self.prev_mid_prices[-3]
            self.prev_mid_prices.append(mid_price)
            if len(self.prev_mid_prices) > 5:
                self.prev_mid_prices.pop(0)

            if abs(diff) > self.vwap_threshold:
                if diff > 0:
                    if slope > 0:
                        if position < 0:
                            buy_qty = min(pos_limit + position, order_depth.sell_orders[best_ask])
                            if buy_qty > 0:
                                orders.append(Order(product, best_ask, buy_qty))  # clear short
                        elif position == 0 and recent_dip:
                            buy_qty = min(pos_limit, order_depth.sell_orders[best_ask])
                            if buy_qty > 0:
                                orders.append(Order(product, best_ask, buy_qty))  # enter long after dip

                elif diff < 0:
                    if slope > 0 and position < pos_limit:
                        buy_qty = min(pos_limit - position, order_depth.sell_orders[best_ask])
                        if buy_qty > 0:
                            orders.append(Order(product, best_ask, buy_qty))  # buy expecting recovery

            if slope < 0 and position > 0:
                sell_qty = min(position, order_depth.buy_orders[best_bid])
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))  # clear long

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

    def run(self, state: TradingState):
        result = {}

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

        result[Product.RAINFOREST_RESIN] = resin_orders
        result[Product.KELP] = kelp_orders

        traderData = jsonpickle.encode({
            "resin_prices": self.resin_prices,
            "resin_vwap": self.resin_vwap,
            "kelp_prices": self.kelp_prices,
            "kelp_vwap": self.kelp_vwap,
            "squid_ink_prices": self.squid_ink_prices,
            "squid_ink_vwap": self.squid_ink_vwap,
            "squid_ink_volatility": self.squid_ink_volatility
        })

        conversions = 1
        return result, conversions, traderData
