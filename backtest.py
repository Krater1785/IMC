# from one_day_Trader import Trader
from plotters import Plotter
from datamodel import *
import pandas as pd
import statistics
import copy
import os

from consts import *

# from Strategy2023.trader import Trader
from traders.round1.submission1 import Trader
from backtester_logging import create_log_file


def process_prices(df_prices, round_, time_limit) -> dict[int, TradingState]:
    states = {}
    for _, row in df_prices.iterrows():
        time: int = int(row["timestamp"])
        if time > time_limit:
            break
        product: str = row["product"]
        if states.get(time) == None:
            position: Dict[Product, Position] = {}
            own_trades: Dict[Symbol, List[Trade]] = {}
            market_trades: Dict[Symbol, List[Trade]] = {}
            observations: Dict[Product, Observation] = {}
            listings = {}
            depths = {}
            states[time] = TradingState(time, listings, depths, own_trades, market_trades, position,
                                        observations)

        if product not in states[time].position and product in SYMBOLS_BY_ROUND_POSITIONABLE[round_]:
            states[time].position[product] = 0
            states[time].own_trades[product] = []
            states[time].market_trades[product] = []

        states[time].listings[product] = Listing(product, product, "1")

        if product == "DOLPHIN_SIGHTINGS":
            states[time].observations["DOLPHIN_SIGHTINGS"] = row['mid_price']

        depth = OrderDepth()
        if row["bid_price_1"] > 0:
            depth.buy_orders[row["bid_price_1"]] = int(row["bid_volume_1"])
        if row["bid_price_2"] > 0:
            depth.buy_orders[row["bid_price_2"]] = int(row["bid_volume_2"])
        if row["bid_price_3"] > 0:
            depth.buy_orders[row["bid_price_3"]] = int(row["bid_volume_3"])
        if row["ask_price_1"] > 0:
            depth.sell_orders[row["ask_price_1"]] = -int(row["ask_volume_1"])
        if row["ask_price_2"] > 0:
            depth.sell_orders[row["ask_price_2"]] = -int(row["ask_volume_2"])
        if row["ask_price_3"] > 0:
            depth.sell_orders[row["ask_price_3"]] = -int(row["ask_volume_3"])
        states[time].order_depths[product] = depth

    return states


def process_trades(df_trades, states: dict[int, TradingState], time_limit, names=True, lag=0):
    '''
    add trades to the book orders. No order exercise included
    :param df_trades:
    :param states:
    :param time_limit:
    :param names:
    :return:
    '''
    for _, trade in df_trades.iterrows():
        time: int = trade['timestamp']
        if time > time_limit:
            break
        symbol = trade['symbol']
        if symbol not in states[time].market_trades:
            states[time].market_trades[symbol] = []
        t = Trade(
            symbol,
            trade['price'],
            trade['quantity'],
            str(trade['buyer']),
            str(trade['seller']),
            time)
        states[time + lag*TIME_DELTA].market_trades[symbol].append(t)
    return states


def calc_mid(states: dict[int, TradingState], round_: int, time: int, max_time: int) -> dict[
    str, float]:
    medians_by_symbol = {}
    non_empty_time = time

    for psymbol in SYMBOLS_BY_ROUND_POSITIONABLE[round_]:
        hitted_zero = False
        while len(states[non_empty_time].order_depths[psymbol].sell_orders.keys()) == 0 or len(
                states[non_empty_time].order_depths[psymbol].buy_orders.keys()) == 0:
            # little hack
            if time == 0 or hitted_zero and time != max_time:
                hitted_zero = True
                non_empty_time += TIME_DELTA
            else:
                non_empty_time -= TIME_DELTA
        min_ask = min(states[non_empty_time].order_depths[psymbol].sell_orders.keys())
        max_bid = max(states[non_empty_time].order_depths[psymbol].buy_orders.keys())
        median_price = statistics.median([min_ask, max_bid])
        medians_by_symbol[psymbol] = median_price
    return medians_by_symbol


# Setting a high time_limit can be harder to visualize
# print_position prints the position before! every Trader.run
def simulate_alternative(
        round_: int,
        day: int,
        trader,
        time_limit=999900,
        names=True,
        logging=True,
        plotting=True,
        verbose=True,
        df_prices=None,  # df can be passed to avoid reading from file
        df_trades=None,  # df can be passed to avoid reading from file
        plot_symbols=None,  # symbols to plot None = all :)
        trades_lag=0,  # lag in trades
):
    '''main function that parse trades from csv and runs the simulation. After that runs plots and visualize them'''
    if df_prices is None:
        prices_path = os.path.join(TRAINING_DATA_PREFIX, f'prices_round_{round_}_day_{day}.csv')
        df_prices = pd.read_csv(prices_path, sep=';')

    if df_trades is None:
        trades_path = os.path.join(TRAINING_DATA_PREFIX, f'trades_round_{round_}_day_{day}.csv')
        if not names:
            # change the path if names arrive!
            trades_path = os.path.join(TRAINING_DATA_PREFIX, f'trades_round_{round_}_day_{day}.csv')

        # check if the file exists
        if os.path.exists(trades_path):
            df_trades = pd.read_csv(trades_path, sep=';', dtype={'seller': str, 'buyer': str})

    states = process_prices(df_prices, round_, time_limit)
    if df_trades is not None:
        states = process_trades(df_trades, states, time_limit, names, trades_lag)
    ref_symbols = list(states[0].position.keys())
    max_time = max(list(states.keys()))

    # handling these four is rather tricky
    profits_by_symbol: dict[int, dict[str, float]] = {
        0: dict(zip(ref_symbols, [0.0] * len(ref_symbols)))
    }
    # balance_by_symbol: dict[int, dict[str, float]] = {0: copy.deepcopy(profits_by_symbol[0])}
    # credit_by_symbol: dict[int, dict[str, float]] = {0: copy.deepcopy(profits_by_symbol[0])}
    # unrealized_by_symbol: dict[int, dict[str, float]] = {0: copy.deepcopy(profits_by_symbol[0])}

    balance_by_symbol: dict[int, dict[str, float]] = {0: profits_by_symbol[0]}
    credit_by_symbol: dict[int, dict[str, float]] = {0: profits_by_symbol[0]}
    unrealized_by_symbol: dict[int, dict[str, float]] = {0: profits_by_symbol[0]}


    states, profits_by_symbol, balance_by_symbol, trader_orders = trades_position_pnl_run(states, max_time, trader,
                                                                           profits_by_symbol,
                                                                           balance_by_symbol,
                                                                           credit_by_symbol,
                                                                           unrealized_by_symbol,
                                                                           round_,
                                                                           verbose=verbose)
    if logging:
        create_log_file(round_, day, states, profits_by_symbol, balance_by_symbol, trader)
    if plotting:
        kwargs = {
            "states": states,
            "trader": trader,
            "profits_by_symbol": profits_by_symbol,
            "balance_by_symbol": balance_by_symbol,
            "trader_orders": trader_orders,
        }
        if plot_symbols is not None:
            plotter = Plotter(plot_symbols, **kwargs)
        else:
            plotter = Plotter(SYMBOLS_BY_ROUND_POSITIONABLE[round_], **kwargs)
        plotter.plot_stats()
    res = {}
    for symbol in SYMBOLS_BY_ROUND_POSITIONABLE[round_]:
        # res[symbol] = profits_by_symbol[max_time][symbol] # + balance_by_symbol[max_time][symbol]  # ??
        # print(balance_by_symbol[max_time][symbol])
        res[symbol] = profits_by_symbol[max_time][symbol] + balance_by_symbol[max_time][symbol]  # ??
    return res


def trades_position_pnl_run(
        states: dict[int, TradingState],
        max_time: int,
        trader: Trader,
        profits_by_symbol: dict[int, dict[str, float]],
        balance_by_symbol: dict[int, dict[str, float]],
        credit_by_symbol: dict[int, dict[str, float]],
        unrealized_by_symbol: dict[int, dict[str, float]],
        round_,
        verbose=True,
):
    spent_buy = 0  # variable is useless, only double check the profits in the end
    spent_sell = 0
    trader_orders = {}
    for time, state in states.items():
        position = copy.deepcopy(state.position)
        orders, _, _ = trader.run(state)
        trader_orders[time] = orders
        trades = clear_order_book(orders, state.order_depths, time, position)
        mids = calc_mid(states, round_, time, max_time)
        if time != max_time:
            profits_by_symbol[time + TIME_DELTA] = copy.deepcopy(profits_by_symbol[time])
            credit_by_symbol[time + TIME_DELTA] = copy.deepcopy(credit_by_symbol[time])
            balance_by_symbol[time + TIME_DELTA] = copy.deepcopy(balance_by_symbol[time])
            unrealized_by_symbol[time + TIME_DELTA] = copy.deepcopy(unrealized_by_symbol[time])
            for psymbol in SYMBOLS_BY_ROUND_POSITIONABLE[round_]:
                unrealized_by_symbol[time + TIME_DELTA][psymbol] = mids[psymbol] * position[psymbol]

        valid_trades = trades
        grouped_by_symbol = {}

        FLEX_TIME_DELTA = TIME_DELTA
        if time == max_time:
            FLEX_TIME_DELTA = 0
        for valid_trade in valid_trades:
            # print(f'considering trade: {valid_trade.quantity} for {valid_trade.price} at time {time}')
            # print(f'position: {position[valid_trade.symbol]}')

            if grouped_by_symbol.get(valid_trade.symbol) is None:
                grouped_by_symbol[valid_trade.symbol] = []
            grouped_by_symbol[valid_trade.symbol].append(valid_trade)
            if valid_trade.quantity > 0:
                spent_buy += valid_trade.price * valid_trade.quantity
            else:
                spent_sell += -valid_trade.price * valid_trade.quantity

            new_credit, profit = calculate_credit_and_profit(valid_trade,
                                                             position[valid_trade.symbol],
                                                             credit_by_symbol[
                                                                 time + FLEX_TIME_DELTA][
                                                                 valid_trade.symbol])
            profits_by_symbol[time + FLEX_TIME_DELTA][valid_trade.symbol] += profit
            credit_by_symbol[time + FLEX_TIME_DELTA][valid_trade.symbol] = new_credit
            position[valid_trade.symbol] += valid_trade.quantity
            if abs(position[valid_trade.symbol]) > current_limits[valid_trade.symbol]:
                # should not happen, but still:
                trades_str = [str(x.quantity) + " for " + str(x.price) for x in valid_trades]
                print(f"trades: {trades_str}")
                print(
                    f'Position limit exceeded: {position[valid_trade.symbol]}, illegal trade: {valid_trade.__dict__}, time: {time}')
                raise ValueError('Position limit exceeded - backtester has a bug')

        if states.get(time + FLEX_TIME_DELTA) is not None:
            states[time + FLEX_TIME_DELTA].own_trades = grouped_by_symbol
        for psymbol in SYMBOLS_BY_ROUND_POSITIONABLE[round_]:
            unrealized_by_symbol[time + FLEX_TIME_DELTA][psymbol] = mids[psymbol] * position[
                psymbol]
            balance_by_symbol[time + FLEX_TIME_DELTA][psymbol] = \
                credit_by_symbol[time + FLEX_TIME_DELTA][psymbol] + \
                unrealized_by_symbol[time + FLEX_TIME_DELTA][psymbol]

        if time == max_time:
            if verbose:
                print("End of simulation reached. All positions left are liquidated")
            # i have the feeling this already has been done, and only repeats the same values as before
            for osymbol in position.keys():
                profits_by_symbol[time + FLEX_TIME_DELTA][osymbol] += \
                    unrealized_by_symbol[time + FLEX_TIME_DELTA][osymbol] + \
                    credit_by_symbol[time + FLEX_TIME_DELTA][osymbol]
                balance_by_symbol[time + FLEX_TIME_DELTA][osymbol] = 0
                if position[osymbol] > 0:
                    spent_sell += mids[osymbol] * position[osymbol]
                else:
                    spent_buy += -mids[osymbol] * position[osymbol]

        if states.get(time + FLEX_TIME_DELTA) is not None:
            states[time + FLEX_TIME_DELTA].position = copy.deepcopy(position)
        if verbose and trades:
            print(f'Trades at time {time}: {[x.__dict__ for x in trades]}')
            print(f"Profits after time {time}: {profits_by_symbol[time + TIME_DELTA]}")
    if verbose:
        print(
            f"spent_buy: {spent_buy}, spent_sell: {spent_sell}, spent_sell - spent_buy: {spent_sell - spent_buy}")
    return states, profits_by_symbol, balance_by_symbol, trader_orders


def clear_order_book(trader_orders: dict[str, List[Order]], order_depth: dict[str, OrderDepth],
                     time: int, position: dict[str, int]) -> list[Trade]:
    trades = []
    for symbol in trader_orders.keys():
        if order_depth.get(symbol) is None:
            continue
        symbol_order_depth = copy.deepcopy(order_depth[symbol])
        # t_orders = cleanup_order_volumes(trader_orders[symbol])
        t_orders = trader_orders[symbol]

        pos = position[symbol] if symbol in position else 0
        for order in t_orders:
            order_cp = copy.deepcopy(order)
            # print(f"order: {order_cp.quantity} for {order_cp.price}, position: {pos}")
            if order.quantity < 0:
                # selling
                while order_cp.quantity < 0:
                    potential_matches = list(filter(lambda o: o[0] >= order_cp.price,
                                                    symbol_order_depth.buy_orders.items()))

                    if len(potential_matches) == 0:
                        break

                    match = potential_matches[0]
                    if abs(match[1]) > abs(order_cp.quantity):
                        final_volume = order_cp.quantity
                    else:
                        # this should be negative
                        final_volume = -match[1]

                    max_volume_pos = -current_limits[symbol] - pos
                    final_volume = max(final_volume, max_volume_pos)

                    if final_volume == 0:
                        break

                    trades.append(Trade(symbol, match[0], final_volume, "BOT", "YOU", time))
                    pos += final_volume
                    # print(f"    trade: {final_volume} for {match[0]}, position: {pos}")
                    order_cp.quantity -= final_volume
                    symbol_order_depth.buy_orders[match[0]] += final_volume
                    if symbol_order_depth.buy_orders[match[0]] == 0:
                        symbol_order_depth.buy_orders.pop(match[0])

            if order.quantity > 0:
                # buying
                while order_cp.quantity > 0:
                    potential_matches = list(filter(lambda o: o[0] <= order.price,
                                                    symbol_order_depth.sell_orders.items()))
                    if len(potential_matches) == 0:
                        break
                    match = potential_matches[0]
                    if abs(match[1]) > abs(order.quantity):
                        final_volume = order.quantity
                    else:
                        final_volume = abs(match[1])

                    max_volume_pos = current_limits[symbol] - pos
                    final_volume = min(final_volume, max_volume_pos)

                    if final_volume == 0:
                        break

                    trades.append(Trade(symbol, match[0], final_volume, "YOU", "BOT", time))
                    pos += final_volume

                    # print(f"    trade: {final_volume} for {match[0]}, position: {pos}")
                    order_cp.quantity -= final_volume
                    symbol_order_depth.sell_orders[match[0]] += final_volume
                    if symbol_order_depth.sell_orders[match[0]] == 0:
                        symbol_order_depth.sell_orders.pop(match[0])
    return trades


def calculate_credit_and_profit(trade, position, credit):
    if trade.quantity * position >= 0:
        credit -= trade.quantity * trade.price
        profit = 0
        return credit, profit
    else:
        if abs(trade.quantity) <= abs(position):
            # we sold/bought less than we had
            avg_price = abs(credit) / abs(position)
            profit = (trade.price - avg_price) * (-trade.quantity)
            credit -= trade.quantity * avg_price
            return credit, profit
        else:
            # we sold/bought everything and have new position
            avg_price = abs(credit) / abs(position)
            profit = (trade.price - avg_price) * (position)
            credit = -trade.price * (
                    trade.quantity + position)  # trade quantity and position are different signs
            return credit, profit