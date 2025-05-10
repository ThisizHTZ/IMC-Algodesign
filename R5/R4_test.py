import json
import math
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, Product
from statistics import NormalDist
from typing import Any, Dict, List, Tuple
import jsonpickle
import numpy as np

JSON = dict[str, Any] | list[Any] | str | int | float | bool | None

def BS_CALL(S: float, K: float, T: float, r: float, sigma: float) -> float:
    N = NormalDist().cdf
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * N(d1) - K * math.exp(-r * T) * N(d2)

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
        max_item_length = (self.max_log_length - base_length) // 3
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
        return value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self) -> None:
        self.LIMIT = {
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
        }
        self.strategies = {
            # ... existing option strategies initialization ...
        }
        self.params = PARAMS
        self.trader_data: dict = {}  # 用于存储价差历史等

    # —— 1. Picnic Basket 1 现货套利 —— #
    def compute_orders_picnic1(self, order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        basket_pos = self.position.get(Product.PICNIC_BASKET1, 0)
        arb = self.spread_orders(order_depths, Product.PICNIC_BASKET1, basket_pos,
                                 self.trader_data.get(Product.SPREAD, {}))
        if not arb:
            return {}
        return {
            Product.CROISSANTS: arb.get(Product.CROISSANTS, []),
            Product.JAMS:       arb.get(Product.JAMS, []),
            Product.DJEMBES:    arb.get(Product.DJEMBES, []),
            Product.PICNIC_BASKET1: arb.get(Product.PICNIC_BASKET1, [])
        }

    # —— 2. Picnic Basket 2 现货套利 —— #
    def compute_orders_picnic2(self, order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        basket2_pos = self.position.get(Product.PICNIC_BASKET2, 0)
        arb = self.spread2_orders(order_depths, Product.PICNIC_BASKET2, basket2_pos,
                                  self.trader_data.get(Product.SPREAD2, {}))
        if not arb:
            return {}
        return {
            Product.CROISSANTS: arb.get(Product.CROISSANTS, []),
            Product.JAMS:       arb.get(Product.JAMS, []),
            Product.PICNIC_BASKET2: arb.get(Product.PICNIC_BASKET2, [])
        }

    # —— 3. Rainforest Resin 做市/吃单 —— #
    def compute_orders_resin(self, order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        if Product.RAINFOREST_RESIN not in order_depths or not self.params[Product.RAINFOREST_RESIN]["main_switch"]:
            return {}
        od = order_depths[Product.RAINFOREST_RESIN]
        pos = self.position.get(Product.RAINFOREST_RESIN, 0)
        orders: List[Order] = []
        fv = self.params[Product.RAINFOREST_RESIN]["fair_value"]
        # 吃对手档
        if od.sell_orders and min(od.sell_orders) < fv:
            qty = min(-od.sell_orders[min(od.sell_orders)], self.LIMIT[Product.RAINFOREST_RESIN] - pos)
            if qty>0: orders.append(Order(Product.RAINFOREST_RESIN, min(od.sell_orders), qty))
        if od.buy_orders and max(od.buy_orders) > fv:
            qty = min(od.buy_orders[max(od.buy_orders)], self.LIMIT[Product.RAINFOREST_RESIN] + pos)
            if qty>0: orders.append(Order(Product.RAINFOREST_RESIN, max(od.buy_orders), -qty))
        # 挂做市单
        lower = max([p for p in od.buy_orders if p < fv-1], default=fv-1) + 1
        upper = min([p for p in od.sell_orders if p > fv+1], default=fv+1) - 1
        bid_qty = self.LIMIT[Product.RAINFOREST_RESIN] - (pos + sum(o.quantity for o in orders if o.quantity>0))
        ask_qty = self.LIMIT[Product.RAINFOREST_RESIN] + (pos - sum(-o.quantity for o in orders if o.quantity<0))
        if bid_qty>0: orders.append(Order(Product.RAINFOREST_RESIN, round(lower), bid_qty))
        if ask_qty>0: orders.append(Order(Product.RAINFOREST_RESIN, round(upper), -ask_qty))
        return { Product.RAINFOREST_RESIN: orders }

    # —— 4. Kelp 做市/吃单 —— #
    def compute_orders_kelp(self, order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        if Product.KELP not in order_depths or not self.params[Product.KELP]["main_switch"]:
            return {}
        od = order_depths[Product.KELP]
        pos = self.position.get(Product.KELP, 0)
        orders: List[Order] = []
        best_ask = min(od.sell_orders)
        best_bid = max(od.buy_orders)
        filtered_asks = [p for p,v in od.sell_orders.items() if abs(v)>=self.params[Product.KELP]["adverse_volume"]]
        filtered_bids = [p for p,v in od.buy_orders.items() if abs(v)>=self.params[Product.KELP]["adverse_volume"]]
        mm_ask = min(filtered_asks, default=best_ask)
        mm_bid = max(filtered_bids, default=best_bid)
        # VWAP
        bid_vwap = sum(p*v for p,v in od.buy_orders.items())/sum(od.buy_orders.values())
        ask_vwap = sum(p*abs(v) for p,v in od.sell_orders.items())/sum(abs(v) for v in od.sell_orders.values())
        fv = (bid_vwap+ask_vwap)/2 + self.params[Product.KELP]["upward_bias"]
        # 吃单
        if best_ask <= fv - self.params[Product.KELP]["take_width"]:
            qty = min(-od.sell_orders[best_ask], self.LIMIT[Product.KELP]-pos)
            if qty>0: orders.append(Order(Product.KELP, best_ask, qty))
        if best_bid >= fv + self.params[Product.KELP]["take_width"]:
            qty = min(od.buy_orders[best_bid], self.LIMIT[Product.KELP]+pos)
            if qty>0: orders.append(Order(Product.KELP, best_bid, -qty))
        # 做市单
        lower = max([p for p in od.buy_orders if p < fv-1], default=fv-1) + 1
        upper = min([p for p in od.sell_orders if p > fv+1], default=fv+1) - 1
        bid_qty = self.LIMIT[Product.KELP] - (pos + sum(o.quantity for o in orders if o.quantity>0))
        ask_qty = self.LIMIT[Product.KELP] + (pos - sum(-o.quantity for o in orders if o.quantity<0))
        if bid_qty>0: orders.append(Order(Product.KELP, round(lower), bid_qty))
        if ask_qty>0: orders.append(Order(Product.KELP, round(upper), -ask_qty))
        return { Product.KELP: orders }

    # —— 修改 run，调用这些现货策略 —— #
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        # 恢复历史 state.traderData 到 self.trader_data
        if state.traderData:
            try:
                self.trader_data = jsonpickle.decode(state.traderData)
            except:
                self.trader_data = {}
        # 更新持仓
        self.position = state.position.copy()

        result: Dict[str, List[Order]] = {}
        conversions = 0

        # （先执行已有的期权策略，将 orders 加入 result，并累加 conversions）
        for sym, strat in self.strategies.items():
            if sym in state.order_depths:
                orders, conv = strat.run(state)
                result.setdefault(sym, []).extend(orders)
                conversions += conv

        # —— 执行现货套利策略 —— #
        for fn in (
            self.compute_orders_picnic1,
            self.compute_orders_picnic2,
            self.compute_orders_resin,
            self.compute_orders_kelp
        ):
            arb = fn(state.order_depths)
            for sym, olist in arb.items():
                result.setdefault(sym, []).extend(olist)

        # 序列化 self.trader_data，输出日志
        traderData = jsonpickle.encode(self.trader_data)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
