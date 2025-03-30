import json
import numpy as np
from typing import Any, Dict, List
from datamodel import Listing, Order, OrderDepth, ProsperityEncoder, Symbol, TradingState

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        """自定义日志记录函数"""
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        """将交易状态压缩后输出"""
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        """压缩交易状态"""
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

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        """压缩商品列表"""
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        """压缩盘口数据"""
        return {symbol: [od.buy_orders, od.sell_orders] for symbol, od in order_depths.items()}

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        """压缩交易记录"""
        return [
            [trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
            for arr in trades.values()
            for trade in arr
        ]

    def compress_observations(self, observations: Observation) -> List[Any]:
        """压缩观察数据"""
        conversion_observations = {product: [
            obs.bidPrice,
            obs.askPrice,
            obs.transportFees,
            obs.exportTariff,
            obs.importTariff,
            obs.sunlightIndex,
        ] for product, obs in observations.conversionObservations.items()}
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        """压缩订单信息"""
        return [[order.symbol, order.price, order.quantity] for arr in orders.values() for order in arr]

    def to_json(self, value: Any) -> str:
        """转换为 JSON 格式"""
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, s: str, length: int) -> str:
        """截断超长字符串"""
        return s if len(s) <= length else s[: length - 3] + "..."


logger = Logger()


class Status:
    POSITION_LIMITS = {"RAINFOREST_RESIN": 50, "KELP": 50}
    _state = None
    _hist_prices = {"RAINFOREST_RESIN": [], "KELP": []}

    @classmethod
    def cls_update(cls, state: TradingState) -> None:
        """更新状态"""
        cls._state = state
        for product, od in state.order_depths.items():
            if od.buy_orders and od.sell_orders:
                mid_price = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2
                cls._hist_prices[product].append(mid_price)

    def __init__(self, product: str):
        self.product = product

    def hist_prices(self, size: int) -> np.ndarray:
        return np.array(self._hist_prices[self.product][-size:], dtype=np.float32)

    @property
    def best_bid(self) -> int:
        return max(self._state.order_depths[self.product].buy_orders.keys())

    @property
    def best_ask(self) -> int:
        return min(self._state.order_depths[self.product].sell_orders.keys())


class Strategy:
    @staticmethod
    def trade_resin(state: Status) -> list[Order]:
        """均值回归 + RSI 策略"""
        orders = []
        limit = Status.POSITION_LIMITS["RAINFOREST_RESIN"]
        if len(state.hist_prices(20)) < 20:
            return orders

        mean_price = np.mean(state.hist_prices(20))
        std_price = np.std(state.hist_prices(20))
        lower_bound = mean_price - 1.2 * std_price
        upper_bound = mean_price + 1.2 * std_price

        # RSI 计算
        price_changes = np.diff(state.hist_prices(14))
        gains = np.maximum(price_changes, 0)
        losses = np.abs(np.minimum(price_changes, 0))
        avg_gain = np.mean(gains) if len(gains) > 0 else 1
        avg_loss = np.mean(losses) if len(losses) > 0 else 1
        rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))

        if state.best_ask < lower_bound and rsi < 30:
            orders.append(Order("RAINFOREST_RESIN", state.best_ask, min(limit, 10)))

        if state.best_bid > upper_bound and rsi > 70:
            orders.append(Order("RAINFOREST_RESIN", state.best_bid, -min(limit, 10)))

        return orders

    @staticmethod
    def trade_kelp(state: Status) -> list[Order]:
        """布林带 + MACD 趋势跟随策略"""
        orders = []
        limit = Status.POSITION_LIMITS["KELP"]
        prices = state.hist_prices(50)
        if len(prices) < 26:
            return orders

        mid_band = np.mean(prices[-20:])
        std = np.std(prices[-20:])
        upper_band = mid_band + 1.5 * std
        lower_band = mid_band - 1.5 * std

        short_ema = np.mean(prices[-12:])
        long_ema = np.mean(prices[-26:])
        macd = short_ema - long_ema

        if macd > 0 and state.best_ask < upper_band:
            orders.append(Order("KELP", state.best_ask, min(limit, 10)))

        if macd < 0 and state.best_bid > lower_band:
            orders.append(Order("KELP", state.best_bid, -min(limit, 10)))

        return orders


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        Status.cls_update(state)
        result = {"RAINFOREST_RESIN": Strategy.trade_resin(Status("RAINFOREST_RESIN")),
                  "KELP": Strategy.trade_kelp(Status("KELP"))}
        logger.flush(state, result, 0, "SAMPLE")
        return result, 0, "SAMPLE"



