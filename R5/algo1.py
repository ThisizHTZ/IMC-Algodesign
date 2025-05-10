import json
import math
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
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

class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SYNTHETIC2 = "SYNTHETIC2"
    SPREAD = "SPREAD"
    SPREAD2 = "SPREAD2"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"

PARAMS = {
    Product.SPREAD: {
        "default_spread_mean": -96.6218621119938,
        "default_spread_std": 73.40634808153631,
        "spread_std_window": 7.5,
        "zscore_threshold": 30,
        "target_position": 60,
    },
    Product.SPREAD2: {
        "default_spread_mean": 99.16378679358289,
        "default_spread_std": 33.526801209539585,
        "spread_std_window": 9,
        "zscore_threshold": 10,
        "target_position": 100,
    },
    Product.RAINFOREST_RESIN: {
        "main_switch": True,
        "take": True,
        "clear": True,
        "make": True,
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.KELP: {
        "main_switch": True,
        "take": True,
        "clear": True,
        "make": True,
        "fval_model": "mean_rev",
        "upward_bias": 0.3,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "ema_alpha": 0.2,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    }
}

BASKET_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}

BASKET2_WEIGHTS = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2,
}

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders = []

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0
        self.act(state)
        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class SignalStrategy(Strategy):
    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        return (popular_buy_price + popular_sell_price) / 2

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.sell_orders.keys())
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        self.buy(price, to_buy)

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.buy_orders.keys())
        position = state.position.get(self.symbol, 0)
        to_sell = self.limit + position
        self.sell(price, to_sell)

class VolcanicRockStrategy(SignalStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.threshold = None
        self.price_history = deque(maxlen=20)

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            return
        price = self.get_mid_price(state, self.symbol)
        self.price_history.append(price)
        if self.threshold is None:
            self.threshold = price
        if len(self.price_history) == self.price_history.maxlen:
            self.threshold = sum(self.price_history) / len(self.price_history)
        if price > self.threshold * 1.01:
            self.go_long(state)
        elif price < self.threshold * 0.99:
            self.go_short(state)

    def save(self) -> JSON:
        return {"threshold": self.threshold, "price_history": list(self.price_history)}

    def load(self, data: JSON) -> None:
        self.threshold = data.get("threshold")
        self.price_history = deque(data.get("price_history", []), maxlen=20)

class VolcanicRockVoucherStrategy(SignalStrategy):
    def __init__(self, symbol: Symbol, limit: int, strike_price: int) -> None:
        super().__init__(symbol, limit)
        self.strike_price = strike_price
        self.volcanic_rock_symbol = Product.VOLCANIC_ROCK
        self.sigma = 0.2
        self.r = 0

    def fit_vol(self, S: float, px: float, T: float, initial_vol: float = 0.2, step: float = 0.00001) -> float:
        vol = initial_vol
        for _ in range(15):
            px_new = BS_CALL(S, self.strike_price, T, self.r, vol)
            if abs(px_new - px) < 0.05:
                break
            vega = (BS_CALL(S, self.strike_price, T, self.r, vol + step) - px_new) / step
            vol += (px - px_new) / vega if vega != 0 else 0
        return max(0.01, min(vol, 1.0))

    def calculate_delta(self, S: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.5
        d1 = (math.log(S / self.strike_price) + (self.r + sigma**2/2) * T) / (sigma * math.sqrt(T))
        return NormalDist().cdf(d1)

    def act(self, state: TradingState) -> None:
        if self.volcanic_rock_symbol not in state.order_depths or not state.order_depths[self.volcanic_rock_symbol].buy_orders or not state.order_depths[self.volcanic_rock_symbol].sell_orders:
            logger.print(f"{self.symbol}: No market data for VOLCANIC_ROCK at timestamp={state.timestamp}")
            return
        if self.symbol not in state.order_depths or not state.order_depths[self.symbol].buy_orders or not state.order_depths[self.symbol].sell_orders:
            logger.print(f"{self.symbol}: No market data at timestamp={state.timestamp}")
            return
        rock_price = self.get_mid_price(state, self.volcanic_rock_symbol)
        voucher_price = self.get_mid_price(state, self.symbol)
        days_to_expiry = max(7 - (state.timestamp // 1000), 1)
        T = days_to_expiry / 365
        try:
            new_sigma = self.fit_vol(rock_price, voucher_price, T, initial_vol=self.sigma)
            self.sigma = 0.7 * self.sigma + 0.3 * new_sigma
        except Exception as e:
            logger.print(f"{self.symbol}: Volatility fit failed, using default sigma=0.2, error={str(e)}")
            self.sigma = 0.2
        logger.print(f"{self.symbol}: Implied vol={self.sigma:.4f}, T={days_to_expiry} days")
        fair_value = BS_CALL(rock_price, self.strike_price, T, self.r, self.sigma)
        delta = self.calculate_delta(rock_price, T, self.sigma)
        position = state.position.get(self.symbol, 0)
        threshold = 1 * self.sigma if abs(rock_price - self.strike_price) < 500 else 2 * self.sigma
        logger.print(f"{self.symbol}: voucher_price={voucher_price:.2f}, fair_value={fair_value:.2f}, diff={voucher_price - fair_value:.2f}, threshold={threshold:.2f}")
        if voucher_price > fair_value + threshold and days_to_expiry > 2:
            self.go_short(state)
            hedge_qty = int(abs(position) * delta)
            if hedge_qty > 0 and state.position.get(self.volcanic_rock_symbol, 0) + hedge_qty <= 400:
                self.orders.append(Order(self.volcanic_rock_symbol, min(state.order_depths[self.volcanic_rock_symbol].sell_orders.keys()), hedge_qty))
                logger.print(f"{self.symbol}: Short, hedge_qty={hedge_qty}")
        elif voucher_price < fair_value - threshold:
            self.go_long(state)
            hedge_qty = int(abs(position) * delta)
            if hedge_qty > 0 and state.position.get(self.volcanic_rock_symbol, 0) - hedge_qty >= -400:
                self.orders.append(Order(self.volcanic_rock_symbol, max(state.order_depths[self.volcanic_rock_symbol].buy_orders.keys()), -hedge_qty))
                logger.print(f"{self.symbol}: Long, hedge_qty={hedge_qty}")

    def save(self) -> JSON:
        return {"strike_price": self.strike_price, "sigma": self.sigma}

    def load(self, data: JSON) -> None:
        self.strike_price = data.get("strike_price", self.strike_price)
        self.sigma = data.get("sigma", self.sigma)

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
            symbol: clazz(symbol, self.LIMIT[symbol], *args) for symbol, clazz, args in [
                (Product.VOLCANIC_ROCK, VolcanicRockStrategy, []),
                (Product.VOLCANIC_ROCK_VOUCHER_9500, VolcanicRockVoucherStrategy, [9500]),
                (Product.VOLCANIC_ROCK_VOUCHER_9750, VolcanicRockVoucherStrategy, [9750]),
                (Product.VOLCANIC_ROCK_VOUCHER_10000, VolcanicRockVoucherStrategy, [10000]),
                (Product.VOLCANIC_ROCK_VOUCHER_10250, VolcanicRockVoucherStrategy, [10250]),
                (Product.VOLCANIC_ROCK_VOUCHER_10500, VolcanicRockVoucherStrategy, [10500]),
            ]
        }
        self.params = PARAMS
        self.kelp_prices = []
        self.kelp_vwap = []
        self.kelp_last_price = None

    def get_swmid(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        CROISSANTS_PER_BASKET = BASKET_WEIGHTS[Product.CROISSANTS]
        JAMS_PER_BASKET = BASKET_WEIGHTS[Product.JAMS]
        DJEMBES_PER_BASKET = BASKET_WEIGHTS[Product.DJEMBES]
        synthetic_order_price = OrderDepth()
        croissants_best_bid = max(order_depths[Product.CROISSANTS].buy_orders.keys(), default=0)
        croissants_best_ask = min(order_depths[Product.CROISSANTS].sell_orders.keys(), default=float("inf"))
        jams_best_bid = max(order_depths[Product.JAMS].buy_orders.keys(), default=0)
        jams_best_ask = min(order_depths[Product.JAMS].sell_orders.keys(), default=float("inf"))
        djembes_best_bid = max(order_depths[Product.DJEMBES].buy_orders.keys(), default=0)
        djembes_best_ask = min(order_depths[Product.DJEMBES].sell_orders.keys(), default=float("inf"))
        implied_bid = croissants_best_bid * CROISSANTS_PER_BASKET + jams_best_bid * JAMS_PER_BASKET + djembes_best_bid * DJEMBES_PER_BASKET
        implied_ask = croissants_best_ask * CROISSANTS_PER_BASKET + jams_best_ask * JAMS_PER_BASKET + djembes_best_ask * DJEMBES_PER_BASKET
        if implied_bid > 0:
            croissants_bid_volume = order_depths[Product.CROISSANTS].buy_orders.get(croissants_best_bid, 0) // CROISSANTS_PER_BASKET
            jams_bid_volume = order_depths[Product.JAMS].buy_orders.get(jams_best_bid, 0) // JAMS_PER_BASKET
            djembes_bid_volume = order_depths[Product.DJEMBES].buy_orders.get(djembes_best_bid, 0) // DJEMBES_PER_BASKET
            implied_bid_volume = min(croissants_bid_volume, jams_bid_volume, djembes_bid_volume)
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume
        if implied_ask < float("inf"):
            croissants_ask_volume = -order_depths[Product.CROISSANTS].sell_orders.get(croissants_best_ask, 0) // CROISSANTS_PER_BASKET
            jams_ask_volume = -order_depths[Product.JAMS].sell_orders.get(jams_best_ask, 0) // JAMS_PER_BASKET
            djembes_ask_volume = -order_depths[Product.DJEMBES].sell_orders.get(djembes_best_ask, 0) // DJEMBES_PER_BASKET
            implied_ask_volume = min(croissants_ask_volume, jams_ask_volume, djembes_ask_volume)
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume
        return synthetic_order_price

    def convert_synthetic_basket_orders(self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        component_orders = {Product.CROISSANTS: [], Product.JAMS: [], Product.DJEMBES: []}
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        best_bid = max(synthetic_basket_order_depth.buy_orders.keys(), default=0)
        best_ask = min(synthetic_basket_order_depth.sell_orders.keys(), default=float("inf"))
        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity
            if quantity > 0 and price >= best_ask:
                croissants_price = min(order_depths[Product.CROISSANTS].sell_orders.keys(), default=best_ask)
                jams_price = min(order_depths[Product.JAMS].sell_orders.keys(), default=best_ask)
                djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys(), default=best_ask)
            elif quantity < 0 and price <= best_bid:
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys(), default=best_bid)
                jams_price = max(order_depths[Product.JAMS].buy_orders.keys(), default=best_bid)
                djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys(), default=best_bid)
            else:
                continue
            croissants_order = Order(Product.CROISSANTS, croissants_price, quantity * BASKET_WEIGHTS[Product.CROISSANTS])
            jams_order = Order(Product.JAMS, jams_price, quantity * BASKET_WEIGHTS[Product.JAMS])
            djembes_order = Order(Product.DJEMBES, djembes_price, quantity * BASKET_WEIGHTS[Product.DJEMBES])
            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.JAMS].append(jams_order)
            component_orders[Product.DJEMBES].append(djembes_order)
        return component_orders

    def execute_spread_orders(self, target_position: int, basket_position: int, order_depths: Dict[str, OrderDepth]):
        if target_position == basket_position:
            return None
        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            basket_orders = [Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)]
            aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders
        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            basket_orders = [Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)]
            aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders

    def spread_orders(self, order_depths: Dict[str, OrderDepth], product: Product, basket_position: int, spread_data: Dict[str, Any]):
        if Product.PICNIC_BASKET1 not in order_depths:
            return None
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = float(basket_swmid - synthetic_swmid)
        if "spread_history" not in spread_data:
            spread_data["spread_history"] = []
        max_history = self.params[Product.SPREAD]["spread_std_window"] + 5
        if len(spread_data["spread_history"]) >= max_history:
            spread_data["spread_history"] = spread_data["spread_history"][-max_history:]
        spread_data["spread_history"].append(float(spread))
        if len(spread_data["spread_history"]) < self.params[Product.SPREAD]["spread_std_window"]:
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD]["spread_std_window"]:
            spread_data["spread_history"].pop(0)
        try:
            spread_std = float(np.std(spread_data["spread_history"]))
        except:
            spread_std = self.params[Product.SPREAD]["default_spread_std"]
        zscore = float((spread - self.params[Product.SPREAD]["default_spread_mean"]) / spread_std)
        if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(-self.params[Product.SPREAD]["target_position"], basket_position, order_depths)
        if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(self.params[Product.SPREAD]["target_position"], basket_position, order_depths)
        spread_data["prev_zscore"] = float(zscore)
        return None

    def get_synthetic_basket2_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        CROISSANTS_PER_BASKET = BASKET2_WEIGHTS[Product.CROISSANTS]
        JAMS_PER_BASKET = BASKET2_WEIGHTS[Product.JAMS]
        synthetic_order_price = OrderDepth()
        croissants_best_bid = max(order_depths[Product.CROISSANTS].buy_orders.keys(), default=0)
        croissants_best_ask = min(order_depths[Product.CROISSANTS].sell_orders.keys(), default=float("inf"))
        jams_best_bid = max(order_depths[Product.JAMS].buy_orders.keys(), default=0)
        jams_best_ask = min(order_depths[Product.JAMS].sell_orders.keys(), default=float("inf"))
        implied_bid = croissants_best_bid * CROISSANTS_PER_BASKET + jams_best_bid * JAMS_PER_BASKET
        implied_ask = croissants_best_ask * CROISSANTS_PER_BASKET + jams_best_ask * JAMS_PER_BASKET
        if implied_bid > 0:
            croissants_bid_volume = order_depths[Product.CROISSANTS].buy_orders.get(croissants_best_bid, 0) // CROISSANTS_PER_BASKET
            jams_bid_volume = order_depths[Product.JAMS].buy_orders.get(jams_best_bid, 0) // JAMS_PER_BASKET
            implied_bid_volume = min(croissants_bid_volume, jams_bid_volume)
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume
        if implied_ask < float("inf"):
            croissants_ask_volume = -order_depths[Product.CROISSANTS].sell_orders.get(croissants_best_ask, 0) // CROISSANTS_PER_BASKET
            jams_ask_volume = -order_depths[Product.JAMS].sell_orders.get(jams_best_ask, 0) // JAMS_PER_BASKET
            implied_ask_volume = min(croissants_ask_volume, jams_ask_volume)
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume
        return synthetic_order_price

    def convert_synthetic_basket2_orders(self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        component_orders = {Product.CROISSANTS: [], Product.JAMS: []}
        synthetic_basket_order_depth = self.get_synthetic_basket2_order_depth(order_depths)
        best_bid = max(synthetic_basket_order_depth.buy_orders.keys(), default=0)
        best_ask = min(synthetic_basket_order_depth.sell_orders.keys(), default=float("inf"))
        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity
            if quantity > 0 and price >= best_ask:
                croissants_price = min(order_depths[Product.CROISSANTS].sell_orders.keys(), default=best_ask)
                jams_price = min(order_depths[Product.JAMS].sell_orders.keys(), default=best_ask)
            elif quantity < 0 and price <= best_bid:
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys(), default=best_bid)
                jams_price = max(order_depths[Product.JAMS].buy_orders.keys(), default=best_bid)
            else:
                continue
            croissants_order = Order(Product.CROISSANTS, croissants_price, quantity * BASKET2_WEIGHTS[Product.CROISSANTS])
            jams_order = Order(Product.JAMS, jams_price, quantity * BASKET2_WEIGHTS[Product.JAMS])
            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.JAMS].append(jams_order)
        return component_orders

    def execute_spread2_orders(self, target_position: int, basket_position: int, order_depths: Dict[str, OrderDepth]):
        if target_position == basket_position:
            return None
        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)
        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            basket_orders = [Order(Product.PICNIC_BASKET2, basket_ask_price, execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC2, synthetic_bid_price, -execute_volume)]
            aggregate_orders = self.convert_synthetic_basket2_orders(synthetic_orders, order_depths)
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
            return aggregate_orders
        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            basket_orders = [Order(Product.PICNIC_BASKET2, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC2, synthetic_ask_price, execute_volume)]
            aggregate_orders = self.convert_synthetic_basket2_orders(synthetic_orders, order_depths)
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
            return aggregate_orders

    def spread2_orders(self, order_depths: Dict[str, OrderDepth], product: Product, basket_position: int, spread_data: Dict[str, Any]):
        if Product.PICNIC_BASKET2 not in order_depths:
            return None
        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = float(basket_swmid - synthetic_swmid)
        if "spread_history" not in spread_data:
            spread_data["spread_history"] = []
        max_history = self.params[Product.SPREAD2]["spread_std_window"] + 5
        if len(spread_data["spread_history"]) >= max_history:
            spread_data["spread_history"] = spread_data["spread_history"][-max_history:]
        spread_data["spread_history"].append(float(spread))
        if len(spread_data["spread_history"]) < self.params[Product.SPREAD2]["spread_std_window"]:
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD2]["spread_std_window"]:
            spread_data["spread_history"].pop(0)
        try:
            spread_std = float(np.std(spread_data["spread_history"]))
        except:
            spread_std = self.params[Product.SPREAD2]["default_spread_std"]
        zscore = float((spread - self.params[Product.SPREAD2]["default_spread_mean"]) / spread_std)
        if zscore >= self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(-self.params[Product.SPREAD2]["target_position"], basket_position, order_depths)
        if zscore <= -self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(self.params[Product.SPREAD2]["target_position"], basket_position, order_depths)
        spread_data["prev_zscore"] = float(zscore)
        return None

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        traderObject = {}
        if state.traderData and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
                if "kelp_prices" in traderObject:
                    self.kelp_prices = traderObject["kelp_prices"]
                if "kelp_vwap" in traderObject:
                    self.kelp_vwap = traderObject["kelp_vwap"]
                if "kelp_last_price" in traderObject:
                    self.kelp_last_price = traderObject["kelp_last_price"]
            except:
                traderObject = {}

        result = {}
        conversions = 0
        new_trader_data = {}

        # Volcanic Rock and Vouchers
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])
            if symbol in state.order_depths:
                strategy_orders, strategy_conversions = strategy.run(state)
                result[symbol] = strategy_orders
                conversions += strategy_conversions
            new_trader_data[symbol] = strategy.save()

        # Picnic Basket 1 Spread
        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                "spread_history": [],
                "prev_zscore": 0.0,
                "clear_flag": False,
                "curr_avg": 0.0,
            }
        max_history = self.params[Product.SPREAD]["spread_std_window"] + 5
        if "spread_history" in traderObject[Product.SPREAD] and len(traderObject[Product.SPREAD]["spread_history"]) > max_history:
            traderObject[Product.SPREAD]["spread_history"] = traderObject[Product.SPREAD]["spread_history"][-max_history:]
        basket_position = state.position.get(Product.PICNIC_BASKET1, 0)
        spread_orders = self.spread_orders(state.order_depths, Product.PICNIC_BASKET1, basket_position, traderObject[Product.SPREAD])
        if spread_orders:
            result[Product.CROISSANTS] = spread_orders.get(Product.CROISSANTS, [])
            result[Product.JAMS] = spread_orders.get(Product.JAMS, [])
            result[Product.DJEMBES] = spread_orders.get(Product.DJEMBES, [])
            result[Product.PICNIC_BASKET1] = spread_orders.get(Product.PICNIC_BASKET1, [])

        # Picnic Basket 2 Spread
        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
                "spread_history": [],
                "prev_zscore": 0.0,
                "clear_flag": False,
                "curr_avg": 0.0,
            }
        max_history2 = self.params[Product.SPREAD2]["spread_std_window"] + 5
        if "spread_history" in traderObject[Product.SPREAD2] and len(traderObject[Product.SPREAD2]["spread_history"]) > max_history2:
            traderObject[Product.SPREAD2]["spread_history"] = traderObject[Product.SPREAD2]["spread_history"][-max_history2:]
        basket2_position = state.position.get(Product.PICNIC_BASKET2, 0)
        spread2_orders = self.spread2_orders(state.order_depths, Product.PICNIC_BASKET2, basket2_position, traderObject[Product.SPREAD2])
        if spread2_orders:
            if Product.CROISSANTS in result:
                result[Product.CROISSANTS].extend(spread2_orders.get(Product.CROISSANTS, []))
            else:
                result[Product.CROISSANTS] = spread2_orders.get(Product.CROISSANTS, [])
            if Product.JAMS in result:
                result[Product.JAMS].extend(spread2_orders.get(Product.JAMS, []))
            else:
                result[Product.JAMS] = spread2_orders.get(Product.JAMS, [])
            result[Product.PICNIC_BASKET2] = spread2_orders.get(Product.PICNIC_BASKET2, [])

        # Rainforest Resin
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths and self.params[Product.RAINFOREST_RESIN]["main_switch"]:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_orders = []
            buy_order_volume = 0
            sell_order_volume = 0
            fair_value = self.params[Product.RAINFOREST_RESIN]["fair_value"]
            order_depth = state.order_depths[Product.RAINFOREST_RESIN]
            baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1], default=fair_value + 1)
            bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1], default=fair_value - 1)
            if len(order_depth.sell_orders) != 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_amount = -order_depth.sell_orders[best_ask]
                if best_ask < fair_value:
                    quantity = min(best_ask_amount, self.LIMIT[Product.RAINFOREST_RESIN] - resin_position)
                    if quantity > 0:
                        resin_orders.append(Order(Product.RAINFOREST_RESIN, best_ask, quantity))
                        buy_order_volume += quantity
            if len(order_depth.buy_orders) != 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_amount = order_depth.buy_orders[best_bid]
                if best_bid > fair_value:
                    quantity = min(best_bid_amount, self.LIMIT[Product.RAINFOREST_RESIN] + resin_position)
                    if quantity > 0:
                        resin_orders.append(Order(Product.RAINFOREST_RESIN, best_bid, -quantity))
                        sell_order_volume += quantity
            buy_quantity = self.LIMIT[Product.RAINFOREST_RESIN] - (resin_position + buy_order_volume)
            if buy_quantity > 0:
                bid_price = round(bbbf + 1)
                resin_orders.append(Order(Product.RAINFOREST_RESIN, bid_price, buy_quantity))
            sell_quantity = self.LIMIT[Product.RAINFOREST_RESIN] + (resin_position - sell_order_volume)
            if sell_quantity > 0:
                ask_price = round(baaf - 1)
                resin_orders.append(Order(Product.RAINFOREST_RESIN, ask_price, -sell_quantity))
            result[Product.RAINFOREST_RESIN] = resin_orders

        # Kelp
        if Product.KELP in self.params and Product.KELP in state.order_depths and self.params[Product.KELP]["main_switch"]:
            kelp_position = state.position.get(Product.KELP, 0)
            order_depth = state.order_depths[Product.KELP]
            orders = []
            buy_order_volume = 0
            sell_order_volume = 0
            if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
                filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
                mm_ask = min(filtered_ask, default=best_ask)
                mm_bid = max(filtered_bid, default=best_bid)
                mmmid_price = (mm_ask + mm_bid) / 2
                self.kelp_prices.append(mmmid_price)
                total_bid_volume = sum(order_depth.buy_orders.values())
                weighted_bid_sum = sum(price * volume for price, volume in order_depth.buy_orders.items())
                weighted_bid_avg = weighted_bid_sum / total_bid_volume if total_bid_volume > 0 else best_bid
                total_ask_volume = sum(abs(volume) for volume in order_depth.sell_orders.values())
                weighted_ask_sum = sum(price * abs(volume) for price, volume in order_depth.sell_orders.items())
                weighted_ask_avg = weighted_ask_sum / total_ask_volume if total_ask_volume > 0 else best_ask
                fair_value = (weighted_bid_avg + weighted_ask_avg) / 2
                fair_value += self.params[Product.KELP]["upward_bias"]
                self.kelp_last_price = fair_value
                kelp_take_width = self.params[Product.KELP]["take_width"]
                if best_ask <= fair_value - kelp_take_width:
                    ask_amount = -order_depth.sell_orders[best_ask]
                    if ask_amount <= 50:
                        quantity = min(ask_amount, self.LIMIT[Product.KELP] - kelp_position)
                        if quantity > 0:
                            orders.append(Order(Product.KELP, best_ask, quantity))
                            buy_order_volume += quantity
                if best_bid >= fair_value + kelp_take_width:
                    bid_amount = order_depth.buy_orders[best_bid]
                    if bid_amount <= 50:
                        quantity = min(bid_amount, self.LIMIT[Product.KELP] + kelp_position)
                        if quantity > 0:
                            orders.append(Order(Product.KELP, best_bid, -quantity))
                            sell_order_volume += quantity
                aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
                bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
                baaf = min(aaf, default=fair_value + 1)
                bbbf = max(bbf, default=fair_value - 1)
                buy_quantity = self.LIMIT[Product.KELP] - (kelp_position + buy_order_volume)
                if buy_quantity > 0:
                    bid_price = round(bbbf + 1)
                    orders.append(Order(Product.KELP, bid_price, buy_quantity))
                sell_quantity = self.LIMIT[Product.KELP] + (kelp_position - sell_order_volume)
                if sell_quantity > 0:
                    ask_price = round(baaf - 1)
                    orders.append(Order(Product.KELP, ask_price, -sell_quantity))
                if len(self.kelp_prices) > 50:
                    self.kelp_prices.pop(0)
                result[Product.KELP] = orders

        # Save trader data
        compact_trader_object = {
            "kelp_last_price": float(self.kelp_last_price) if self.kelp_last_price is not None else None,
            "kelp_prices": [float(p) for p in self.kelp_prices[-20:]] if self.kelp_prices else [],
        }
        for symbol in [Product.SPREAD, Product.SPREAD2]:
            compact_trader_object[symbol] = traderObject.get(symbol, {})
            if "spread_history" in compact_trader_object[symbol]:
                history_length = min(self.params[symbol]["spread_std_window"] + 5, len(compact_trader_object[symbol]["spread_history"]))
                compact_trader_object[symbol]["spread_history"] = [float(x) for x in compact_trader_object[symbol]["spread_history"][-history_length:]]
            for key in ["prev_zscore", "curr_avg"]:
                if key in compact_trader_object[symbol]:
                    compact_trader_object[symbol][key] = float(compact_trader_object[symbol][key])
        compact_trader_object.update(new_trader_data)
        try:
            traderData = jsonpickle.encode(compact_trader_object)
        except:
            traderData = "{}"
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData