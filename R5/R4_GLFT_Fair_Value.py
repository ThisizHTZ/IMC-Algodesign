import json
import math
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, TradingState
from statistics import NormalDist
from typing import Any, Dict, List, Tuple
import jsonpickle
import numpy as np
from numpy.linalg import inv

JSON = Dict[str, Any] | List[Any] | str | int | float | bool | None

def BS_CALL(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    N = NormalDist().cdf
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * N(d1) - K * math.exp(-r * T) * N(d2)

class Logger:
    """Compact logger to fit within message size limits."""
    def __init__(self) -> None:
        self.logs: str = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base = self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions, "", ""
        ])
        max_item = (self.max_log_length - len(base)) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item),
            self.truncate(self.logs, max_item),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        return [
            state.timestamp,
            trader_data,
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
            {s: [od.buy_orders, od.sell_orders] for s, od in state.order_depths.items()},
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
             for arr in state.own_trades.values() for t in arr],
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
             for arr in state.market_trades.values() for t in arr],
            state.position,
            [
                state.observations.plainValueObservations,
                {
                    p: [
                        o.bidPrice,
                        o.askPrice,
                        o.transportFees,
                        o.exportTariff,
                        o.importTariff,
                        o.sugarPrice,
                        o.sunlightIndex,
                    ]
                    for p, o in state.observations.conversionObservations.items()
                }
            ]
        ]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, v: Any) -> str:
        return json.dumps(v, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, v: str, n: int) -> str:
        return v if len(v) <= n else v[:n - 3] + "..."

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
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

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
        "fair_value": 10000,
    },
    Product.KELP: {
        "main_switch": True,
        "upward_bias": 0.3,
    },
    Product.MAGNIFICENT_MACARONS: {
        "csi_alpha": 0.1,
        "persistence": 5,
        "take_width": 1,
        "margin": 2,
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
    """Base for a single-symbol strategy."""
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: List[Order] = []

    @abstractmethod
    def act(self, state: TradingState) -> None:
        ...

    def run(self, state: TradingState) -> Tuple[List[Order], int]:
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

class SignalStrategy(Strategy):
    """Mean-reversion on mid-price threshold."""
    def get_mid(self, state: TradingState, sym: str) -> float:
        od = state.order_depths[sym]
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        return (best_bid + best_ask) / 2

    def go_long(self, state: TradingState) -> None:
        od = state.order_depths[self.symbol]
        price = min(od.sell_orders)
        pos = state.position.get(self.symbol, 0)
        self.buy(price, self.limit - pos)

    def go_short(self, state: TradingState) -> None:
        od = state.order_depths[self.symbol]
        price = max(od.buy_orders)
        pos = state.position.get(self.symbol, 0)
        self.sell(price, self.limit + pos)

class VolcanicRockStrategy(SignalStrategy):
    def __init__(self, symbol: str, limit: int) -> None:
        super().__init__(symbol, limit)
        self.threshold: float | None = None
        self.hist: deque[float] = deque(maxlen=20)

    def act(self, state: TradingState) -> None:
        od = state.order_depths.get(self.symbol)
        if not od or not od.buy_orders or not od.sell_orders:
            return
        mid = self.get_mid(state, self.symbol)
        self.hist.append(mid)
        if self.threshold is None:
            self.threshold = mid
        if len(self.hist) == self.hist.maxlen:
            self.threshold = sum(self.hist) / len(self.hist)
        if mid > self.threshold * 1.01:
            self.go_long(state)
        elif mid < self.threshold * 0.99:
            self.go_short(state)

    def save(self) -> JSON:
        return {"threshold": self.threshold, "hist": list(self.hist)}

    def load(self, data: JSON) -> None:
        self.threshold = data.get("threshold", self.threshold)
        self.hist = deque(data.get("hist", []), maxlen=20)

class VolcanicRockVoucherStrategy(SignalStrategy):
    def __init__(self, symbol: str, limit: int, strike_price: int) -> None:
        super().__init__(symbol, limit)
        self.K = strike_price
        self.rock_sym = Product.VOLCANIC_ROCK
        self.sigma = 0.2
        self.r = 0.0

    def fit_vol(self, S: float, px: float, T: float) -> float:
        vol = self.sigma
        for _ in range(15):
            price0 = BS_CALL(S, self.K, T, self.r, vol)
            if abs(price0 - px) < 0.05:
                break
            vega = (BS_CALL(S, self.K, T, self.r, vol + 1e-5) - price0) / 1e-5
            if vega != 0:
                vol += (px - price0) / vega
        return max(0.01, min(vol, 1.0))

    def calc_delta(self, S: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.5
        d1 = (math.log(S / self.K) + (self.r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
        return NormalDist().cdf(d1)

    def act(self, state: TradingState) -> None:
        od_rock = state.order_depths.get(self.rock_sym)
        od_voucher = state.order_depths.get(self.symbol)
        if not od_rock or not od_rock.buy_orders or not od_rock.sell_orders:
            return
        if not od_voucher or not od_voucher.buy_orders or not od_voucher.sell_orders:
            return
        S = self.get_mid(state, self.rock_sym)
        V = self.get_mid(state, self.symbol)
        days = max(7 - (state.timestamp // 1000), 1)
        T = days / 365
        try:
            new_sigma = self.fit_vol(S, V, T)
            self.sigma = 0.7 * self.sigma + 0.3 * new_sigma
        except:
            self.sigma = 0.2
        fair = BS_CALL(S, self.K, T, self.r, self.sigma)
        delta = self.calc_delta(S, T, self.sigma)
        pos = state.position.get(self.symbol, 0)
        thresh = (1 if abs(S - self.K) < 500 else 2) * self.sigma
        if V > fair + thresh and days > 2:
            self.go_short(state)
            hedge_qty = int(abs(pos) * delta)
            if hedge_qty > 0 and state.position.get(self.rock_sym, 0) + hedge_qty <= 400:
                p = min(od_rock.sell_orders)
                self.orders.append(Order(self.rock_sym, p, hedge_qty))
        elif V < fair - thresh:
            self.go_long(state)
            hedge_qty = int(abs(pos) * delta)
            if hedge_qty > 0 and state.position.get(self.rock_sym, 0) - hedge_qty >= -400:
                p = max(od_rock.buy_orders)
                self.orders.append(Order(self.rock_sym, p, -hedge_qty))

    def save(self) -> JSON:
        return {"K": self.K, "sigma": self.sigma}

    def load(self, data: JSON) -> None:
        self.K = data.get("K", self.K)
        self.sigma = data.get("sigma", self.sigma)

def mm_glft(
    fair: float,
    best_bid: float,
    best_ask: float,
    q: float,
    gamma: float,
    sigma: float,
    amount: int,
) -> Tuple[int, int]:
    """Avellaneda-Stoikov fair quote offsets."""
    k_b = 1 / max((fair - best_bid) - 1, 1)
    k_a = 1 / max((best_ask - fair) - 1, 1)
    A_b = A_a = 0.25
    delta_b = (
        1 / gamma * math.log(1 + gamma / k_b)
        + ((2 * q + 1) / 2)
        * math.sqrt((sigma**2 * gamma) / (2 * k_b * A_b) * (1 + gamma / k_b) ** (1 + k_b / gamma))
    )
    delta_a = (
        1 / gamma * math.log(1 + gamma / k_a)
        - ((2 * q - 1) / 2)
        * math.sqrt((sigma**2 * gamma) / (2 * k_a * A_a) * (1 + gamma / k_a) ** (1 + k_a / gamma))
    )
    p_b = int(round(fair - delta_b))
    p_a = int(round(fair + delta_a))
    p_b = min(p_b, best_bid + 1)
    p_a = max(p_a, best_ask - 1)
    return p_b, p_a

def predict_resin_fair(hist: Dict[str, List[float]]) -> float:
    """Linear regression on rainforest resin factors."""
    X = np.c_[hist['sugar'], hist['tf'], hist['et'], hist['it']]
    y = np.array(hist['mid'])
    Xb = np.c_[np.ones(len(y)), X]
    theta = inv(Xb.T @ Xb) @ (Xb.T @ y)
    latest = np.array([1, hist['sugar'][-1], hist['tf'][-1], hist['et'][-1], hist['it'][-1]])
    return float(latest @ theta)

class MacaronStrategy(Strategy):
    def __init__(self, symbol: str, limit: int) -> None:
        super().__init__(symbol, limit)
        p = PARAMS[Product.MAGNIFICENT_MACARONS]
        self.ewma: float | None = None
        self.alpha = p['csi_alpha']
        self.low_count = 0
        self.persist = p['persistence']
        self.take = p['take_width']
        self.margin = p['margin']

    def act(self, state: TradingState) -> None:
        od = state.order_depths.get(self.symbol)
        obs = state.observations.conversionObservations.get(self.symbol)
        if not od or not od.buy_orders or not od.sell_orders or not obs:
            return
        sun = obs.sunlightIndex
        if self.ewma is None:
            self.ewma = sun
        else:
            self.ewma = self.alpha * sun + (1 - self.alpha) * self.ewma
        std = abs(sun - self.ewma)
        CSI = self.ewma - std
        self.low_count = self.low_count + 1 if sun < CSI else 0

        cost = obs.sugarPrice + obs.transportFees + obs.exportTariff + obs.importTariff
        fair = cost + self.margin
        best_ask = min(od.sell_orders)
        best_bid = max(od.buy_orders)
        pos = state.position.get(self.symbol, 0)

        # Panic regime
        if self.low_count >= self.persist:
            qty = self.limit - pos
            if qty > 0:
                self.buy(best_ask, qty)
            return

        # Normal Avellaneda–Stoikov market making
        p_b, p_a = mm_glft(fair, best_bid, best_ask, pos, 1e-9, 0.2, 20)
        buy_amt = min(20, self.limit - pos)
        sell_amt = min(20, self.limit + pos)

        if buy_amt > 0 and best_ask <= fair - self.take:
            self.buy(best_ask, buy_amt)
        if sell_amt > 0 and best_bid >= fair + self.take:
            self.sell(best_bid, sell_amt)

        if buy_amt > 0:
            self.buy(p_b, buy_amt)
        if sell_amt > 0:
            self.sell(p_a, sell_amt)

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
            Product.MAGNIFICENT_MACARONS: 75,
        }
        self.strategies: Dict[str, Strategy] = {
            Product.VOLCANIC_ROCK: VolcanicRockStrategy(Product.VOLCANIC_ROCK, 400),
            Product.VOLCANIC_ROCK_VOUCHER_9500: VolcanicRockVoucherStrategy(Product.VOLCANIC_ROCK_VOUCHER_9500, 200, 9500),
            Product.VOLCANIC_ROCK_VOUCHER_9750: VolcanicRockVoucherStrategy(Product.VOLCANIC_ROCK_VOUCHER_9750, 200, 9750),
            Product.VOLCANIC_ROCK_VOUCHER_10000: VolcanicRockVoucherStrategy(Product.VOLCANIC_ROCK_VOUCHER_10000, 200, 10000),
            Product.VOLCANIC_ROCK_VOUCHER_10250: VolcanicRockVoucherStrategy(Product.VOLCANIC_ROCK_VOUCHER_10250, 200, 10250),
            Product.VOLCANIC_ROCK_VOUCHER_10500: VolcanicRockVoucherStrategy(Product.VOLCANIC_ROCK_VOUCHER_10500, 200, 10500),
            Product.MAGNIFICENT_MACARONS: MacaronStrategy(Product.MAGNIFICENT_MACARONS, 75),
        }
        self.params = PARAMS
        self.kelp_prices: List[float] = []
        self.kelp_last_price: float | None = None
        self.resin_hist: Dict[str, List[float]] = {
            "sugar": [], "tf": [], "et": [], "it": [], "mid": []
        }

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # Run each registered strategy
        for sym, strat in self.strategies.items():
            if sym in state.order_depths:
                orders, conv = strat.run(state)
                if orders:
                    result[sym] = orders
                conversions += conv

        # Optimized Rainforest Resin
        if (Product.RAINFOREST_RESIN in state.order_depths
            and Product.RAINFOREST_RESIN in state.observations.conversionObservations
            and self.params[Product.RAINFOREST_RESIN]["main_switch"]):
            od = state.order_depths[Product.RAINFOREST_RESIN]
            obs = state.observations.conversionObservations[Product.RAINFOREST_RESIN]
            mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0

            # Update history
            h = self.resin_hist
            h["sugar"].append(obs.sugarPrice)
            h["tf"].append(obs.transportFees)
            h["et"].append(obs.exportTariff)
            h["it"].append(obs.importTariff)
            h["mid"].append(mid)
            for k in h:
                h[k] = h[k][-50:]

            if len(h["mid"]) > 10:
                fair = predict_resin_fair(h)
            else:
                fair = self.params[Product.RAINFOREST_RESIN]["fair_value"]

            # Simple take/clear around fair
            best_ask = min(od.sell_orders)
            best_bid = max(od.buy_orders)
            pos = state.position.get(Product.RAINFOREST_RESIN, 0)

            buy_amt = min(-od.sell_orders[best_ask], self.LIMIT[Product.RAINFOREST_RESIN] - pos)
            if buy_amt > 0 and best_ask < fair:
                result.setdefault(Product.RAINFOREST_RESIN, []).append(
                    Order(Product.RAINFOREST_RESIN, best_ask, buy_amt)
                )

            sell_amt = min(od.buy_orders[best_bid], self.LIMIT[Product.RAINFOREST_RESIN] + pos)
            if sell_amt > 0 and best_bid > fair:
                result.setdefault(Product.RAINFOREST_RESIN, []).append(
                    Order(Product.RAINFOREST_RESIN, best_bid, -sell_amt)
                )

        # Optimized Kelp market making
        if Product.KELP in state.order_depths and self.params[Product.KELP]["main_switch"]:
            od = state.order_depths[Product.KELP]
            pos = state.position.get(Product.KELP, 0)
            bid_vwap = sum(p * v for p, v in od.buy_orders.items()) / sum(od.buy_orders.values())
            ask_vwap = sum(p * abs(v) for p, v in od.sell_orders.items()) / sum(abs(v) for v in od.sell_orders.values())
            fair = (bid_vwap + ask_vwap) / 2 + self.params[Product.KELP]["upward_bias"]
            mmmid = (min(od.sell_orders) + max(od.buy_orders)) / 2
            self.kelp_prices.append(mmmid)
            if len(self.kelp_prices) > 50:
                self.kelp_prices.pop(0)
            sigma = np.std(self.kelp_prices[-20:]) if len(self.kelp_prices) >= 20 else 0.2

            p_b, p_a = mm_glft(
                fair,
                max(od.buy_orders),
                min(od.sell_orders),
                pos,
                1e-9,
                sigma,
                20
            )
            buy_amt = min(20, self.LIMIT[Product.KELP] - pos)
            sell_amt = min(20, self.LIMIT[Product.KELP] + pos)

            if buy_amt > 0:
                result.setdefault(Product.KELP, []).append(Order(Product.KELP, p_b, buy_amt))
            if sell_amt > 0:
                result.setdefault(Product.KELP, []).append(Order(Product.KELP, p_a, -sell_amt))

        # Log and return
        trader_data = jsonpickle.encode({})
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
