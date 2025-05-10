import math
from math import log, sqrt, exp
from statistics import NormalDist
from typing import Dict, List
from datamodel import Order, TradingState, OrderDepth


class Trader:  # 必须保持类名为Trader
    def __init__(self):
        self.voucher_strikes = {
            'VOLCANIC_ROCK_VOUCHER_9500': 9500,
            'VOLCANIC_ROCK_VOUCHER_9750': 9750,
            'VOLCANIC_ROCK_VOUCHER_10000': 10000,
            'VOLCANIC_ROCK_VOUCHER_10250': 10250,
            'VOLCANIC_ROCK_VOUCHER_10500': 10500
        }
        self.position_limits = 200
        self.vol_cache = {}
        self.iv_history = {k: [] for k in self.voucher_strikes}

    def black_scholes(self, S: float, K: float, T: float, sigma: float, r: float = 0.0,
                      call_put: str = 'call') -> float:
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        nd = NormalDist().cdf

        if call_put == 'call':
            return S * nd(d1) - K * exp(-r * T) * nd(d2)
        else:
            return K * exp(-r * T) * nd(-d2) - S * nd(-d1)

    def implied_vol(self, S: float, K: float, T: float, market_price: float) -> float:
        cache_key = (round(S, 2), round(K, 2), round(T, 4))
        if cache_key in self.vol_cache:
            return self.vol_cache[cache_key]

        vol_min, vol_max = 0.001, 5.0
        for _ in range(50):
            vol_mid = (vol_min + vol_max) / 2
            price = self.black_scholes(S, K, T, vol_mid)

            if abs(price - market_price) < 0.01:
                self.vol_cache[cache_key] = vol_mid
                return vol_mid
            elif price > market_price:
                vol_max = vol_mid
            else:
                vol_min = vol_mid

        result = (vol_min + vol_max) / 2
        self.vol_cache[cache_key] = result
        return result

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
        return (best_bid + best_ask) / 2 if best_ask != float('inf') else best_bid

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        # 获取标的资产价格
        S = self.get_mid_price(state.order_depths.get('VOLCANIC_ROCK', OrderDepth()))
        if S == 0:
            S = 10000  # 默认值

        # 计算剩余到期时间（假设1 round = 1天）
        days_to_expiry = max(7 - (state.timestamp // 1_000_000), 1)

        # 每个VOUCHER产品独立处理
        for product, strike in self.voucher_strikes.items():
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else strike * 0.95
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else strike * 1.05

                # 简化版网格搜索
                bid_price = int(best_bid * 0.995)
                ask_price = int(best_ask * 1.005)

                if bid_price < ask_price:
                    result[product] = [
                        Order(product, bid_price, self.position_limits),
                        Order(product, ask_price, -self.position_limits)
                    ]

        return result, 0, ""