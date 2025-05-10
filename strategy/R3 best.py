import json
from typing import Any, Dict, List
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation, Symbol, Listing, Observation, ProsperityEncoder, Trade
import jsonpickle
import numpy as np
from math import log, sqrt, exp
from statistics import NormalDist
import uuid

class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"

PARAMS = {
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.14,
        "volatility_std":  0.005066171529344382,
        "z_score_threshold": 1,
        "z_score_close_threshold": 0.3,
        "strike": 10000,
        "starting_time_to_expiry": 6/250,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.1,
        "volatility_std": 0.007377301666201331,
        "z_score_threshold": 1,
        "z_score_close_threshold": 0.3,
        "strike": 10250,
        "starting_time_to_expiry": 6/250,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.15,
        "volatility_std":  0.00737730166620133,
        "z_score_threshold": 1,
        "z_score_close_threshold": 0.2,
        "strike": 10500,
        "starting_time_to_expiry": 6/250,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.01,
        "volatility_std": 0.012722957843708135,
        "z_score_threshold": 1,
        "z_score_close_threshold": 0.51,
        "strike": 9500,
        "starting_time_to_expiry": 6/250,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.15,
        "volatility_std": 0.015989031023267774,
        "z_score_threshold": 0.5,
        "z_score_close_threshold": 0.1,
        "strike": 9750,
        "starting_time_to_expiry": 6/250,
    }
}

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        time_to_expiry = max(0.001, time_to_expiry)
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        time_to_expiry = max(0.001, time_to_expiry)
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        return NormalDist().cdf(d1)

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        time_to_expiry = max(0.001, time_to_expiry)
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

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
        return value[: max_length - 3] + "..."

logger = Logger()

class VolatilitySurfaceModel:
    def __init__(self):
        self.smile_params = {
            'a': 0.284823,
            'b': 0.001092,
            'c': 0.122348
        }
        self.deviation_threshold = 0.01

    def calculate_theoretical_volatility(self, moneyness):
        a, b, c = self.smile_params['a'], self.smile_params['b'], self.smile_params['c']
        return a * moneyness**2 + b * moneyness + c

    def detect_mispricing(self, moneyness, actual_volatility):
        theoretical_vol = self.calculate_theoretical_volatility(moneyness)
        deviation = actual_volatility - theoretical_vol
        deviation_percent = deviation / theoretical_vol
        if abs(deviation_percent) > self.deviation_threshold:
            action = "BUY" if deviation_percent < 0 else "SELL"
            signal_strength = abs(deviation_percent) / self.deviation_threshold
            return {
                "action": action,
                "deviation": deviation,
                "deviation_percent": deviation_percent,
                "theoretical_vol": theoretical_vol,
                "signal_strength": signal_strength
            }
        return None

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
        }
        self.vol_surface_model = VolatilitySurfaceModel()

    def get_volcanic_rock_voucher_mid_price(
            self,
            voucher_order_depth: OrderDepth,
            traderData: Dict[str, Any],
            product: str
    ):
        logger.print(f"Calculating mid price for {product}")
        if len(voucher_order_depth.buy_orders) > 0 and len(voucher_order_depth.sell_orders) > 0:
            best_bid = max(voucher_order_depth.buy_orders.keys())
            best_ask = min(voucher_order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            traderData[f"prev_{product}_price"] = mid_price
            logger.print(f"{product} - Best bid: {best_bid}, Best ask: {best_ask}, Mid price: {mid_price}")
            return mid_price
        else:
            prev_price = traderData.get(f"prev_{product}_price", 0)
            logger.print(f"{product} - No valid orders, using previous price: {prev_price}")
            return prev_price

    def delta_hedge_volcanic_rock_position(
        self,
        volcanic_rock_order_depth: OrderDepth,
        voucher_position: int,
        volcanic_rock_position: int,
        volcanic_rock_buy_orders: int,
        volcanic_rock_sell_orders: int,
        delta: float,
        traderData: Dict[str, Any],
        product: str
    ) -> List[Order]:
        if traderData["hedged"]:
            logger.print(f"{product} - Already hedged, skipping delta hedge")
            return None

        target_volcanic_rock_position = -int(delta * voucher_position)
        hedge_quantity = target_volcanic_rock_position - (volcanic_rock_position + volcanic_rock_buy_orders - volcanic_rock_sell_orders)
        logger.print(f"{product} - Delta hedge: Target position: {target_volcanic_rock_position}, Hedge quantity: {hedge_quantity}")

        orders: List[Order] = []
        if hedge_quantity > 0:
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(abs(hedge_quantity), -volcanic_rock_order_depth.sell_orders[best_ask])
            quantity = min(quantity, self.LIMIT[Product.VOLCANIC_ROCK] - (volcanic_rock_position + volcanic_rock_buy_orders))
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity))
                logger.print(f"{product} - Delta hedge buy order: Price: {best_ask}, Quantity: {quantity}")
        elif hedge_quantity < 0:
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(abs(hedge_quantity), volcanic_rock_order_depth.buy_orders[best_bid])
            quantity = min(quantity, self.LIMIT[Product.VOLCANIC_ROCK] + (volcanic_rock_position - volcanic_rock_sell_orders))
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))
                logger.print(f"{product} - Delta hedge sell order: Price: {best_bid}, Quantity: {-quantity}")
        return orders

    def delta_hedge_volcanic_rock_voucher_orders(
            self,
            volcanic_rock_order_depth: OrderDepth,
            voucher_orders: List[Order],
            volcanic_rock_position: int,
            volcanic_rock_buy_orders: int,
            volcanic_rock_sell_orders: int,
            delta: float,
            traderData: Dict[str, Any],
            product: str
    ) -> List[Order]:
        if traderData["hedged"]:
            logger.print(f"{product} - Already hedged, skipping voucher order hedge")
            return None

        if len(voucher_orders) == 0:
            logger.print(f"{product} - No voucher orders to hedge")
            return None

        net_voucher_quantity = sum(order.quantity for order in voucher_orders)
        target_volcanic_rock_quantity = -int(delta * net_voucher_quantity)
        logger.print(f"{product} - Voucher order hedge: Net voucher quantity: {net_voucher_quantity}, Target volcanic rock: {target_volcanic_rock_quantity}")

        orders: List[Order] = []
        if target_volcanic_rock_quantity > 0:
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(abs(target_volcanic_rock_quantity), -volcanic_rock_order_depth.sell_orders[best_ask])
            quantity = min(quantity, self.LIMIT[Product.VOLCANIC_ROCK] - (volcanic_rock_position + volcanic_rock_buy_orders))
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity))
                logger.print(f"{product} - Voucher hedge buy order: Price: {best_ask}, Quantity: {quantity}")
        elif target_volcanic_rock_quantity < 0:
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(abs(target_volcanic_rock_quantity), volcanic_rock_order_depth.buy_orders[best_bid])
            quantity = min(quantity, self.LIMIT[Product.VOLCANIC_ROCK] + (volcanic_rock_position - volcanic_rock_sell_orders))
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))
                logger.print(f"{product} - Voucher hedge sell order: Price: {best_bid}, Quantity: {-quantity}")
        return orders

    def volcanic_rock_voucher_orders(
            self,
            voucher_order_depth: OrderDepth,
            voucher_position: int,
            traderData: Dict[str, Any],
            volatility: float,
            product: str,
            tte: float
    ) -> List[Order]:
        logger.print(f"{product} - Evaluating voucher orders: Position: {voucher_position}, Volatility: {volatility}, TTE: {tte}")
        if tte < 1/250:
            logger.print(f"{product} - TTE < 1/250, attempting to clear position")
            if voucher_position > 0 and len(voucher_order_depth.buy_orders) > 0:
                best_bid = max(voucher_order_depth.buy_orders.keys())
                quantity = min(voucher_position, voucher_order_depth.buy_orders[best_bid])
                traderData["hedged"] = True
                logger.print(f"{product} - Clear position sell order: Price: {best_bid}, Quantity: {-quantity}")
                return [Order(product, best_bid, -quantity)]
            elif voucher_position < 0 and len(voucher_order_depth.sell_orders) > 0:
                best_ask = min(voucher_order_depth.sell_orders.keys())
                quantity = min(abs(voucher_position), -voucher_order_depth.sell_orders[best_ask])
                traderData["hedged"] = True
                logger.print(f"{product} - Clear position buy order: Price: {best_ask}, Quantity: {quantity}")
                return [Order(product, best_ask, quantity)]
            logger.print(f"{product} - No valid orders to clear position")
            return None

        # Calculate z-score
        mean_vol = self.params[product]["mean_volatility"]
        vol_std = self.params[product]["volatility_std"]
        z_score = (volatility - mean_vol) / vol_std if vol_std > 0 else 0
        z_score_threshold = self.params[product]["z_score_threshold"]
        z_score_close_threshold = self.params[product]["z_score_close_threshold"]
        logger.print(f"{product} - Z-score: {z_score}, Z-score threshold: {z_score_threshold}, Z-score close threshold: {z_score_close_threshold}")

        if z_score > z_score_threshold:
            logger.print(f"{product} - Z-score high: {z_score} > {z_score_threshold}")
            if voucher_position != -self.LIMIT[product]:
                target_voucher_position = -self.LIMIT[product]
                if len(voucher_order_depth.buy_orders) > 0:
                    best_bid = max(voucher_order_depth.buy_orders.keys())
                    quantity = min(abs(target_voucher_position - voucher_position), abs(voucher_order_depth.buy_orders[best_bid]))
                    traderData["hedged"] = False
                    logger.print(f"{product} - Sell order due to high z-score: Price: {best_bid}, Quantity: {-quantity}")
                    return [Order(product, best_bid, -quantity)]
        elif z_score < -z_score_threshold:
            logger.print(f"{product} - Z-score low: {z_score} < {-z_score_threshold}")
            if voucher_position != self.LIMIT[product]:
                target_voucher_position = self.LIMIT[product]
                if len(voucher_order_depth.sell_orders) > 0:
                    best_ask = min(voucher_order_depth.sell_orders.keys())
                    quantity = min(abs(target_voucher_position - voucher_position), abs(voucher_order_depth.sell_orders[best_ask]))
                    traderData["hedged"] = False
                    logger.print(f"{product} - Buy order due to low z-score: Price: {best_ask}, Quantity: {quantity}")
                    return [Order(product, best_ask, quantity)]
        elif abs(z_score) <= z_score_close_threshold and voucher_position != 0:
            logger.print(f"{product} - Z-score close to zero, closing position")
            if voucher_position > 0 and len(voucher_order_depth.buy_orders) > 0:
                best_bid = max(voucher_order_depth.buy_orders.keys())
                quantity = min(voucher_position, voucher_order_depth.buy_orders[best_bid])
                traderData["hedged"] = True
                logger.print(f"{product} - Close position sell order: Price: {best_bid}, Quantity: {-quantity}")
                return [Order(product, best_bid, -quantity)]
            elif voucher_position < 0 and len(voucher_order_depth.sell_orders) > 0:
                best_ask = min(voucher_order_depth.sell_orders.keys())
                quantity = min(abs(voucher_position), -voucher_order_depth.sell_orders[best_ask])
                traderData["hedged"] = True
                logger.print(f"{product} - Close position buy order: Price: {best_ask}, Quantity: {quantity}")
                return [Order(product, best_ask, quantity)]
        logger.print(f"{product} - No trading signal generated")
        return None

    def calculate_moneyness(self, strike, spot_price, time_to_expiry):
        moneyness = math.log(strike/spot_price) / math.sqrt(time_to_expiry)
        logger.print(f"Calculating moneyness: Strike: {strike}, Spot: {spot_price}, TTE: {time_to_expiry}, Moneyness: {moneyness}")
        return moneyness

    def get_option_signals_from_volatility_surface(self, product, order_depth, spot_price, tte, position):
        logger.print(f"{product} - Evaluating volatility surface signals: Spot: {spot_price}, TTE: {tte}, Position: {position}")
        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            logger.print(f"{product} - No valid orders for volatility surface analysis")
            return []

        option_mid_price = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2
        logger.print(f"{product} - Option mid price: {option_mid_price}")

        strike = self.params[product]["strike"]
        try:
            volatility = BlackScholes.implied_volatility(option_mid_price, spot_price, strike, tte)
            logger.print(f"{product} - Implied volatility: {volatility}")

            moneyness = self.calculate_moneyness(strike, spot_price, tte)
            mispricing = self.vol_surface_model.detect_mispricing(moneyness, volatility)
            if mispricing:
                logger.print(f"{product} - Mispricing detected: {mispricing}")
                action = mispricing["action"]
                signal_strength = mispricing["signal_strength"]
                position_limit = self.LIMIT[product]
                quantity = int(min(position_limit / 2, position_limit * signal_strength / 3))

                if action == "BUY" and position < position_limit:
                    best_ask = min(order_depth.sell_orders.keys())
                    available_quantity = min(quantity, abs(order_depth.sell_orders[best_ask]))
                    if available_quantity > 0:
                        logger.print(f"{product} - Volatility surface buy order: Price: {best_ask}, Quantity: {available_quantity}")
                        return [Order(product, best_ask, available_quantity)]
                elif action == "SELL" and position > -position_limit:
                    best_bid = max(order_depth.buy_orders.keys())
                    available_quantity = min(quantity, order_depth.buy_orders[best_bid])
                    if available_quantity > 0:
                        logger.print(f"{product} - Volatility surface sell order: Price: {best_bid}, Quantity: {-available_quantity}")
                        return [Order(product, best_bid, -available_quantity)]
        except Exception as e:
            logger.print(f"{product} - Volatility surface calculation error: {str(e)}")
        logger.print(f"{product} - No volatility surface trading signal")
        return []

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        logger.print(f"Starting trading cycle at timestamp: {state.timestamp}")
        try:
            traderObject = {}
            if state.traderData and state.traderData != "":
                traderObject = jsonpickle.decode(state.traderData)
                logger.print("Loaded trader data:", traderObject)

            result = {}
            conversions = 0

            option_products = [
                Product.VOLCANIC_ROCK_VOUCHER_9500,
                Product.VOLCANIC_ROCK_VOUCHER_9750,
                Product.VOLCANIC_ROCK_VOUCHER_10000,
                Product.VOLCANIC_ROCK_VOUCHER_10250,
                Product.VOLCANIC_ROCK_VOUCHER_10500
            ]

            if Product.VOLCANIC_ROCK in state.order_depths:
                volcanic_rock_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
                if len(volcanic_rock_order_depth.buy_orders) > 0 and len(volcanic_rock_order_depth.sell_orders) > 0:
                    volcanic_rock_mid_price = (max(volcanic_rock_order_depth.buy_orders.keys()) +
                                            min(volcanic_rock_order_depth.sell_orders.keys())) / 2
                    logger.print(f"VOLCANIC_ROCK - Mid price: {volcanic_rock_mid_price}")

                    tte = max(0.001, self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]['starting_time_to_expiry'] -
                            (state.timestamp) / 1000000 / 7)
                    logger.print(f"Time to expiry: {tte}")

                    total_delta_exposure = 0
                    for product in option_products:
                        if product in state.order_depths:
                            position = state.position.get(product, 0)
                            logger.print(f"{product} - Current position: {position}")

                            volatility_surface_orders = self.get_option_signals_from_volatility_surface(
                                product,
                                state.order_depths[product],
                                volcanic_rock_mid_price,
                                tte,
                                position
                            )
                            if volatility_surface_orders:
                                result[product] = volatility_surface_orders
                                logger.print(f"{product} - Volatility surface orders: {volatility_surface_orders}")
                            else:
                                if product not in traderObject:
                                    traderObject[product] = {
                                        "hedged": True,
                                        "prev_voucher_price": 0
                                    }
                                voucher_order_depth = state.order_depths[product]
                                voucher_mid_price = self.get_volcanic_rock_voucher_mid_price(
                                    voucher_order_depth,
                                    traderObject[product],
                                    product
                                )
                                try:
                                    volatility = BlackScholes.implied_volatility(
                                        voucher_mid_price,
                                        volcanic_rock_mid_price,
                                        self.params[product]["strike"],
                                        tte
                                    )
                                    delta = BlackScholes.delta(
                                        volcanic_rock_mid_price,
                                        self.params[product]["strike"],
                                        tte,
                                        volatility
                                    )
                                    logger.print(f"{product} - Volatility: {volatility}, Delta: {delta}")

                                    voucher_orders = self.volcanic_rock_voucher_orders(
                                        state.order_depths[product],
                                        position,
                                        traderObject[product],
                                        volatility,
                                        product,
                                        tte
                                    )
                                    if voucher_orders:
                                        result[product] = voucher_orders
                                        logger.print(f"{product} - Traditional orders: {voucher_orders}")

                                    if product in state.position:
                                        total_delta_exposure += state.position[product] * delta
                                        logger.print(f"{product} - Delta exposure contribution: {state.position[product] * delta}")
                                except Exception as e:
                                    logger.print(f"{product} - Calculation error: {str(e)}")

                    target_volcanic_position = -int(round(total_delta_exposure))
                    current_volcanic_position = state.position.get(Product.VOLCANIC_ROCK, 0)
                    logger.print(f"VOLCANIC_ROCK - Target position: {target_volcanic_position}, Current: {current_volcanic_position}")

                    if target_volcanic_position != current_volcanic_position:
                        hedge_quantity = target_volcanic_position - current_volcanic_position
                        logger.print(f"VOLCANIC_ROCK - Hedge quantity: {hedge_quantity}")
                        volcanic_rock_orders = []
                        if hedge_quantity > 0:
                            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
                            quantity = min(abs(hedge_quantity), -volcanic_rock_order_depth.sell_orders[best_ask])
                            quantity = min(quantity, self.LIMIT[Product.VOLCANIC_ROCK] - current_volcanic_position)
                            if quantity > 0:
                                volcanic_rock_orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity))
                                logger.print(f"VOLCANIC_ROCK - Delta hedge buy order: Price: {best_ask}, Quantity: {quantity}")
                        else:
                            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
                            quantity = min(abs(hedge_quantity), volcanic_rock_order_depth.buy_orders[best_bid])
                            quantity = min(quantity, self.LIMIT[Product.VOLCANIC_ROCK] + current_volcanic_position)
                            if quantity > 0:
                                volcanic_rock_orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))
                                logger.print(f"VOLCANIC_ROCK - Delta hedge sell order: Price: {best_bid}, Quantity: {-quantity}")
                        if volcanic_rock_orders:
                            result[Product.VOLCANIC_ROCK] = volcanic_rock_orders

            traderData = jsonpickle.encode(traderObject)
            logger.print(f"Final result: {result}")
            logger.flush(state, result, conversions, traderData)
            return result, conversions, traderData
        except Exception as e:
            logger.print(f"Runtime error: {str(e)}")
            result = {}
            conversions = 0
            traderData = jsonpickle.encode(traderObject) if 'traderObject' in locals() else ""
            logger.flush(state, result, conversions, traderData)
            return result, conversions, traderData