from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import json
from typing import Any, Dict, List
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[list[Order]], conversions: int, trader_data: str) -> None:
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

    def compress_orders(self, orders: dict[list[Order]]) -> list[list[Any]]:
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
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SYNTHETIC2 = "SYNTHETIC2"
    SPREAD = "SPREAD"
    SPREAD2 = "SPREAD2"
    RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"


PARAMS = {
    Product.SPREAD: {
        "default_spread_mean": 96.69596635887267,
        "default_spread_std": 79.24916501711826,
        "spread_std_window": 42,
        "zscore_threshold": 21.68,
        "target_position": 60,
    },
    Product.SPREAD2: {
        "default_spread_mean": 35.08325361256638,
        "default_spread_std": 54.19968093790806,
        "spread_std_window": 51,
        "zscore_threshold": 30,
        "target_position": 100,
    },
    Product.RESIN: {
        "main_switch": True,
        # strategy params
        "take": True,
        "clear": True,
        "make": True,
        # assume fixed fair value
        "fair_value": 10000,
        # order params
        "take_width": 1,
        "clear_width": 0,
        # market making params
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.KELP: {
        "main_switch": True,
        # strategy params
        "take": True,
        "clear": True,
        "make": True,
        # fair value params
        "fval_model": "mean_rev",  # mean_rev, arma, ema, none
        "upward_bias": 0.3,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "ema_alpha": 0.2,
        # order params
        "take_width": 1,
        "clear_width": 0,
        # market making params
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
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


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.RESIN: 50,
            Product.KELP: 50,
        }

        # 初始化KELP相关的状态变量
        self.kelp_prices = []
        self.kelp_vwap = []
        self.kelp_last_price = None

    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
                best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket_order_depth(
            self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CROISSANTS_PER_BASKET = BASKET_WEIGHTS[Product.CROISSANTS]
        JAMS_PER_BASKET = BASKET_WEIGHTS[Product.JAMS]
        DJEMBES_PER_BASKET = BASKET_WEIGHTS[Product.DJEMBES]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        croissants_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        croissants_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        jams_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        jams_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        djembes_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        djembes_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
                croissants_best_bid * CROISSANTS_PER_BASKET
                + jams_best_bid * JAMS_PER_BASKET
                + djembes_best_bid * DJEMBES_PER_BASKET
        )
        implied_ask = (
                croissants_best_ask * CROISSANTS_PER_BASKET
                + jams_best_ask * JAMS_PER_BASKET
                + djembes_best_ask * DJEMBES_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            croissants_bid_volume = (
                    order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid]
                    // CROISSANTS_PER_BASKET
            )
            jams_bid_volume = (
                    order_depths[Product.JAMS].buy_orders[jams_best_bid]
                    // JAMS_PER_BASKET
            )
            djembes_bid_volume = (
                    order_depths[Product.DJEMBES].buy_orders[djembes_best_bid]
                    // DJEMBES_PER_BASKET
            )
            implied_bid_volume = min(
                croissants_bid_volume, jams_bid_volume, djembes_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            croissants_ask_volume = (
                    -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask]
                    // CROISSANTS_PER_BASKET
            )
            jams_ask_volume = (
                    -order_depths[Product.JAMS].sell_orders[jams_best_ask]
                    // JAMS_PER_BASKET
            )
            djembes_ask_volume = (
                    -order_depths[Product.DJEMBES].sell_orders[djembes_best_ask]
                    // DJEMBES_PER_BASKET
            )
            implied_ask_volume = min(
                croissants_ask_volume, jams_ask_volume, djembes_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket_orders(
            self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissants_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                jams_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
                djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            croissants_order = Order(
                Product.CROISSANTS,
                croissants_price,
                quantity * BASKET_WEIGHTS[Product.CROISSANTS],
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * BASKET_WEIGHTS[Product.JAMS],
            )
            djembes_order = Order(
                Product.DJEMBES, djembes_price, quantity * BASKET_WEIGHTS[Product.DJEMBES]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.JAMS].append(jams_order)
            component_orders[Product.DJEMBES].append(djembes_order)

        return component_orders

    def execute_spread_orders(
            self,
            target_position: int,
            basket_position: int,
            order_depths: Dict[str, OrderDepth],
    ):
        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders

    def spread_orders(
            self,
            order_depths: Dict[str, OrderDepth],
            product: Product,
            basket_position: int,
            spread_data: Dict[str, Any],
    ):
        if Product.PICNIC_BASKET1 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = float(basket_swmid - synthetic_swmid)

        # 确保spread_history是列表
        if "spread_history" not in spread_data:
            spread_data["spread_history"] = []

        # 限制历史数据长度
        max_history = self.params[Product.SPREAD]["spread_std_window"] + 5
        if len(spread_data["spread_history"]) >= max_history:
            spread_data["spread_history"] = spread_data["spread_history"][-max_history:]

        # 添加当前价差
        spread_data["spread_history"].append(float(spread))

        if (
                len(spread_data["spread_history"])
                < self.params[Product.SPREAD]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = float(np.std(spread_data["spread_history"]))

        zscore = float((
                               spread - self.params[Product.SPREAD]["default_spread_mean"]
                       ) / spread_std)

        if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(
                    -self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(
                    self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = float(zscore)
        return None

    def get_synthetic_basket2_order_depth(
            self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CROISSANTS_PER_BASKET = BASKET2_WEIGHTS[Product.CROISSANTS]
        JAMS_PER_BASKET = BASKET2_WEIGHTS[Product.JAMS]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        croissants_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        croissants_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        jams_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        jams_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
                croissants_best_bid * CROISSANTS_PER_BASKET
                + jams_best_bid * JAMS_PER_BASKET
        )
        implied_ask = (
                croissants_best_ask * CROISSANTS_PER_BASKET
                + jams_best_ask * JAMS_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            croissants_bid_volume = (
                    order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid]
                    // CROISSANTS_PER_BASKET
            )
            jams_bid_volume = (
                    order_depths[Product.JAMS].buy_orders[jams_best_bid]
                    // JAMS_PER_BASKET
            )
            implied_bid_volume = min(
                croissants_bid_volume, jams_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            croissants_ask_volume = (
                    -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask]
                    // CROISSANTS_PER_BASKET
            )
            jams_ask_volume = (
                    -order_depths[Product.JAMS].sell_orders[jams_best_ask]
                    // JAMS_PER_BASKET
            )
            implied_ask_volume = min(
                croissants_ask_volume, jams_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket2_orders(
            self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket2_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissants_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                jams_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            croissants_order = Order(
                Product.CROISSANTS,
                croissants_price,
                quantity * BASKET2_WEIGHTS[Product.CROISSANTS],
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * BASKET2_WEIGHTS[Product.JAMS],
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.JAMS].append(jams_order)

        return component_orders

    def execute_spread2_orders(
            self,
            target_position: int,
            basket_position: int,
            order_depths: Dict[str, OrderDepth],
    ):
        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET2, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC2, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket2_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET2, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC2, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket2_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
            return aggregate_orders

    def spread2_orders(
            self,
            order_depths: Dict[str, OrderDepth],
            product: Product,
            basket_position: int,
            spread_data: Dict[str, Any],
    ):
        if Product.PICNIC_BASKET2 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = float(basket_swmid - synthetic_swmid)

        # 确保spread_history是列表
        if "spread_history" not in spread_data:
            spread_data["spread_history"] = []

        # 限制历史数据长度
        max_history = self.params[Product.SPREAD2]["spread_std_window"] + 5
        if len(spread_data["spread_history"]) >= max_history:
            spread_data["spread_history"] = spread_data["spread_history"][-max_history:]

        # 添加当前价差
        spread_data["spread_history"].append(float(spread))

        if (
                len(spread_data["spread_history"])
                < self.params[Product.SPREAD2]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD2]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = float(np.std(spread_data["spread_history"]))

        zscore = float((
                               spread - self.params[Product.SPREAD2]["default_spread_mean"]
                       ) / spread_std)

        if zscore >= self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(
                    -self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(
                    self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = float(zscore)
        return None

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
                # 获取KELP的状态数据
                if "kelp_prices" in traderObject:
                    self.kelp_prices = traderObject["kelp_prices"]
                if "kelp_vwap" in traderObject:
                    self.kelp_vwap = traderObject["kelp_vwap"]
                if "kelp_last_price" in traderObject:
                    self.kelp_last_price = traderObject["kelp_last_price"]
            except:
                # 如果解析失败，使用空字典
                traderObject = {}

        result = {}
        conversions = 0

        # 篮子1 ETF 相关的交易逻辑
        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                "spread_history": [],
                "prev_zscore": 0.0,
                "clear_flag": False,
                "curr_avg": 0.0,
            }

        # 限制历史数据长度
        max_history = self.params[Product.SPREAD]["spread_std_window"] + 5  # 稍微多留一些空间
        if "spread_history" in traderObject[Product.SPREAD] and len(
                traderObject[Product.SPREAD]["spread_history"]) > max_history:
            traderObject[Product.SPREAD]["spread_history"] = traderObject[Product.SPREAD]["spread_history"][
                                                             -max_history:]

        basket_position = (
            state.position[Product.PICNIC_BASKET1]
            if Product.PICNIC_BASKET1 in state.position
            else 0
        )

        spread_orders = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET1,
            basket_position,
            traderObject[Product.SPREAD],
        )
        if spread_orders != None:
            result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
            result[Product.JAMS] = spread_orders[Product.JAMS]
            result[Product.DJEMBES] = spread_orders[Product.DJEMBES]
            result[Product.PICNIC_BASKET1] = spread_orders[Product.PICNIC_BASKET1]

        # 篮子2 ETF 相关的交易逻辑
        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
                "spread_history": [],
                "prev_zscore": 0.0,
                "clear_flag": False,
                "curr_avg": 0.0,
            }

        # 限制历史数据长度
        max_history2 = self.params[Product.SPREAD2]["spread_std_window"] + 5  # 稍微多留一些空间
        if "spread_history" in traderObject[Product.SPREAD2] and len(
                traderObject[Product.SPREAD2]["spread_history"]) > max_history2:
            traderObject[Product.SPREAD2]["spread_history"] = traderObject[Product.SPREAD2]["spread_history"][
                                                              -max_history2:]

        basket2_position = (
            state.position[Product.PICNIC_BASKET2]
            if Product.PICNIC_BASKET2 in state.position
            else 0
        )

        spread2_orders = self.spread2_orders(
            state.order_depths,
            Product.PICNIC_BASKET2,
            basket2_position,
            traderObject[Product.SPREAD2],
        )
        if spread2_orders != None:
            # 合并订单结果
            if Product.CROISSANTS in result and Product.CROISSANTS in spread2_orders:
                result[Product.CROISSANTS].extend(spread2_orders[Product.CROISSANTS])
            elif Product.CROISSANTS in spread2_orders:
                result[Product.CROISSANTS] = spread2_orders[Product.CROISSANTS]

            if Product.JAMS in result and Product.JAMS in spread2_orders:
                result[Product.JAMS].extend(spread2_orders[Product.JAMS])
            elif Product.JAMS in spread2_orders:
                result[Product.JAMS] = spread2_orders[Product.JAMS]

            result[Product.PICNIC_BASKET2] = spread2_orders[Product.PICNIC_BASKET2]

        # RAINFOREST_RESIN 交易逻辑
        if Product.RESIN in self.params and Product.RESIN in state.order_depths and self.params[Product.RESIN][
            "main_switch"]:
            resin_position = state.position[Product.RESIN] if Product.RESIN in state.position else 0

            resin_orders = []
            buy_order_volume = 0
            sell_order_volume = 0

            fair_value = self.params[Product.RESIN]["fair_value"]

            # 提取订单深度信息
            order_depth = state.order_depths[Product.RESIN]

            baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1],
                       default=fair_value + 1)
            bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1],
                       default=fair_value - 1)

            # 吃单逻辑 - 买入
            if len(order_depth.sell_orders) != 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_amount = -1 * order_depth.sell_orders[best_ask]
                if best_ask < fair_value:
                    quantity = min(best_ask_amount, self.LIMIT[Product.RESIN] - resin_position)
                    if quantity > 0:
                        resin_orders.append(Order(Product.RESIN, best_ask, quantity))
                        buy_order_volume += quantity

            # 吃单逻辑 - 卖出
            if len(order_depth.buy_orders) != 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_amount = order_depth.buy_orders[best_bid]
                if best_bid > fair_value:
                    quantity = min(best_bid_amount, self.LIMIT[Product.RESIN] + resin_position)
                    if quantity > 0:
                        resin_orders.append(Order(Product.RESIN, best_bid, -1 * quantity))
                        sell_order_volume += quantity

            # 挂单逻辑 - 买入
            buy_quantity = self.LIMIT[Product.RESIN] - (resin_position + buy_order_volume)
            if buy_quantity > 0:
                bid_price = round(bbbf + 1)
                resin_orders.append(Order(Product.RESIN, bid_price, buy_quantity))

            # 挂单逻辑 - 卖出
            sell_quantity = self.LIMIT[Product.RESIN] + (resin_position - sell_order_volume)
            if sell_quantity > 0:
                ask_price = round(baaf - 1)
                resin_orders.append(Order(Product.RESIN, ask_price, -sell_quantity))

            result[Product.RESIN] = resin_orders

        # KELP 交易逻辑
        if Product.KELP in self.params and Product.KELP in state.order_depths and self.params[Product.KELP][
            "main_switch"]:
            kelp_position = state.position[Product.KELP] if Product.KELP in state.position else 0

            order_depth = state.order_depths[Product.KELP]
            orders = []
            buy_order_volume = 0
            sell_order_volume = 0

            if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())

                # 过滤大订单以识别做市商报价
                filtered_ask = [price for price in order_depth.sell_orders.keys()
                                if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
                filtered_bid = [price for price in order_depth.buy_orders.keys()
                                if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]

                mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
                mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid

                # 计算做市商中间价并存储历史
                mmmid_price = (mm_ask + mm_bid) / 2
                self.kelp_prices.append(mmmid_price)

                # 计算加权平均价格作为公平价值
                total_bid_volume = sum(order_depth.buy_orders.values())
                weighted_bid_sum = sum(price * volume for price, volume in order_depth.buy_orders.items())
                weighted_bid_avg = weighted_bid_sum / total_bid_volume if total_bid_volume > 0 else best_bid

                total_ask_volume = sum(abs(volume) for volume in order_depth.sell_orders.values())
                weighted_ask_sum = sum(price * abs(volume) for price, volume in order_depth.sell_orders.items())
                weighted_ask_avg = weighted_ask_sum / total_ask_volume if total_ask_volume > 0 else best_ask

                fair_value = (weighted_bid_avg + weighted_ask_avg) / 2
                fair_value += self.params[Product.KELP]["upward_bias"]  # 上行偏差

                self.kelp_last_price = fair_value

                # 买入逻辑
                kelp_take_width = self.params[Product.KELP]["take_width"]
                if best_ask <= fair_value - kelp_take_width:
                    ask_amount = -1 * order_depth.sell_orders[best_ask]
                    if ask_amount <= 50:  # A避免大订单
                        quantity = min(ask_amount, self.LIMIT[Product.KELP] - kelp_position)
                        if quantity > 0:
                            orders.append(Order(Product.KELP, best_ask, quantity))
                            buy_order_volume += quantity

                # 卖出逻辑
                if best_bid >= fair_value + kelp_take_width:
                    bid_amount = order_depth.buy_orders[best_bid]
                    if bid_amount <= 50:  # 避免大订单
                        quantity = min(bid_amount, self.LIMIT[Product.KELP] + kelp_position)
                        if quantity > 0:
                            orders.append(Order(Product.KELP, best_bid, -1 * quantity))
                            sell_order_volume += quantity

                # 计算挂单价格
                aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
                bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
                baaf = min(aaf) if len(aaf) > 0 else fair_value + 1
                bbbf = max(bbf) if len(bbf) > 0 else fair_value - 1

                # 挂买单
                buy_quantity = self.LIMIT[Product.KELP] - (kelp_position + buy_order_volume)
                if buy_quantity > 0:
                    bid_price = round(bbbf + 1)
                    orders.append(Order(Product.KELP, bid_price, buy_quantity))

                # 挂卖单
                sell_quantity = self.LIMIT[Product.KELP] + (kelp_position - sell_order_volume)
                if sell_quantity > 0:
                    ask_price = round(baaf - 1)
                    orders.append(Order(Product.KELP, ask_price, -sell_quantity))

                # 保存价格历史数据，保持合理的长度
                if len(self.kelp_prices) > 50:
                    self.kelp_prices.pop(0)

            result[Product.KELP] = orders

        # 保存交易者数据 - 简化数据存储，只保留必要信息
        compact_trader_object = {
            "kelp_last_price": float(self.kelp_last_price) if self.kelp_last_price is not None else None
        }

        # 保留最近的kelp价格，但限制数量
        if len(self.kelp_prices) > 0:
            compact_trader_object["kelp_prices"] = [float(p) for p in self.kelp_prices[-20:]]  # 只保留最近20个价格
        else:
            compact_trader_object["kelp_prices"] = []

        # 只保留必要的spread数据
        if Product.SPREAD in traderObject:
            compact_trader_object[Product.SPREAD] = {}

            # 保留必要的数值
            for key in ["prev_zscore", "clear_flag", "curr_avg"]:
                if key in traderObject[Product.SPREAD]:
                    compact_trader_object[Product.SPREAD][key] = float(traderObject[Product.SPREAD][key]) if isinstance(
                        traderObject[Product.SPREAD][key], (int, float, np.number)) else traderObject[Product.SPREAD][
                        key]

            # 只保留必要的历史数据
            if "spread_history" in traderObject[Product.SPREAD]:
                history_length = min(self.params[Product.SPREAD]["spread_std_window"] + 5,
                                     len(traderObject[Product.SPREAD]["spread_history"]))
                compact_trader_object[Product.SPREAD]["spread_history"] = [float(x) for x in
                                                                           traderObject[Product.SPREAD][
                                                                               "spread_history"][-history_length:]]

        if Product.SPREAD2 in traderObject:
            compact_trader_object[Product.SPREAD2] = {}

            # 保留必要的数值
            for key in ["prev_zscore", "clear_flag", "curr_avg"]:
                if key in traderObject[Product.SPREAD2]:
                    compact_trader_object[Product.SPREAD2][key] = float(
                        traderObject[Product.SPREAD2][key]) if isinstance(traderObject[Product.SPREAD2][key],
                                                                          (int, float, np.number)) else \
                    traderObject[Product.SPREAD2][key]

            # 只保留必要的历史数据
            if "spread_history" in traderObject[Product.SPREAD2]:
                history_length = min(self.params[Product.SPREAD2]["spread_std_window"] + 5,
                                     len(traderObject[Product.SPREAD2]["spread_history"]))
                compact_trader_object[Product.SPREAD2]["spread_history"] = [float(x) for x in
                                                                            traderObject[Product.SPREAD2][
                                                                                "spread_history"][-history_length:]]

        # 使用更简单的序列化方式
        try:
            traderData = jsonpickle.encode(compact_trader_object)
        except:
            # 如果序列化失败，使用一个简单的空对象
            traderData = "{}"

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData