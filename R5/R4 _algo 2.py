from datamodel import Listing, ConversionObservation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any, Tuple
import jsonpickle
import json

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

    def compress_observations(self, observations: ConversionObservation) -> list[Any]:
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
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            encoded_candidate = json.dumps(candidate)
            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out

logger = Logger()

class Product:
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

PARAMS = {
    Product.MAGNIFICENT_MACARONS: {
        "make_edge": 0.3,  # Reduced for competitive orders
        "make_min_edge": 0.1,  # Tighter spreads
        "make_probability": 0.9,  # High execution probability
        "init_make_edge": 0.3,  # Initial competitiveness
        "min_edge": 0.1,  # Tighter spreads
        "volume_avg_timestamp": 5,
        "volume_bar": 30,  # Trigger edge adjustments frequently
        "dec_edge_discount": 0.8,
        "step_size": 0.05,  # Fine edge control
        "csi_edge_multiplier_below": 1.01,  # Very tight for low sunlight
        "csi_edge_multiplier_above": 0.99,  # Slightly looser for high sunlight
    }
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {Product.MAGNIFICENT_MACARONS: 75}
        self.CONVERSION_LIMIT = 10
        self.CSI_MIN = 45.0  # Lower bound of CSI range
        self.CSI_MAX = 46.5  # Upper bound of CSI range

    def implied_bid_ask(self, observation: ConversionObservation) -> Tuple[float, float]:
        implied_bid = observation.bidPrice - observation.exportTariff - observation.transportFees
        implied_ask = observation.askPrice + observation.importTariff + observation.transportFees
        return implied_bid, implied_ask

    def adap_edge(
        self,
        timestamp: int,
        curr_edge: float,
        position: int,
        trader_object: Dict[str, Any],
        sunlight_index: float
    ) -> float:
        if timestamp == 0:
            trader_object[Product.MAGNIFICENT_MACARONS]["curr_edge"] = self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"]
            return self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"]

        trader_object[Product.MAGNIFICENT_MACARONS]["volume_history"].append(abs(position))
        if len(trader_object[Product.MAGNIFICENT_MACARONS]["volume_history"]) > self.params[Product.MAGNIFICENT_MACARONS]["volume_avg_timestamp"]:
            trader_object[Product.MAGNIFICENT_MACARONS]["volume_history"].pop(0)

        if len(trader_object[Product.MAGNIFICENT_MACARONS]["volume_history"]) < self.params[Product.MAGNIFICENT_MACARONS]["volume_avg_timestamp"]:
            return curr_edge

        # Replaced np.mean with standard Python mean
        volume_avg = sum(trader_object[Product.MAGNIFICENT_MACARONS]["volume_history"]) / len(trader_object[Product.MAGNIFICENT_MACARONS]["volume_history"])
        if sunlight_index < self.CSI_MIN:
            csi_adjustment = self.params[Product.MAGNIFICENT_MACARONS]["csi_edge_multiplier_below"]
        elif sunlight_index > self.CSI_MAX:
            csi_adjustment = self.params[Product.MAGNIFICENT_MACARONS]["csi_edge_multiplier_above"]
        else:
            csi_adjustment = 1.0  # Neutral within CSI range
        adjusted_edge = curr_edge * csi_adjustment

        if volume_avg >= self.params[Product.MAGNIFICENT_MACARONS]["volume_bar"]:
            trader_object[Product.MAGNIFICENT_MACARONS]["volume_history"] = []
            trader_object[Product.MAGNIFICENT_MACARONS]["curr_edge"] = adjusted_edge + self.params[Product.MAGNIFICENT_MACARONS]["step_size"]
            return adjusted_edge + self.params[Product.MAGNIFICENT_MACARONS]["step_size"]

        elif self.params[Product.MAGNIFICENT_MACARONS]["dec_edge_discount"] * self.params[Product.MAGNIFICENT_MACARONS]["volume_bar"] * (adjusted_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"]) > volume_avg * adjusted_edge:
            if adjusted_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"] > self.params[Product.MAGNIFICENT_MACARONS]["min_edge"]:
                trader_object[Product.MAGNIFICENT_MACARONS]["volume_history"] = []
                trader_object[Product.MAGNIFICENT_MACARONS]["curr_edge"] = adjusted_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"]
                trader_object[Product.MAGNIFICENT_MACARONS]["optimized"] = True
                return adjusted_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"]
            else:
                trader_object[Product.MAGNIFICENT_MACARONS]["curr_edge"] = self.params[Product.MAGNIFICENT_MACARONS]["min_edge"]
                return self.params[Product.MAGNIFICENT_MACARONS]["min_edge"]

        trader_object[Product.MAGNIFICENT_MACARONS]["curr_edge"] = adjusted_edge
        return adjusted_edge

    def arb_clear(self, position: int, sunlight_index: float) -> int:
        """Clear position aggressively."""
        if position == 0:
            return 0
        if sunlight_index < self.CSI_MIN:
            if position > 0:
                conversions = max(-self.CONVERSION_LIMIT, -position * 4 // 5)  # Clear 80% of long
            else:
                conversions = max(-self.CONVERSION_LIMIT, min(self.CONVERSION_LIMIT, -position * 3 // 4))  # Clear 75% of short
        elif sunlight_index > self.CSI_MAX:
            conversions = max(-self.CONVERSION_LIMIT, min(self.CONVERSION_LIMIT, -position * 2 // 3))  # Clear 66% of position
        else:
            conversions = max(-self.CONVERSION_LIMIT, min(self.CONVERSION_LIMIT, -position * 3 // 4))  # Clear 75% in neutral range
        logger.print(f"arb_clear: Position={position}, SunlightIndex={sunlight_index}, Conversions={conversions}")
        return conversions

    def arb_take(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        adap_edge: float,
        position: int,
        sunlight_index: float
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]
        buy_order_volume = 0
        sell_order_volume = 0

        implied_bid, implied_ask = self.implied_bid_ask(observation)
        best_bid = max(order_depth.buy_orders.keys(), default=implied_bid - 1) if order_depth.buy_orders else implied_bid - 1
        best_ask = min(order_depth.sell_orders.keys(), default=implied_ask + 1) if order_depth.sell_orders else implied_ask + 1
        spread = best_ask - best_bid
        logger.print(f"arb_take: ImpliedBid={implied_bid}, ImpliedAsk={implied_ask}, AdapEdge={adap_edge}, Spread={spread}, BestBid={best_bid}, BestAsk={best_ask}")

        buy_quantity = position_limit - position
        sell_quantity = position_limit + position

        # Dynamic edge based on market spread
        edge_factor = min(0.2, spread * 0.2) if spread > 0 else 0.2
        if sunlight_index < self.CSI_MIN:
            buy_edge = edge_factor * 0.8
            sell_edge = edge_factor * 1.0
        elif sunlight_index > self.CSI_MAX:
            buy_edge = edge_factor * 0.9
            sell_edge = edge_factor * 0.9
        else:
            buy_edge = edge_factor  # Balanced in neutral range
            sell_edge = edge_factor

        # Take sell orders (buy)
        for price in sorted(list(order_depth.sell_orders.keys())):
            if price >= implied_bid - buy_edge:
                logger.print(f"arb_take: Skip Buy Price={price}, Above Threshold={implied_bid - buy_edge}")
                break
            quantity = min(abs(order_depth.sell_orders[price]), buy_quantity)
            if quantity > 0:
                orders.append(Order(Product.MAGNIFICENT_MACARONS, round(price), quantity))
                buy_order_volume += quantity
                logger.print(f"arb_take: Buy Order Price={price}, Quantity={quantity}, vs BestAsk={best_ask}")

        # Take buy orders (sell)
        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            if price <= implied_ask + sell_edge:
                logger.print(f"arb_take: Skip Sell Price={price}, Below Threshold={implied_ask + sell_edge}")
                break
            quantity = min(abs(order_depth.buy_orders[price]), sell_quantity)
            if quantity > 0:
                orders.append(Order(Product.MAGNIFICENT_MACARONS, round(price), -quantity))
                sell_order_volume += quantity
                logger.print(f"arb_take: Sell Order Price={price}, Quantity={quantity}, vs BestBid={best_bid}")

        # Fallback: Always place competitive orders
        if buy_quantity > 0:
            buy_price = min(best_ask, implied_bid - edge_factor)
            orders.append(Order(Product.MAGNIFICENT_MACARONS, round(buy_price), min(buy_quantity, 20)))
            buy_order_volume += min(buy_quantity, 20)
            logger.print(f"arb_take: Fallback Buy Order Price={buy_price}, Quantity={min(buy_quantity, 20)}, vs BestAsk={best_ask}")
        if sell_quantity > 0:
            sell_price = max(best_bid, implied_ask + edge_factor)
            orders.append(Order(Product.MAGNIFICENT_MACARONS, round(sell_price), -min(sell_quantity, 20)))
            sell_order_volume += min(sell_quantity, 20)
            logger.print(f"arb_take: Fallback Sell Order Price={sell_price}, Quantity={min(sell_quantity, 20)}, vs BestBid={best_bid}")

        logger.print(f"arb_take: SunlightIndex={sunlight_index}, BuyOrders={len([o for o in orders if o.quantity > 0])}, SellOrders={len([o for o in orders if o.quantity < 0])}")
        return orders, buy_order_volume, sell_order_volume

    def arb_make(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        position: int,
        edge: float,
        buy_order_volume: int,
        sell_order_volume: int,
        sunlight_index: float
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]

        implied_bid, implied_ask = self.implied_bid_ask(observation)
        best_bid = max(order_depth.buy_orders.keys(), default=implied_bid - 1) if order_depth.buy_orders else implied_bid - 1
        best_ask = min(order_depth.sell_orders.keys(), default=implied_ask + 1) if order_depth.sell_orders else implied_ask + 1
        spread = best_ask - best_bid
        logger.print(f"arb_make: ImpliedBid={implied_bid}, ImpliedAsk={implied_ask}, Edge={edge}, Spread={spread}, BestBid={best_bid}, BestAsk={best_ask}")

        # Dynamic edge based on market spread
        edge_factor = min(edge, spread * 0.2) if spread > 0 else edge
        if sunlight_index < self.CSI_MIN:
            bid = implied_bid - edge_factor * 1.0
            ask = implied_ask + edge_factor * 1.0
        elif sunlight_index > self.CSI_MAX:
            bid = implied_bid - edge_factor * 0.8
            ask = implied_ask + edge_factor * 0.8
        else:
            bid = implied_bid - edge_factor * 0.9  # Balanced spread
            ask = implied_ask + edge_factor * 0.9

        # Aggressive ask adjustment
        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 0.3
        if aggressive_ask > implied_ask:
            ask = aggressive_ask

        # Relaxed filtering
        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 5]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 5]

        if filtered_ask and ask > filtered_ask[0]:
            ask = max(filtered_ask[0] - 0.1, implied_ask + edge_factor)
        if filtered_bid and bid < filtered_bid[0]:
            bid = min(filtered_bid[0] + 0.1, implied_bid - edge_factor)

        net_position = position + buy_order_volume - sell_order_volume

        # Place sell order
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.MAGNIFICENT_MACARONS, round(ask), -sell_quantity))
            logger.print(f"arb_make: Sell Order Price={ask}, Quantity={sell_quantity}, vs BestAsk={best_ask}")

        # Place buy order
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            if net_position > -60:
                buy_quantity = max(0, min(buy_quantity, 30 + net_position))
                if buy_quantity > 0:
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, round(bid), buy_quantity))
                    logger.print(f"arb_make: Buy Order Price={bid}, Quantity={buy_quantity}, vs BestBid={best_bid}")

        # Default orders if none generated
        if not orders:
            if buy_quantity > 0:
                orders.append(Order(Product.MAGNIFICENT_MACARONS, round(best_ask), min(buy_quantity, 20)))
                logger.print(f"arb_make: Default Buy Order Price={best_ask}, Quantity={min(buy_quantity, 20)}, vs BestAsk={best_ask}")
            if sell_quantity > 0:
                orders.append(Order(Product.MAGNIFICENT_MACARONS, round(best_bid), -min(sell_quantity, 20)))
                logger.print(f"arb_make: Default Sell Order Price={best_bid}, Quantity={min(sell_quantity, 20)}, vs BestBid={best_bid}")

        logger.print(f"arb_make: SunlightIndex={sunlight_index}, BuyOrders={len([o for o in orders if o.quantity > 0])}, SellOrders={len([o for o in orders if o.quantity < 0])}")
        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        trader_object = {}
        if state.traderData:
            try:
                trader_object = jsonpickle.decode(state.traderData)
            except Exception as e:
                logger.print(f"Error decoding traderData: {e}")
                trader_object = {}

        result = {}
        conversions = 0

        if Product.MAGNIFICENT_MACARONS not in state.order_depths or Product.MAGNIFICENT_MACARONS not in state.observations.conversionObservations:
            logger.print("Missing data for MAGNIFICENT_MACARONS")
            return {}, 0, jsonpickle.encode(trader_object)

        macaron_position = state.position.get(Product.MAGNIFICENT_MACARONS, 0)
        observation = state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS]
        sunlight_index = observation.sunlightIndex

        logger.print(f"run: Timestamp={state.timestamp}, Position={macaron_position}, SunlightIndex={sunlight_index}")
        logger.print(f"run: MarketDepth BuyOrders={state.order_depths[Product.MAGNIFICENT_MACARONS].buy_orders}, SellOrders={state.order_depths[Product.MAGNIFICENT_MACARONS].sell_orders}")

        if Product.MAGNIFICENT_MACARONS not in trader_object:
            trader_object[Product.MAGNIFICENT_MACARONS] = {
                "curr_edge": self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"],
                "volume_history": [],
                "optimized": True,
                "last_conversion_time": 0
            }

        # Aggressive conversions
        if macaron_position != 0:
            if (state.timestamp - trader_object[Product.MAGNIFICENT_MACARONS]["last_conversion_time"] > 10 or
                abs(macaron_position) > 5):
                conversions = self.arb_clear(macaron_position, sunlight_index)
                trader_object[Product.MAGNIFICENT_MACARONS]["last_conversion_time"] = state.timestamp
            else:
                conversions = self.arb_clear(macaron_position, sunlight_index)

        adap_edge = self.adap_edge(
            state.timestamp,
            trader_object[Product.MAGNIFICENT_MACARONS].get("curr_edge", self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"]),
            macaron_position,
            trader_object,
            sunlight_index
        )

        adjusted_position = macaron_position + conversions

        macaron_take_orders, buy_order_volume, sell_order_volume = self.arb_take(
            state.order_depths[Product.MAGNIFICENT_MACARONS],
            observation,
            adap_edge,
            adjusted_position,
            sunlight_index
        )

        macaron_make_orders, _, _ = self.arb_make(
            state.order_depths[Product.MAGNIFICENT_MACARONS],
            observation,
            adjusted_position,
            adap_edge,
            buy_order_volume,
            sell_order_volume,
            sunlight_index
        )

        result[Product.MAGNIFICENT_MACARONS] = macaron_take_orders + macaron_make_orders
        trader_object[Product.MAGNIFICENT_MACARONS]["curr_edge"] = adap_edge

        if adjusted_position + buy_order_volume - sell_order_volume != 0:
            expected_position = adjusted_position + buy_order_volume - sell_order_volume
            expected_storage_cost = abs(expected_position) * 0.1
            trader_object[Product.MAGNIFICENT_MACARONS]["expected_storage_cost"] = expected_storage_cost
            if conversions == 0 and macaron_position != 0:
                conversions = 1  # Force position adjustment

        logger.print(f"run: Conversions={conversions}, BuyVolume={buy_order_volume}, SellVolume={sell_order_volume}, TotalOrders={len(result[Product.MAGNIFICENT_MACARONS])}")

        trader_data = jsonpickle.encode(trader_object)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data