from abc import ABC, abstractmethod

from commons.enums.signal_enum import Signal
from commons.enums.timeframe_enum import TimeframeEnum
from commons.models.ohlcv_format_dclass import OhlcvFormat
from commons.models.open_position_dclass import OpenPosition
from commons.models.opened_order_dclass import OpenedOrder
from commons.utils.config_loader import PairConfig


class ExchangeBase(ABC):
    
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: TimeframeEnum, limit: int, is_higher: bool = False) -> OhlcvFormat:
        pass

    @abstractmethod
    async def get_available_balance(self) -> float:
        pass

    @abstractmethod
    async def get_open_position(self, symbol: str) -> (OpenPosition | None):
        pass

    @abstractmethod
    async def place_entry_order(self, symbol: str, size: float, side: Signal) -> OpenedOrder:
        pass

    @abstractmethod
    async def cancel_all_orders(self, symbol: str):
        pass

    @abstractmethod
    async def close_position(self, symbol: str, amount: float, side: Signal):
        pass

    @abstractmethod
    async def get_entry_price(self, symbol: str) -> float:
        pass
    
    @abstractmethod
    async def open_new_position(self, symbol: str, leverage: float, signal: Signal, capital_amount: float, pair: PairConfig, sl: (float|None), tp: (float|None)) -> (OpenedOrder | None):
         pass
    
    @abstractmethod
    async def print_open_orders(self, symbol: str):
        pass

    @abstractmethod
    async def print_balance(self):
        pass