from abc import ABC, abstractmethod
from typing import Any, List

from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.ohlcv_wrapper import OhlcvWrapper


class StrategyBase(ABC):
    MIN_REQUIRED_CANDLES = 50
    VOLUME_ANALYSIS_PERIOD = 20
    ohlcv: OhlcvWrapper

    @abstractmethod
    async def get_signal(self) -> Any:
        pass

    @abstractmethod
    def set_params(self, params: StrategyParams):
        pass

    @abstractmethod
    def set_candles(self, candles: List[list]):
        pass

    @abstractmethod
    def set_higher_timeframe_candles(self, ohlcv_higher: List[list]):
        self.ohlcv_higher = ohlcv_higher

    @abstractmethod
    def required_init(self, ohlcv: OhlcvWrapper, ohlcv_higher: (OhlcvWrapper| None), symbol: str, price_ref: float):
        pass

    def has_enough_candles(self) -> bool:
        return hasattr(self, "ohlcv") and len(self.ohlcv) >= self.MIN_REQUIRED_CANDLES