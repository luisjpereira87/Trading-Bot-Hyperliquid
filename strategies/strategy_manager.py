import logging
from typing import List

from commons.enums.strategy_enum import StrategyEnum
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from strategies.cross_ema_linear_regression_strategy import \
    CrossEmaLinearRegressionStrategy
from strategies.cross_ema_strategy import CrossEmaStrategy
from strategies.luxalgo_supertrend_strategy import LuxAlgoSupertrendStrategy
from strategies.ml_strategy import MLModelType, MLStrategy
from trading_bot.exchange_base import ExchangeBase
from trading_bot.exchange_client import ExchangeClient


class StrategyManager(StrategyBase):
    def __init__(self, exchange: ExchangeBase, name=StrategyEnum.AI_SUPERTREND):
        super().__init__()
        self.exchange = exchange
        self.symbol = None
        self.ohlcv: OhlcvWrapper
        self.timeframe = None
        self.name = name
        self.strategy:StrategyBase
        self.mode = 'normal'

        self._load_strategy()

    def _load_strategy(self):
        if self.name == StrategyEnum.CROSS_EMA:
            self.strategy = CrossEmaStrategy(self.exchange)
        elif self.name == StrategyEnum.LUXALGO_SUPERTREND:
            self.strategy = LuxAlgoSupertrendStrategy(self.exchange)
        elif self.name == StrategyEnum.ML_XGBOOST:
            self.strategy = MLStrategy(self.exchange, MLModelType.XGBOOST)
        elif self.name == StrategyEnum.ML_MLP:
            self.strategy = MLStrategy(self.exchange, MLModelType.MLP)
        elif self.name == StrategyEnum.ML_RANDOM_FOREST:
            self.strategy = MLStrategy(self.exchange, MLModelType.RANDOM_FOREST)
        elif self.name == StrategyEnum.ML_LSTM:
            self.strategy = MLStrategy(self.exchange, MLModelType.LSTM)
        elif self.name == StrategyEnum.CROSS_EMA_LINEAR_REGRESSION:
            self.strategy = CrossEmaLinearRegressionStrategy(self.exchange)
        else:
            raise ValueError(f"Estratégia '{self.name}' não reconhecida.")
        

    async def get_signal(self)-> SignalResult:
        return await self.strategy.get_signal()


    def required_init(self, ohlcv: OhlcvWrapper, ohlcv_higher: (OhlcvWrapper|None), symbol: str, price_ref: float):
        self.ohlcv = ohlcv
        self.symbol = symbol
        self.strategy.required_init(ohlcv, ohlcv_higher, symbol, price_ref)
        
    def set_candles(self, ohlcv):
        self.strategy.set_candles(ohlcv)

    def set_higher_timeframe_candles(self, ohlcv):
        self.strategy.set_higher_timeframe_candles(ohlcv)


