import logging
from typing import List

from commons.enums.strategy_enum import StrategyEnum
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from strategies.ai_supertrend_strategy import AISuperTrendStrategy
from strategies.ml_strategy import MLModelType, MLStrategy
from trading_bot.exchange_client import ExchangeClient


class StrategyManager(StrategyBase):
    def __init__(self, exchange: ExchangeClient, name=StrategyEnum.AI_SUPERTREND):
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
        if self.name == StrategyEnum.AI_SUPERTREND:
            self.strategy = AISuperTrendStrategy(self.exchange)
        elif self.name == StrategyEnum.ML_XGBOOST:
            self.strategy = MLStrategy(self.exchange, MLModelType.XGBOOST)
        elif self.name == StrategyEnum.ML_MLP:
            self.strategy = MLStrategy(self.exchange, MLModelType.MLP)
        elif self.name == StrategyEnum.ML_RANDOM_FOREST:
            self.strategy = MLStrategy(self.exchange, MLModelType.RANDOM_FOREST)
        elif self.name == StrategyEnum.ML_LSTM:
            self.strategy = MLStrategy(self.exchange, MLModelType.LSTM)
        else:
            raise ValueError(f"Estratégia '{self.name}' não reconhecida.")
        

    async def get_signal(self)-> SignalResult:
        #if self.name == StrategyEnum.AI_SUPERTREND:
        #self.mode = await self._detect_mode()

        return await self.strategy.get_signal()


    def required_init(self, ohlcv: OhlcvWrapper, ohlcv_higher: (OhlcvWrapper|None), symbol: str, price_ref: float):
        self.ohlcv = ohlcv
        self.symbol = symbol
        self.strategy.required_init(ohlcv, ohlcv_higher, symbol, price_ref)
    
    def set_params(self, params: StrategyParams):
        self.strategy.set_params(params)
        
    def set_candles(self, ohlcv):
        self.strategy.set_candles(ohlcv)

    def set_higher_timeframe_candles(self, ohlcv):
        self.strategy.set_higher_timeframe_candles(ohlcv)
    

    async def _detect_mode(self, volume_multiplier=1.1):
        try:
            if len(self.ohlcv) < self.VOLUME_ANALYSIS_PERIOD:
                logging.info(f"[VolumeAnalyzer] {self.symbol} Dados insuficientes para detectar modo. Usando modo conservador.")
                return 'conservative'

            volumes = [self.ohlcv.get_candle(i).volume for i in range(-self.VOLUME_ANALYSIS_PERIOD - 1, -1)]
            avg_volume = sum(volumes) / len(volumes)
            current_volume = self.ohlcv.get_current_candle().volume

            if current_volume > avg_volume * volume_multiplier:
                logging.info(f"[VolumeAnalyzer] {self.symbol} Volume ALTO: {current_volume:.2f} > {avg_volume:.2f} * {volume_multiplier}")
                return 'aggressive'
            else:
                logging.info(f"[VolumeAnalyzer] {self.symbol} Volume BAIXO: {current_volume:.2f} <= {avg_volume:.2f} * {volume_multiplier}")
                return 'conservative'

        except Exception as e:
            logging.error(f"[VolumeAnalyzer] Erro ao analisar volume para {self.symbol}: {e}")
            return 'conservative'

