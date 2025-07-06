import logging

from strategies.ai_supertrend import AISuperTrend
from strategies.combined import CombinedStrategy
from strategies.ml_strategy import MLModelType, MLStrategy
from strategies.signal_result import SignalResult
from strategies.strategy_base import StrategyBase
from strategies.supertrend import SuperTrend
from strategies.ut_bot_alerts import UTBotAlerts


class StrategyManager:
    def __init__(self, exchange, symbol, timeframe, name='ai_supertrend'):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.name = name.lower()
        self.strategy:StrategyBase
        self.mode = 'normal'

    async def get_signal(self)->SignalResult:
        if self.name == 'ai_supertrend':
            self.mode = await self._detect_mode()
            self.strategy = AISuperTrend(self.exchange, self.symbol, self.timeframe, mode=self.mode)
        #elif self.name == 'combined':
        #    self.strategy = CombinedStrategy(self.exchange, self.symbol, self.timeframe)
        elif self.name == 'ml':
            self.strategy = MLStrategy(self.exchange, self.symbol, self.timeframe, model_type=MLModelType.XGBOOST)
        #elif self.name == 'supertrend':
        #    self.strategy = SuperTrend(self.exchange, self.symbol, self.timeframe)
        #elif self.name == 'ut_bot_alerts':
        #    self.strategy = UTBotAlerts(self.exchange, self.symbol, self.timeframe)
        else:
            raise ValueError(f"Estratégia '{self.name}' não reconhecida.")

        return await self.strategy.get_signal()
        #return {'side':signal, 'mode':self.mode} 
    

    async def _detect_mode(self, period=20, volume_multiplier=1.1):
        try:
            # Pega o histórico com limite = período + 1 para pegar vela atual e média
            ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=period + 1)
            
            # Checa se tem dados suficientes
            if len(ohlcv) <= period:
                logging.info(f"[VolumeAnalyzer] {self.symbol} Dados insuficientes para detectar modo. Usando modo conservador.")
                return 'conservative'
            
            # Pega volumes do período anterior (ignora a vela atual que pode estar incompleta)
            volumes = [candle[5] for candle in ohlcv[-(period + 1):-1]]
            avg_volume = sum(volumes) / len(volumes)
            current_volume = ohlcv[-1][5]  # volume da vela atual
            
            if current_volume > avg_volume * volume_multiplier:
                logging.info(f"[VolumeAnalyzer] {self.symbol} Volume ALTO: {current_volume} > {avg_volume:.6f} * {volume_multiplier}")
                return 'aggressive'
            else:
                logging.info(f"[VolumeAnalyzer] {self.symbol} Volume BAIXO: {current_volume} <= {avg_volume:.6f} * {volume_multiplier}")
                return 'conservative'
            
        except Exception as e:
            logging.error(f"[VolumeAnalyzer] Erro ao analisar volume para {self.symbol}: {e}")
            return 'conservative'

