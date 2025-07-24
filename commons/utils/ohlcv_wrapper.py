import numpy as np

from commons.models.ohlcv_type import Ohlcv


class OhlcvWrapper:
    def __init__(self, ohlcv):
        self.ohlcv = ohlcv
        self.timestamps = [c[0] for c in ohlcv]
        self.opens = [c[1] for c in ohlcv]
        self.highs = [c[2] for c in ohlcv]
        self.lows = [c[3] for c in ohlcv]
        self.closes = [c[4] for c in ohlcv]
        self.volumes = [c[5] for c in ohlcv]
        self.raw = ohlcv

    def get_candle(self, index) -> Ohlcv:
        return Ohlcv(self.timestamps[index], self.opens[index], self.highs[index], self.lows[index], self.closes[index], self.volumes[index])

    
    def get_last_closed_candle(self) -> Ohlcv:
        """Último candle fechado"""
        return self.get_candle(-2 if len(self.ohlcv) >= 2 else -1)

    def get_current_candle(self) -> Ohlcv:
        """Candle mais recente (em formação se em tempo real)"""
        return self.get_candle(-1)

    def get_previous_candle(self) -> Ohlcv:
        """Candle anterior ao último fechado"""
        return self.get_candle(-3 if len(self.ohlcv) >= 3 else -1)
    
    def get_close(self, index: int = -1):
        return self.closes[index]

    def get_open(self, index: int = -1):
        return self.opens[index]
    
    def get_recent_closed(self, lookback: int) -> list[Ohlcv]:
        """
        Retorna os últimos `lookback` candles **fechados**, excluindo o candle atual.
        """
        if len(self.ohlcv) < lookback + 1:
            return [self.get_candle(i) for i in range(len(self.ohlcv) - 1)]
        return [self.get_candle(i) for i in range(-lookback - 1, -1)]
    

    def __len__(self):
        return len(self.ohlcv)
    
    def candles_to_arrays(self):
        opens = np.array([c.open for c in self.ohlcv])
        highs = np.array([c.high for c in self.ohlcv])
        lows = np.array([c.low for c in self.ohlcv])
        closes = np.array([c.close for c in self.ohlcv])
        volumes = np.array([c.volume for c in self.ohlcv])
        return opens, highs, lows, closes, volumes