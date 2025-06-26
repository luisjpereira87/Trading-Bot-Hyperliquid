import numpy as np

class Indicators:
    def __init__(self, ohlcv):
        """
        ohlcv: lista de velas, onde cada vela é [timestamp, open, high, low, close, volume]
        """
        self.ohlcv = ohlcv
        self.closes = [c[4] for c in ohlcv]
        self.highs = [c[2] for c in ohlcv]
        self.lows = [c[3] for c in ohlcv]

    def ema(self, period=21):
        ema = []
        k = 2 / (period + 1)
        for i in range(len(self.closes)):
            if i < period:
                ema.append(np.mean(self.closes[:i + 1]))
            else:
                ema.append((self.closes[i] * k) + (ema[i - 1] * (1 - k)))
        return ema

    def rsi(self, period=14):
        deltas = np.diff(self.closes)
        seed = deltas[:period]
        up = seed[seed > 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = [100 - 100 / (1 + rs)]

        for i in range(period, len(deltas)):
            delta = deltas[i]
            gain = max(delta, 0)
            loss = -min(delta, 0)
            up = (up * (period - 1) + gain) / period
            down = (down * (period - 1) + loss) / period
            rs = up / down if down != 0 else 0
            rsi.append(100 - 100 / (1 + rs))

        return [0] * (len(self.closes) - len(rsi)) + rsi

    def atr(self, period=10):
        trs = [max(self.highs[i] - self.lows[i], abs(self.highs[i] - self.closes[i - 1]), abs(self.lows[i] - self.closes[i - 1]))
               for i in range(1, len(self.highs))]
        atr = [np.mean(trs[i - period:i]) if i >= period else 0 for i in range(len(trs))]
        atr.insert(0, 0)
        return atr

    def stochastic(self, k_period=14, d_period=3):
        """
        Retorna tuple (%K, %D), ambos como listas alinhadas ao tamanho dos closes.
        %K é a linha rápida, %D é a média móvel simples de %K.
        """
        k_values = []
        for i in range(len(self.closes)):
            if i < k_period - 1:
                k_values.append(0)
            else:
                low_min = min(self.lows[i - k_period + 1:i + 1])
                high_max = max(self.highs[i - k_period + 1:i + 1])
                denominator = high_max - low_min
                if denominator == 0:
                    k_values.append(0)
                else:
                    k = 100 * (self.closes[i] - low_min) / denominator
                    k_values.append(k)

        d_values = []
        for i in range(len(k_values)):
            if i < d_period - 1:
                d_values.append(0)
            else:
                d_values.append(np.mean(k_values[i - d_period + 1:i + 1]))

        return k_values, d_values

