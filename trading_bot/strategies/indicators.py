import numpy as np  # type: ignore


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

    def atr(self, period=14):
        trs = [
            max(
                self.highs[i] - self.lows[i],
                abs(self.highs[i] - self.closes[i - 1]),
                abs(self.lows[i] - self.closes[i - 1])
            )
            for i in range(1, len(self.highs))
        ]
        atr = []
        for i in range(len(trs)):
            if i >= period:
                atr_value = np.mean(trs[i - period + 1 : i + 1])
            else:
                atr_value = 0
            atr.append(atr_value)
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
    
    def adx(self, period=14):
        highs = np.array(self.highs)
        lows = np.array(self.lows)
        closes = np.array(self.closes)

        plus_dm = np.zeros(len(highs))
        minus_dm = np.zeros(len(highs))
        tr = np.zeros(len(highs))

        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]

            plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
            minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0

            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

        # Wilder's smoothing
        tr_smooth = np.zeros(len(tr))
        plus_dm_smooth = np.zeros(len(plus_dm))
        minus_dm_smooth = np.zeros(len(minus_dm))

        tr_smooth[period] = tr[1 : period + 1].sum()
        plus_dm_smooth[period] = plus_dm[1 : period + 1].sum()
        minus_dm_smooth[period] = minus_dm[1 : period + 1].sum()

        for i in range(period + 1, len(tr)):
            tr_smooth[i] = tr_smooth[i - 1] - (tr_smooth[i - 1] / period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i - 1] - (plus_dm_smooth[i - 1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i - 1] - (minus_dm_smooth[i - 1] / period) + minus_dm[i]

        epsilon = 1e-10

        plus_di = 100 * (plus_dm_smooth / (tr_smooth + epsilon))
        minus_di = 100 * (minus_dm_smooth / (tr_smooth + epsilon))

        denominator = plus_di + minus_di
        denominator = np.where(denominator == 0, epsilon, denominator)

        dx = 100 * np.abs(plus_di - minus_di) / denominator

        adx = np.zeros(len(dx))
        adx[period * 2 - 1] = dx[period : period * 2].mean()

        for i in range(period * 2, len(dx)):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        return adx.tolist()

