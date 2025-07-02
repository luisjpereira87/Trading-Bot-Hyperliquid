import numpy as np


class Indicators:
    def __init__(self, ohlcv, mode='ta'):
        """
        ohlcv: lista de velas, onde cada vela é [timestamp, open, high, low, close, volume]
        mode: 'custom' (default) para usar seus cálculos manuais,
              'ta' para usar a biblioteca ta.
        """
        self.ohlcv = ohlcv
        self.mode = mode

        self.opens = [c[1] for c in ohlcv]
        self.highs = [c[2] for c in ohlcv]
        self.lows = [c[3] for c in ohlcv]
        self.closes = [c[4] for c in ohlcv]
        self.volumes = [c[5] for c in ohlcv] if len(ohlcv[0]) > 5 else []

        if self.mode == 'ta':
            import pandas as pd
            from ta.momentum import RSIIndicator, StochasticOscillator
            from ta.trend import ADXIndicator, EMAIndicator
            from ta.volatility import AverageTrueRange

            self.pd = pd
            self.EMAIndicator = EMAIndicator
            self.ADXIndicator = ADXIndicator
            self.RSIIndicator = RSIIndicator
            self.StochasticOscillator = StochasticOscillator
            self.AverageTrueRange = AverageTrueRange

            self.df = pd.DataFrame({
                'open': self.opens,
                'high': self.highs,
                'low': self.lows,
                'close': self.closes,
                'volume': self.volumes if self.volumes else [0]*len(self.closes),
            })

    def ema(self, period=21):
        if self.mode == 'custom':
            ema = []
            k = 2 / (period + 1)
            for i in range(len(self.closes)):
                if i < period:
                    ema.append(np.mean(self.closes[:i + 1]))
                else:
                    ema.append((self.closes[i] * k) + (ema[i - 1] * (1 - k)))
            return ema
        else:
            ema_series = self.EMAIndicator(close=self.df['close'], window=period).ema_indicator()
            return ema_series.tolist()

    def rsi(self, period=14):
        if self.mode == 'custom':
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
        else:
            rsi_series = self.RSIIndicator(close=self.df['close'], window=period).rsi()
            return rsi_series.tolist()

    def atr(self, period=14):
        if self.mode == 'custom':
            trs = []
            for i in range(1, len(self.highs)):
                tr = max(
                    self.highs[i] - self.lows[i],
                    abs(self.highs[i] - self.closes[i - 1]),
                    abs(self.lows[i] - self.closes[i - 1])
                )
                trs.append(tr)

            atr = []
            atr.append(trs[0])  # primeiro valor do ATR

            alpha = 2 / (period + 1)

            for i in range(1, len(trs)):
                atr_val = (alpha * trs[i]) + ((1 - alpha) * atr[i-1])
                atr.append(atr_val)

            atr.insert(0, 0)  # alinhamento com o tamanho dos dados

            return atr
        else:
            atr_series = self.AverageTrueRange(
                high=self.df['high'], low=self.df['low'], close=self.df['close'], window=period
            ).average_true_range()
            return atr_series.tolist()

    def stochastic(self, k_period=14, d_period=3):
        if self.mode == 'custom':
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
        else:
            stoch = self.StochasticOscillator(
                high=self.df['high'],
                low=self.df['low'],
                close=self.df['close'],
                window=k_period,
                smooth_window=d_period
            )
            k_values = stoch.stoch().tolist()
            d_values = stoch.stoch_signal().tolist()
            return k_values, d_values

    def adx(self, period=14):
        if self.mode == 'custom':
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
        else:
            adx_series = self.ADXIndicator(
                high=self.df['high'], low=self.df['low'], close=self.df['close'], window=period
            ).adx()
            return adx_series.tolist()
