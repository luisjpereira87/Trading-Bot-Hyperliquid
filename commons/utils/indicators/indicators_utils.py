import numpy as np
import pandas as pd

from commons.utils.ohlcv_wrapper import OhlcvWrapper


class IndicatorsUtils:
    def __init__(self, ohlcv: OhlcvWrapper, mode='ta'):
        """
        ohlcv: lista de velas, onde cada vela é [timestamp, open, high, low, close, volume]
        mode: 'custom' (default) para usar seus cálculos manuais,
              'ta' para usar a biblioteca ta.
        """
        self.ohlcv = ohlcv
        self.mode = mode
        self.opens = ohlcv.opens
        self.highs = ohlcv.highs
        self.lows = ohlcv.lows
        self.closes = ohlcv.closes
        self.volumes = ohlcv.volumes

        if self.mode == 'ta':
            from ta.momentum import RSIIndicator, StochasticOscillator
            from ta.trend import ADXIndicator, EMAIndicator, PSARIndicator
            from ta.volatility import AverageTrueRange

            
            self.EMAIndicator = EMAIndicator
            self.ADXIndicator = ADXIndicator
            self.RSIIndicator = RSIIndicator
            self.StochasticOscillator = StochasticOscillator
            self.AverageTrueRange = AverageTrueRange
            self.PSARIndicator = PSARIndicator

        self.pd = pd
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
    def ema_array(self, values, period=21):
        ema = []
        k = 2 / (period + 1)
        for i in range(len(values)):
            if i < period:
                ema.append(np.mean(values[:i+1]))
            else:
                ema.append(values[i]*k + ema[i-1]*(1-k))
        return np.array(ema)

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

    def atr(self, period=14) -> list[float]:
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
        
    def macd(self, fast_period=12, slow_period=26, signal_period=9):
        if self.mode == 'custom':
            ema_fast = self.ema(fast_period)
            ema_slow = self.ema(slow_period)
            macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]

            # Calcula EMA da linha MACD para sinal
            signal_line = []
            k = 2 / (signal_period + 1)
            for i in range(len(macd_line)):
                if i == 0:
                    signal_line.append(macd_line[0])
                else:
                    val = macd_line[i] * k + signal_line[-1] * (1 - k)
                    signal_line.append(val)
            return macd_line, signal_line
        else:
            from ta.trend import MACD
            macd_indicator = MACD(close=self.df['close'], window_slow=slow_period, window_fast=fast_period, window_sign=signal_period)
            macd_line = macd_indicator.macd().tolist()
            signal_line = macd_indicator.macd_signal().tolist()
            return macd_line, signal_line
        
    def cci(self, period=20):
        if self.mode == 'custom':
            typical_prices = [(h + l + c) / 3 for h, l, c in zip(self.highs, self.lows, self.closes)]

            # SMA dos Typical Prices
            sma = []
            for i in range(len(typical_prices)):
                if i < period - 1:
                    sma.append(0)
                else:
                    sma.append(np.mean(typical_prices[i - period + 1:i + 1]))

            # Mean Deviation
            mean_dev = []
            for i in range(len(typical_prices)):
                if i < period - 1:
                    mean_dev.append(0)
                else:
                    window = typical_prices[i - period + 1:i + 1]
                    mean = sma[i]
                    md = np.mean([abs(tp - mean) for tp in window])
                    mean_dev.append(md)

            # CCI Calculation
            cci = []
            constant = 0.015
            for i in range(len(typical_prices)):
                if mean_dev[i] == 0:
                    cci.append(0)
                else:
                    val = (typical_prices[i] - sma[i]) / (constant * mean_dev[i])
                    cci.append(val)
            return cci
        else:
            from ta.trend import CCIIndicator
            cci_indicator = CCIIndicator(high=self.df['high'], low=self.df['low'], close=self.df['close'], window=period)
            cci_series = cci_indicator.cci()
        return cci_series.tolist()
    
    def psar(self, step=0.02, max_step=0.2):
        """
        Calcula o Parabolic SAR.
        step: incremento do fator de aceleração (AF)
        max_step: valor máximo do AF
        """
        if self.mode == 'custom':
            highs = self.df['high'].values
            lows = self.df['low'].values

            length = len(highs)
            psar = [0.0] * length
            bull = True
            af = step
            ep = lows[0]

            psar[0] = lows[0]

            for i in range(1, length):
                prev_psar = psar[i-1]
                psar[i] = prev_psar + af * (ep - prev_psar)

                if bull:
                    psar[i] = min(psar[i], lows[i-1], lows[i-2] if i >= 2 else lows[i-1])

                    if highs[i] > ep:
                        ep = highs[i]
                        af = min(af + step, max_step)

                    if lows[i] < psar[i]:
                        bull = False
                        psar[i] = ep
                        ep = lows[i]
                        af = step
                else:
                    psar[i] = max(psar[i], highs[i-1], highs[i-2] if i >= 2 else highs[i-1])

                    if lows[i] < ep:
                        ep = lows[i]
                        af = min(af + step, max_step)

                    if highs[i] > psar[i]:
                        bull = True
                        psar[i] = ep
                        ep = highs[i]
                        af = step

            return psar
        else:
            psar_series = self.PSARIndicator(
                high=self.df['high'],
                low=self.df['low'],
                close=self.df['close'],  # obrigatório mas ignorado pela lib
                step=step,
                max_step=max_step
            ).psar()
            return psar_series.tolist()

    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0):
        """
        Calcula as Bandas de Bollinger.
        Retorna: (upper_band, middle_band, lower_band)
        """
        closes = self.df['close']

        middle_band = closes.rolling(window=period).mean()
        std = closes.rolling(window=period).std()

        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)

        return upper_band.values, middle_band.values, lower_band.values