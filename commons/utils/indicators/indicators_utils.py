import numpy as np
import pandas as pd

from commons.enums.candle_type_enum import CandleType
from commons.enums.signal_enum import Signal
from commons.models.supertrend_dclass import Supertrend
from commons.models.volumatic_vidya_dclass import VolumaticVidya
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
        self._locked_target_factor = {}
        self._last_atr = {}

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

    def set_ohlcv(self, ohlcv: OhlcvWrapper):
        self.ohlcv = ohlcv
        self.opens = ohlcv.opens
        self.highs = ohlcv.highs
        self.lows = ohlcv.lows
        self.closes = ohlcv.closes
        self.volumes = ohlcv.volumes

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

        return upper_band.values.tolist(), middle_band.values.tolist(), lower_band.values.tolist()
    
    def supertrend(self,
                   mode="adaptive",
                   base_length=10, base_mult=3.0,
                   min_len=7, max_len=21,
                   min_mult=2.0, max_mult=4.0,
                   vol_sensitivity=1.5,
                   trend_confirmation=1,
                   smooth_period=3):

        opens, highs, lows, closes, volumes = self.ohlcv.candles_to_arrays()
        hl2 = (highs + lows) / 2
        n = len(closes)

        # ATR fixo para medir volatilidade
        atr_fixed = self.atr(14)
        vol_rel = atr_fixed / closes

        atr_mean = np.mean(atr_fixed)
        atr_std = np.std(atr_fixed)

        # Ajuste dinâmico do range de fatores
        #min_mult = max(0.5, 1.0 - atr_std / atr_mean)
        #max_mult = min(3.0, 2.0 + atr_std / atr_mean)

        # calcular arrays de length e multiplier
        if mode == "adaptive":
            atr_len = base_length * (1 / (1 + vol_sensitivity * vol_rel))
            atr_len = np.clip(atr_len, min_len, max_len)
            atr_mult = base_mult * (1 + vol_sensitivity * vol_rel)
            atr_mult = np.clip(atr_mult, min_mult, max_mult)

            # suavizar com numpy (rolling mean simples)
            def rolling_mean(arr, window=3):
                if window <= 1:
                    return arr
                kernel = np.ones(window) / window
                return np.convolve(arr, kernel, mode="same")

            atr_len = rolling_mean(atr_len, 3)
            atr_mult = rolling_mean(atr_mult, 3)
        else:
            atr_len = np.full(n, base_length)
            atr_mult = np.full(n, base_mult)

        # pré-calcular ATRs de todos os comprimentos possíveis
        atr_cache = {L: self.atr(L) for L in range(min_len, max_len + 1)}

        supertrend = np.zeros(n)
        trend = np.ones(n)
        final_upperband = np.zeros(n)
        final_lowerband = np.zeros(n)
        direction = np.zeros(n, dtype=int) 
        trend_count = 0

        for i in range(n):
            length = int(round(atr_len[i]))
            length = max(min_len, min(max_len, length))
            mult = atr_mult[i]
            atr_curr = atr_cache[length][i]  # usar cache

            if np.isnan(atr_curr):
                supertrend[i] = np.nan
                trend[i] = 1
                final_upperband[i] = np.nan
                final_lowerband[i] = np.nan
                continue

            upperband = hl2[i] + mult * atr_curr
            lowerband = hl2[i] - mult * atr_curr

            if i == 0 or np.isnan(supertrend[i - 1]):
                supertrend[i] = hl2[i]
                trend[i] = 1
                final_upperband[i] = upperband
                final_lowerband[i] = lowerband
                trend_count = 1
                direction[i] = 1
                continue

            final_upperband[i] = upperband if (upperband < final_upperband[i - 1] or closes[i - 1] > final_upperband[i - 1]) else final_upperband[i - 1]
            final_lowerband[i] = lowerband if (lowerband > final_lowerband[i - 1] or closes[i - 1] < final_lowerband[i - 1]) else final_lowerband[i - 1]

            prev_trend = trend[i - 1]
            if prev_trend == 1:
                if closes[i] <= final_upperband[i]:
                    trend_count += 1
                    if trend_count >= trend_confirmation:
                        trend[i] = -1
                        supertrend[i] = final_upperband[i]
                        trend_count = 0
                    else:
                        trend[i] = prev_trend
                        supertrend[i] = supertrend[i - 1]
                else:
                    trend[i] = 1
                    supertrend[i] = final_lowerband[i]
                    trend_count = 0
            else:
                if closes[i] >= final_lowerband[i]:
                    trend_count += 1
                    if trend_count >= trend_confirmation:
                        trend[i] = 1
                        supertrend[i] = final_lowerband[i]
                        trend_count = 0
                    else:
                        trend[i] = prev_trend
                        supertrend[i] = supertrend[i - 1]
                else:
                    trend[i] = -1
                    supertrend[i] = final_upperband[i]
                    trend_count = 0

            # --- direção contínua LuxAlgo-like ---
            if closes[i] > final_upperband[i]:
                direction[i] = 1
            elif closes[i] < final_lowerband[i]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]  # mantém direção anterior
       
        supertrend_smooth = self.ema_array(supertrend, smooth_period)

        # perf_score baseado na distância da vela ao supertrend
        """
        dist = np.abs(closes - supertrend_smooth)
        max_dist = np.max(dist) if np.max(dist) > 0 else 1.0
        perf_score = np.clip((max_dist - dist) / max_dist * 10, 0, 10).astype(int)
        """
        # Movimentos relativos
        perfAlpha=21
        absdiff = np.abs(np.diff(closes, prepend=closes[0]))
        alpha = 2 / (perfAlpha + 1)
        den = np.zeros(n)
        den[0] = absdiff[0]
        for i in range(1, n):
            den[i] = alpha * absdiff[i] + (1 - alpha) * den[i - 1]

        den_safe = np.where(den == 0, 1e-9, den)

        # Normalização por volatilidade
        avg_perf_norm = np.clip(np.mean(supertrend_smooth - hl2), 0.001, 10)
        scale_factor = np.clip(np.mean(atr_fixed) / np.mean(absdiff), 0.5, 50)
        perf_idx_series = (avg_perf_norm * scale_factor) / den_safe

        # Reescalamento para 0-10
        
        p5, p95 = np.percentile(perf_idx_series, [5, 95])
        perf_idx_series = np.clip(perf_idx_series, p5, p95)
        perf_idx_series = (perf_idx_series - p5) / (p95 - p5)
        perf_idx_series = np.power(perf_idx_series, 0.5) * 10

        # Finalmente, o perf_score
        perf_score = np.round(10 - perf_idx_series).astype(int)
        perf_score = np.clip(perf_score, 0, 10)
        return supertrend, trend, final_upperband, final_lowerband, supertrend_smooth, direction, perf_score

    
    def stop_atr_tradingview(self, period=1, multiplier=3.0):
        highs = np.array(self.ohlcv.highs)
        lows = np.array(self.ohlcv.lows)
        closes = np.array(self.ohlcv.closes)
        n = len(closes)

        hl2 = (highs + lows) / 2.0
        atr = np.array(self.atr(period=period)) * multiplier

        long_stop = np.zeros(n)
        short_stop = np.zeros(n)
        direction = np.ones(n)

        long_stop[0] = hl2[0] - atr[0]
        short_stop[0] = hl2[0] + atr[0]
        direction[0] = 1  # começa em alta

        for i in range(1, n):
            long_stop_prev = long_stop[i-1]
            short_stop_prev = short_stop[i-1]
            dir_prev = direction[i-1]

            long_stop[i] = hl2[i] - atr[i]
            short_stop[i] = hl2[i] + atr[i]

            # suavização igual ao PineScript
            if closes[i-1] > long_stop_prev:
                long_stop[i] = max(long_stop[i], long_stop_prev)
            if closes[i-1] < short_stop_prev:
                short_stop[i] = min(short_stop[i], short_stop_prev)

            # reversão só se o candle FECHA acima/abaixo da linha
            if dir_prev == -1 and closes[i] > short_stop_prev:
                direction[i] = 1
            elif dir_prev == 1 and closes[i] < long_stop_prev:
                direction[i] = -1
            else:
                direction[i] = dir_prev

        # linha única para plot
        value_to_plot = np.where(direction == 1, long_stop, short_stop)
        return value_to_plot, direction
    
    def luxalgo_supertrend_ai(
        self,
        symbol: str = '',
        length: int = 10,
        min_mult: float = 1.0,
        max_mult: float = 5.0,
        step: float = 0.5,
        perfAlpha: float = 10.0,
        fromCluster: str = 'Best',   # 'Best' | 'Average' | 'Worst'
        maxIter: int = 1000,
        maxData: int = 10000,
        showSignals: bool = True
    ):
        """
        Reproduces the LuxAlgo 'SuperTrend AI (Clustering)' indicator outputs.
        """
        import numpy as np

        highs = np.array(self.highs, dtype=float)
        lows  = np.array(self.lows, dtype=float)
        closes = np.array(self.closes, dtype=float)
        n = len(closes)
        hl2 = (highs + lows) / 2.0

        # --- ATR (Wilder) ---       
        atr = np.array(self.atr(length), dtype=float)
        atr = np.nan_to_num(atr, nan=0.0)

        # --- Factor range ---
        factors = []
        f = min_mult
        while f <= max_mult + 1e-9:
            factors.append(round(f, 10))
            f += step
        factor_vals = np.array(factors, dtype=float)
        m = len(factor_vals)

        # --- Inicialização ---
        upp = np.zeros((m, n))
        low = np.zeros((m, n))
        outp = np.zeros((m, n))
        perf = np.zeros((m, n))
        trends = np.ones((m, n), dtype=int)

        for idx in range(m):
            upp[idx, 0] = hl2[0] + atr[0] * factor_vals[idx]
            low[idx, 0] = hl2[0] - atr[0] * factor_vals[idx]
            outp[idx, 0] = hl2[0]
            perf[idx, 0] = 0.0
            trends[idx, 0] = 1

        # --- SuperTrend para cada fator ---
        for idx in range(m):
            fval = factor_vals[idx]
            for i in range(1, n):
                up = hl2[i] + atr[i] * fval
                dn = hl2[i] - atr[i] * fval
                prev_upper = upp[idx, i-1]
                prev_lower = low[idx, i-1]
                prev_out = outp[idx, i-1]
                prev_trend = trends[idx, i-1]

                upper = up if closes[i-1] >= prev_upper else min(up, prev_upper)
                lower = dn if closes[i-1] <= prev_lower else max(dn, prev_lower)

                if closes[i] > upper:
                    trend = 1
                elif closes[i] < lower:
                    trend = 0
                else:
                    trend = prev_trend

                out = lower if trend == 1 else upper
                diff = np.sign(closes[i-1] - prev_out)
                perf[idx, i] = perf[idx, i-1] + (2 / (perfAlpha + 1)) * ((closes[i] - closes[i-1]) * diff - perf[idx, i-1])

                upp[idx, i] = upper
                low[idx, i] = lower
                outp[idx, i] = out
                trends[idx, i] = trend

        # --- Clustering (k-means manual) ---
        data_vals = np.array([perf[idx, -1] for idx in range(m)], dtype=float)
        centroids = np.percentile(data_vals, [25, 50, 75])

        for iteration in range(maxIter):
            clusters_perf = [[] for _ in range(3)]
            clusters_factors = [[] for _ in range(3)]
            for i_val, val in enumerate(data_vals):
                j = int(np.argmin(np.abs(val - centroids)))
                clusters_perf[j].append(val)
                clusters_factors[j].append(factor_vals[i_val])
            new_centroids = np.array([
                np.mean(c) if len(c) > 0 else centroids[j]
                for j, c in enumerate(clusters_perf)
            ])
            if np.allclose(new_centroids, centroids, atol=1e-10):
                break
            centroids = new_centroids

        mapping = {'Worst': 0, 'Average': 1, 'Best': 2}
        from_idx = mapping.get(fromCluster, 2)

        avg_perf_cluster = max(
            np.mean(clusters_perf[from_idx]) if len(clusters_perf[from_idx]) > 0 else 0.0,
            0.0
        )
        target_factor = (
            np.mean(clusters_factors[from_idx])
            if len(clusters_factors[from_idx]) > 0
            else float(np.mean(factor_vals))
        )
        

        # --- Perf Index Series corrigido ---
        absdiff = np.abs(np.diff(closes, prepend=closes[0]))
        alpha = 1 / perfAlpha
        den = np.zeros(n)
        den[0] = absdiff[0]
        for i in range(1, n):
            den[i] = den[i - 1] * (1 - alpha) + absdiff[i] * alpha

        den_safe = np.where(den == 0, 1e-6, den)

        # --- Normalização dinâmica do índice de performance ---
        scale = np.maximum(np.mean(den_safe) / np.mean(atr + 1e-6), 1)
        perf_idx_series = (avg_perf_cluster / (den_safe / scale))
        perf_idx_series = np.clip(perf_idx_series, 0, 10)

        # --- SuperTrend final ---
        up = hl2 + atr * target_factor
        dn = hl2 - atr * target_factor
        upper, lower = np.copy(up), np.copy(dn)
        ts = np.zeros(n)
        os = np.zeros(n, dtype=int)
        os[0] = 0
        ts[0] = upper[0]

        for i in range(1, n):
            upper[i] = up[i] if closes[i-1] > upper[i-1] else min(up[i], upper[i-1])
            lower[i] = dn[i] if closes[i-1] < lower[i-1] else max(dn[i], lower[i-1])

            if closes[i] > upper[i]:
                os[i] = 1
            elif closes[i] < lower[i]:
                os[i] = 0
            else:
                os[i] = os[i - 1]

            ts[i] = lower[i] if os[i] == 1 else upper[i]

        # --- Perf Index Series (EMA smoothing como no Pine) ---
        absdiff = np.abs(np.diff(closes, prepend=closes[0]))
        alpha = 2 / (perfAlpha + 1)
        den = np.zeros(n)
        den[0] = absdiff[0]
        for i in range(1, n):
            den[i] = alpha * absdiff[i] + (1 - alpha) * den[i - 1]

        den_safe = np.where(den == 0, 1e-9, den)

            # --- Normalização dinâmica e calibração estatística ---
        avg_perf_norm = np.clip(avg_perf_cluster / (np.mean(np.abs(closes - hl2)) + 1e-9), 0.001, 10)

        mean_atr = np.mean(atr)
        mean_move = np.mean(np.abs(np.diff(closes)))
        if mean_move < 1e-9:
            mean_move = 1e-9

        scale_factor = np.clip(mean_atr / mean_move, 0.5, 50.0)
        perf_idx_series = (avg_perf_norm * scale_factor) / (den_safe + 1e-6)

        # --- Perf AMA ---
        perf_ama = np.zeros(n)
        perf_ama[0] = ts[0]
        for i in range(1, n):
            alpha_ama = np.clip(perf_idx_series[i] / 10, 0.02, 0.6)
            perf_ama[i] = perf_ama[i - 1] + alpha_ama * (ts[i] - perf_ama[i - 1])

        # --- Perf Score (igual ao PineScript) ---
        perf_score = np.round(10 - perf_idx_series).astype(int)
        perf_score = np.clip(perf_score, 0, 10)
        
        """
        print("perf_idx_series mean:", np.mean(perf_idx_series))
        print("perf_idx_series min:", np.min(perf_idx_series))
        print("perf_idx_series max:", np.max(perf_idx_series))
        print("unique perf_score:", np.unique(perf_score))
        print("ATR mean:", np.mean(atr), "ATR std:", np.std(atr))
        """
        #print("AQUII", os)
        return {
            "ts": ts,
            "direction": os,
            "perf_ama": perf_ama,
            "perf_idx": perf_idx_series,
            "perf_score": perf_score,
            "target_factor": target_factor,
            "factors_clusters": clusters_factors,
            "perf_clusters": clusters_perf,
            "centroids": centroids
        }
    
    def get_supertrend_stopatr_signals(self, atr_period=14, multiplier=3.0, lookback_slope=5, min_slope=1e-4):
        """
        Gera sinais de BUY/SELL baseado na inclinação das linhas Stop ATR.
        - Sinal só ocorre se ambas as linhas têm a mesma inclinação.
        - Lateralidade é ignorada.
        """
        closes = np.array(self.ohlcv.closes)
        n = len(closes)

        #stop_up, stop_down, trend = self.stop_atr_(period=atr_period, multiplier=multiplier)
        value, direction = self.stop_atr_tradingview(period=1, multiplier=3.0)
        signals = np.array([Signal.HOLD] * n)

        for i in range(2, n):
            if direction[i-1] == -1 and direction[i] == 1:
               signals[i] = Signal.BUY
            elif direction[i-1] == 1 and direction[i] == -1:
               signals[i] = Signal.SELL

        return signals.tolist()
    
    def detect_low_volatility(
        self,
        lookback=20,
        hist_window=200,
        adx_dynamic_factor=0.8,
        slope_threshold=0.015
    ):
        """
        Retorna um array booleano indicando regiões de baixa volatilidade / lateralidade.
        Usa ATR, ADX e inclinação das linhas da SuperTrend.
        """
        closes = np.array(self.ohlcv.closes)
        n = len(closes)
        if n < max(lookback, hist_window):
            return np.zeros(n, dtype=bool)

        # obtém indicadores (usa cache se já calculados)
        atr = self.atr()
        adx = self.adx()
        _, _, upperband, lowerband, _,_,_ = self.supertrend()

        low_vol = np.zeros(n, dtype=bool)

        for i in range(max(lookback, hist_window), n):
            closes_window = closes[i - hist_window : i]
            atr_window = atr[i - hist_window : i]
            adx_window = adx[i - hist_window : i]
            upperband_window = upperband[i - lookback : i]
            lowerband_window = lowerband[i - lookback : i]

            # --- ATR/ADX adaptativos ---
            atr_ratio = atr_window / closes_window
            avg_atr_recent = np.mean(atr_ratio[-lookback:])
            avg_atr_hist = np.mean(atr_ratio)
            avg_adx_recent = np.mean(adx_window[-lookback:])
            avg_adx_hist = np.mean(adx_window)

            atr_threshold = avg_atr_hist * 0.7
            adx_threshold = avg_adx_hist * adx_dynamic_factor

            atr_condition = avg_atr_recent < atr_threshold
            adx_condition = avg_adx_recent < adx_threshold

            # --- Inclinação da SuperTrend ---
            slope_upper = abs(upperband_window[-1] - upperband_window[0]) / closes_window[-1]
            slope_lower = abs(lowerband_window[-1] - lowerband_window[0]) / closes_window[-1]
            band_width = np.mean(upperband_window - lowerband_window) / closes_window[-1]

            supertrend_flat = (
                slope_upper < slope_threshold
                and slope_lower < slope_threshold
                and band_width < slope_threshold * 5
            )

            # --- Resultado ---
            low_vol[i] = (atr_condition and adx_condition) or supertrend_flat

        return low_vol
    
    def get_volatility_profile(self, atr: list[float], lookback: int = 50):
        """
        Mede a volatilidade média do ativo com base no ATR relativo.
        Retorna:
        - atr_rel (float): média do ATR/preço (ex: 0.018 = 1.8%)
        - profile (str): classificação qualitativa ("low", "medium", "high")
        """
        closes = np.array(self.ohlcv.closes)

        # evitar erro se houver poucos dados
        if len(closes) < lookback or len(atr) < lookback:
            atr_rel = atr[-1] / closes[-1] if len(atr) > 0 else 0
        else:
            atr_rel = np.mean(atr[-lookback:]) / closes[-1]

        # classificação qualitativa (ajusta conforme teu mercado)
        if atr_rel < 0.012:
            profile = "low"     # ex: BTC, ETH
            ema_spread = 0.002 
        elif atr_rel < 0.025:
            profile = "medium"  # ex: BNB, AVAX
            ema_spread = 0.003
        else:
            profile = "high"    # ex: SOL, meme coins
            ema_spread = 0.004

        return atr_rel, profile, ema_spread
    
    def supertrend_ai(
        self,
        length=10,
        minMult=1,
        maxMult=5,
        step=0.5,
        perfAlpha=10,
        fromCluster='Best',
        maxIter=1000,
        maxData=10000
    ) -> Supertrend:
        """
        SuperTrend AI (Clustering) fiel ao LuxAlgo PineScript
        Retorna:
            ts: trailing stop
            perf_ama: trailing stop adaptativo
            signals: 1=BUY, -1=SELL, 0=HOLD
            score: performance index do cluster selecionado
        """

        closes = np.array(self.closes)
        highs = np.array(self.highs)
        lows = np.array(self.lows)
        opens = np.array(self.opens)
        volumes = np.array(self.volumes)
        hl2 = (highs + lows) / 2
        n = len(closes)

        atr = np.array(self.atr(length))

        # ----------------------------
        # Classe SuperTrend
        # ----------------------------
        class ST:
            def __init__(self):
                self.upper = 0.0
                self.lower = 0.0
                self.output = 0.0
                self.perf = 0.0
                self.factor = 0.0
                self.trend = 1

        # ----------------------------
        # Inicialização
        # ----------------------------
        factors = np.arange(minMult, maxMult + step, step)
        holders = [ST() for _ in factors]
        st_final = ST()

        ts = np.zeros(n)
        perf_ama = np.zeros(n)
        signals = np.zeros(n, dtype=int)
        score = np.zeros(n)

        perf_idx_den = (
            pd.Series(np.abs(np.diff(closes, prepend=closes[0])))
            .ewm(span=perfAlpha)
            .mean()
            .to_numpy()
        )

        target_factor = factors[0]
        direction = 0            # estado atual
        direction_arr = np.zeros(n, dtype=int)  # para plot
        avg_vol_delta = 0
        up_trend_vol = 0
        down_trend_vol = 0
        delta_vol_pct = np.zeros(n)

        # ============================
        # LOOP PRINCIPAL
        # ============================
        for i in range(n):

            # ----------------------------
            # Atualiza ST por factor
            # ----------------------------
            for k, factor in enumerate(factors):
                st = holders[k]

                up = hl2[i] + atr[i] * factor
                dn = hl2[i] - atr[i] * factor

                if i == 0:
                    st.upper = up
                    st.lower = dn
                    st.trend = 1
                else:
                    st.upper = min(up, st.upper) if closes[i-1] < st.upper else up
                    st.lower = max(dn, st.lower) if closes[i-1] > st.lower else dn

                    if closes[i] > st.upper:
                        st.trend = 1
                    elif closes[i] < st.lower:
                        st.trend = -1

                diff = np.sign(closes[i-1] - st.output) if i > 0 else 0
                st.perf += 2 / (perfAlpha + 1) * (
                    ((closes[i] - closes[i-1]) if i > 0 else 0) * diff - st.perf
                )

                st.output = st.lower if st.trend == 1 else st.upper
                st.factor = factor

            # ----------------------------
            # Clustering (LuxAlgo)
            # ----------------------------
            recent = holders if n - i <= maxData else []
            if len(recent) >= 3:
                data = np.array([h.perf for h in recent])
                facs = np.array([h.factor for h in recent])

                centroids = np.percentile(data, [25, 50, 75])

                for _ in range(maxIter):
                    clusters = {0: [], 1: [], 2: []}
                    fac_clusters = {0: [], 1: [], 2: []}

                    for v, f in zip(data, facs):
                        idx = int(np.argmin(np.abs(centroids - v)))
                        clusters[idx].append(v)
                        fac_clusters[idx].append(f)

                    new_centroids = np.array([
                        np.mean(clusters[j]) if clusters[j] else centroids[j]
                        for j in range(3)
                    ])

                    if np.allclose(new_centroids, centroids):
                        break
                    centroids = new_centroids

                from_map = {'Worst': 0, 'Average': 1, 'Best': 2}
                sel = from_map[fromCluster]

                target_vals = clusters[sel]
                target_facs = fac_clusters[sel]

                if target_facs:
                    target_factor = np.mean(target_facs)

            # ----------------------------
            # SuperTrend FINAL (o que interessa)
            # ----------------------------
            up = hl2[i] + atr[i] * target_factor
            dn = hl2[i] - atr[i] * target_factor

            if i == 0:
                st_final.upper = up
                st_final.lower = dn
                st_final.trend = 1
            else:
                st_final.upper = min(up, st_final.upper) if closes[i-1] < st_final.upper else up
                st_final.lower = max(dn, st_final.lower) if closes[i-1] > st_final.lower else dn

                if closes[i] > st_final.upper:
                    st_final.trend = 1
                elif closes[i] < st_final.lower:
                    st_final.trend = -1

            ts[i] = st_final.lower if st_final.trend == 1 else st_final.upper

            # ----------------------------
            # Perf AMA
            # ----------------------------
            perf_idx = (
                np.mean(target_vals) / perf_idx_den[i]
                if perf_idx_den[i] != 0 and len(target_vals) > 0
                else 0
            )

            if i == 0:
                perf_ama[i] = ts[i]
            else:
                perf_ama[i] = perf_ama[i-1] + perf_idx * (ts[i] - perf_ama[i-1])
                

            # ----------------------------
            # Signals (cruzamento REAL)
            # ----------------------------
            if i > 0:
                if closes[i] > st_final.upper:
                    direction = 1
                elif closes[i] < st_final.lower:
                    direction = -1
                # else: mantém

            direction_arr[i] = direction
            if i > 0:
                if closes[i-1] <= st_final.upper and closes[i] > st_final.upper:
                    signals[i] = 1
                elif closes[i-1] >= st_final.lower and closes[i] < st_final.lower:
                    signals[i] = -1
                else:
                    signals[i] = 0

            score[i] = perf_idx

            # Acumula volumes de tendência
            if direction == 1:
                up_trend_vol += volumes[i] if closes[i] > opens[i] else 0
            elif direction == -1:
                down_trend_vol += volumes[i] if closes[i] < opens[i] else 0

            # Percentual de delta de volume
            avg_vol_delta = (up_trend_vol + down_trend_vol) / 2 if (up_trend_vol + down_trend_vol) > 0 else 1e-9
            delta_vol_pct[i] = (up_trend_vol - down_trend_vol) / avg_vol_delta * 100

        retest = np.zeros(n, dtype=int)
        for i in range(1, n):
            if direction_arr[i] == 1:
                if lows[i] <= ts[i]:
                    retest[i] = 1
            elif direction_arr[i] == -1:
                if highs[i] >= ts[i]:
                    retest[i] = -1

        return Supertrend(ts.tolist(), perf_ama.tolist(), direction_arr.tolist(), score.tolist(), delta_vol_pct.tolist(), retest.tolist())

    @staticmethod
    def classify_candle(i, opens, highs, lows, closes):
        o = opens[i]
        h = highs[i]
        l = lows[i]
        c = closes[i]

        body = abs(c - o)
        range_candle = h - l
        if range_candle == 0:
            return CandleType.NEUTRAL

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        body_ratio = body / range_candle
        upper_ratio = upper_wick / range_candle
        lower_ratio = lower_wick / range_candle

        # Doji
        if body_ratio < 0.1:
            return CandleType.DOJI

        # Top exhaustion (shooting star, gravestone-like)
        if upper_ratio > 0.6 and body_ratio < 0.3:
            return CandleType.TOP_EXHAUSTION

        # Bottom exhaustion (hammer / dragonfly-like)
        if lower_ratio > 0.6 and body_ratio < 0.3:
            return CandleType.BOTTOM_EXHAUSTION

        # Normal bullish / bearish
        if c > o:
            return CandleType.BULL
        else:
            return CandleType.BEAR
        
    def two_pole_oscillator(self, length=20):
        import numpy as np

        closes = np.array(self.closes, dtype=float)
        highs = np.array(self.highs, dtype=float)
        lows = np.array(self.lows, dtype=float)

        n = len(closes)

        # -------- TRAILING SMA EXACT LIKE ta.sma(close, 25)
        sma1 = np.full(n, np.nan)
        for i in range(n):
            start = max(0, i-24)
            sma1[i] = np.mean(closes[start:i+1])

        # (close - sma1)
        diff = closes - sma1

        # -------- TRAILING SMA OF DIFF  (ta.sma(close-sma1, 25))
        sma_diff = np.full(n, np.nan)
        for i in range(n):
            start = max(0, i-24)
            sma_diff[i] = np.mean(diff[start:i+1])

        # -------- TRAILING STDEV EXACT LIKE ta.stdev(..., 25)
        stdev_diff = np.full(n, np.nan)
        for i in range(n):
            start = max(0, i-24)
            window = diff[start:i+1]
            sd = np.std(window, ddof=0)  # population stdev
            stdev_diff[i] = sd if sd != 0 else 1.0

        # -------- z-score
        sma_n1 = (diff - sma_diff) / stdev_diff

        # -------- TWO POLE FILTER WITH TRUE VAR BEHAVIOUR
        alpha = 2.0 / (length + 1.0)

        smooth1 = np.full(n, np.nan)
        smooth2 = np.full(n, np.nan)

        for i in range(n):
            if np.isnan(smooth1[i-1]) if i > 0 else True:
                smooth1[i] = sma_n1[i]
            else:
                smooth1[i] = (1 - alpha) * smooth1[i-1] + alpha * sma_n1[i]

            if np.isnan(smooth2[i-1]) if i > 0 else True:
                smooth2[i] = smooth1[i]
            else:
                smooth2[i] = (1 - alpha) * smooth2[i-1] + alpha * smooth1[i]

        two_p = smooth2

        # -------- delay 4 (two_p[4])
        two_pp = np.concatenate([two_p[:4], two_p[:-4]])

        buy = np.zeros(n, dtype=bool)
        sell = np.zeros(n, dtype=bool)

        direction = 0            # estado atual
        direction_arr = np.zeros(n, dtype=int)  # para plot
        for i in range(1, n):
            if two_p[i] > two_pp[i] and two_p[i-1] <= two_pp[i-1] and two_p[i] < 0:
                buy[i] = True
                direction = 1
            elif two_p[i] < two_pp[i] and two_p[i-1] >= two_pp[i-1] and two_p[i] > 0:
                sell[i] = True
                direction = -1

            direction_arr[i] = direction

        return two_p.tolist(), two_pp.tolist(), buy.tolist(), sell.tolist(), direction_arr.tolist()
    
    @staticmethod
    def pivothigh(high, left=3, right=3):
        n = len(high)
        ph = np.full(n, False)

        for i in range(left, n-right):
            if all(high[i] > high[i-j] for j in range(1, left+1)) and \
            all(high[i] >= high[i+j] for j in range(1, right+1)):
                ph[i] = True
        return ph

    @staticmethod
    def pivotlow(low, left=3, right=3):
        n = len(low)
        pl = np.full(n, False)

        for i in range(left, n-right):
            if all(low[i] < low[i-j] for j in range(1, left+1)) and \
            all(low[i] <= low[i+j] for j in range(1, right+1)):
                pl[i] = True
        return pl


    def volumatic_vidya(
        self,
        vidya_length=10,
        vidya_momentum=20,
        band_distance=2.0,
        atr_length=200,
    ):
        opens = np.asarray(self.opens)
        closes = np.asarray(self.closes)
        highs = np.asarray(self.highs)
        lows = np.asarray(self.lows)
        volumes = np.asarray(self.volumes)

        n = len(closes)

        # ---------------- VIDYA ----------------
        momentum = np.diff(closes, prepend=closes[0])

        pos = np.where(momentum >= 0, momentum, 0.0)
        neg = np.where(momentum < 0, -momentum, 0.0)

        sum_pos = pd.Series(pos).rolling(vidya_momentum).sum().to_numpy()
        sum_neg = pd.Series(neg).rolling(vidya_momentum).sum().to_numpy()

        cmo = 100 * (sum_pos - sum_neg) / (sum_pos + sum_neg)
        cmo = np.nan_to_num(cmo)

        alpha = 2 / (vidya_length + 1)

        vidya = np.zeros(n)
        vidya[:] = np.nan

        for i in range(1, n):
            vidya[i] = alpha * (abs(cmo[i]) / 100) * closes[i] + \
                    (1 - alpha * abs(cmo[i]) / 100) * (vidya[i-1] if not np.isnan(vidya[i-1]) else closes[i])

        # optional smoothing 15 SMA like Pinescript
        vidya_smooth = pd.Series(vidya).rolling(15).mean().to_numpy()

        # ---------------- ATR bands ----------------
        if len(closes) < atr_length:
            # fallback: usar todos os candles disponíveis
            atr_length = len(closes)
        atr_val = np.asarray(self.atr(atr_length))

        upper_band = vidya_smooth + atr_val * band_distance
        lower_band = vidya_smooth - atr_val * band_distance

        # ---------------- Trend direction ----------------
        is_trend_up = np.full(n, False)

        for i in range(1, n):
            if closes[i-1] <= upper_band[i-1] and closes[i] > upper_band[i]:
                is_trend_up[i] = True
            elif closes[i-1] >= lower_band[i-1] and closes[i] < lower_band[i]:
                is_trend_up[i] = False
            else:
                is_trend_up[i] = is_trend_up[i-1]

        # ---------------- Smoothed value ----------------
        smoothed = np.full(n, np.nan)

        for i in range(n):
            if is_trend_up[i]:
                smoothed[i] = lower_band[i]
            else:
                smoothed[i] = upper_band[i]

        # reset on change
        for i in range(1, n):
            if is_trend_up[i] != is_trend_up[i-1]:
                smoothed[i] = np.nan

        # ---------------- Pivots ----------------
        ph = IndicatorsUtils.pivothigh(highs)
        pl = IndicatorsUtils.pivotlow(lows)

        # ---------------- Volume accumulation ----------------
        up_trend_vol = np.zeros(n)
        down_trend_vol = np.zeros(n)

        u = d = 0.0
        for i in range(n):
            # reset on flip
            if i > 0 and is_trend_up[i] != is_trend_up[i-1]:
                u, d = 0.0, 0.0

            if closes[i] > self.opens[i]:
                u += volumes[i]
            elif closes[i] < self.opens[i]:
                d += volumes[i]

            up_trend_vol[i] = u
            down_trend_vol[i] = d

        avg_vol = (up_trend_vol + down_trend_vol) / 2
        delta_vol_pct = np.where(
            avg_vol != 0,
            (up_trend_vol - down_trend_vol) / avg_vol * 100,
            0.0
        )

        retest = np.zeros(n)
        for i in range(1, n):
            if is_trend_up[i]:
                if lows[i] <= lower_band[i]:
                    retest[i] = 1
            elif not is_trend_up[i]:
                # Retest próximo da upper_band
                if highs[i] >= upper_band[i]:
                    retest[i] = -1

        return VolumaticVidya(
            vidya_smooth.tolist(),
            upper_band.tolist(),
            lower_band.tolist(),
            is_trend_up.tolist(),
            smoothed.tolist(),
            ph.tolist(),
            pl.tolist(),
            up_trend_vol.tolist(),
            down_trend_vol.tolist(),
            delta_vol_pct.tolist(),
            retest.tolist()
            )
    
    def squeeze_index(self, length=20, conv=50):
        """
        LuxAlgo Squeeze Index (PSI)
        Parameters
        ----------
        close : np.ndarray
            closing prices
        length : int
            correlation window
        conv : int
            convergence factor (LuxAlgo default = 50)

        Returns
        -------
        psi : np.ndarray
        """

        closes = np.asarray(self.closes, dtype=float)
        n = len(closes)

        # adaptive envelopes
        max_env = np.zeros(n)
        min_env = np.zeros(n)

        max_env[0] = closes[0]
        min_env[0] = closes[0]

        for i in range(1, n):
            # TradingView:
            # max := max(prev_max - (prev_max - src)/conv, src)
            # min := min(prev_min + (src - prev_min)/conv, src)
            max_candidate = max_env[i-1] - (max_env[i-1] - closes[i]) / conv
            min_candidate = min_env[i-1] + (closes[i] - min_env[i-1]) / conv

            max_env[i] = max(max_candidate, closes[i])
            min_env[i] = min(min_candidate, closes[i])

        # range width
        diff = np.log(np.clip(max_env - min_env, 1e-12, None))

        # rolling correlation with time index
        idx = np.arange(n, dtype=float)

        psi = np.full(n, np.nan)

        for i in range(length - 1, n):
            x = diff[i-length+1:i+1]
            t = idx[i-length+1:i+1]

            x_mean = x.mean()
            t_mean = t.mean()

            cov = np.sum((x - x_mean) * (t - t_mean))
            var_t = np.sum((t - t_mean) ** 2)

            if var_t == 0:
                continue

            corr = cov / np.sqrt(var_t * np.sum((x - x_mean) ** 2) + 1e-12)

            psi[i] = -50 * corr + 50

        return psi
        

