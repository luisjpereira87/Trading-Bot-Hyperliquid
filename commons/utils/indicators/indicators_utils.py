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
            from ta.trend import (ADXIndicator, EMAIndicator, PSARIndicator,
                                  SMAIndicator)
            from ta.volatility import AverageTrueRange

            
            self.EMAIndicator = EMAIndicator
            self.ADXIndicator = ADXIndicator
            self.RSIIndicator = RSIIndicator
            self.StochasticOscillator = StochasticOscillator
            self.AverageTrueRange = AverageTrueRange
            self.PSARIndicator = PSARIndicator
            self.SMAIndicator = SMAIndicator

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
        
    def sma(self, period=21):
        if self.mode == 'custom':
            sma = []
            for i in range(len(self.closes)):
                if i < period - 1:
                    sma.append(0)
                else:
                    sma.append(np.mean(self.closes[i][i - period + 1:i + 1]))
            return sma
        else:
            sma_series = self.SMAIndicator(close=self.df['close'], window=period).sma_indicator()
            
            return sma_series.tolist()
        
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
        
    def atr_wilder(self, length=14):
        """
        Calcula o ATR usando o método Wilder (suavização EMA estilo Wilder)
        
        highs, lows, closes: arrays ou pd.Series
        length: período do ATR
        
        Retorna: ATR como array
        """
        highs = pd.Series(self.highs)
        lows = pd.Series(self.lows)
        closes = pd.Series(self.closes)

        # True Range
        tr = pd.concat([
            highs - lows,
            (highs - closes.shift(1)).abs(),
            (lows - closes.shift(1)).abs()
        ], axis=1).max(axis=1)

        # ATR Wilder: EMA estilo Wilder (α = 1 / length)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()

        return atr.to_numpy()

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
    

    def classify_candles(self):
        n = len(self.closes)
        # Inicializa o array com NEUTRAL
        candles_values = [CandleType.NEUTRAL] * n

        for i in range(0, n):
            range_candle = self.highs[i] - self.lows[i]
            if range_candle == 0:
                continue

            body = abs(self.closes[i] - self.opens[i])
            upper_wick = self.highs[i] - max(self.opens[i], self.closes[i])
            lower_wick = min(self.opens[i], self.closes[i]) - self.lows[i]

            body_ratio = body / range_candle
            upper_ratio = upper_wick / range_candle
            lower_ratio = lower_wick / range_candle

            # --- LÓGICA BULL (FECHO > ABERTURA) ---
            if self.closes[i] > self.opens[i]:
                if upper_ratio > 0.6 and body_ratio < 0.3:
                    candles_values[i] = CandleType.TOP_EXHAUSTION
                elif upper_wick > body: # Pavio superior maior que o corpo
                    candles_values[i] = CandleType.WEAK_BULL
                elif body_ratio > 0.6: # Pouco pavio, muito corpo
                    candles_values[i] = CandleType.STRONG_BULL
                else:
                    candles_values[i] = CandleType.BULL

            # --- LÓGICA BEAR (FECHO < ABERTURA) ---
            elif self.closes[i] < self.opens[i]:
                if lower_ratio > 0.6 and body_ratio < 0.3:
                    candles_values[i] = CandleType.BOTTOM_EXHAUSTION
                elif lower_wick > body: # Pavio inferior maior que o corpo
                    candles_values[i] = CandleType.WEAK_BEAR
                elif body_ratio > 0.6:
                    candles_values[i] = CandleType.STRONG_BEAR
                else:
                    candles_values[i] = CandleType.BEAR
            
            # --- LÓGICA DOJI (FECHO == ABERTURA OU MUITO PRÓXIMO) ---
            else:
                candles_values[i] = CandleType.DOJI

        return candles_values
        


        

