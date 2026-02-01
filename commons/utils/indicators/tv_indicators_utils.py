import math
from collections import deque

import numpy as np
import pandas as pd

from commons.enums.signal_enum import Signal
from commons.models.supertrend_dclass import Supertrend
from commons.models.volumatic_vidya_dclass import VolumaticVidya
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper


class TvIndicatorsUtils(IndicatorsUtils):
    def __init__(self, ohlcv: OhlcvWrapper, mode='ta'):
        super().__init__(ohlcv, mode)

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
    
    def two_pole_oscillator(self, length=20):

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

        def calculate_rma(series, length):
            """Média Móvel de Wilder (RMA) usada no ATR do TradingView"""
            alpha = 1.0 / length
            rma = np.zeros_like(series)
            # Inicializa com a média simples (SMA) para o primeiro ponto válido
            rma[length-1] = np.mean(series[:length])
            for i in range(length, len(series)):
                rma[i] = (series[i] * alpha) + (rma[i-1] * (1 - alpha))
            return rma

        def get_pine_atr(highs, lows, closes, length=200):
            """ATR idêntico ao ta.atr(200) do Pine Script"""
            tr1 = highs - lows
            tr2 = np.abs(highs - np.roll(closes, 1))
            tr3 = np.abs(lows - np.roll(closes, 1))
            # True Range
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            tr[0] = tr1[0] # Primeiro candle não tem close anterior
            return calculate_rma(tr, length)

        # 1. ---- VIDYA Core Calculation ----
        momentum = pd.Series(closes).diff()
        
        # Separar momentum positivo e negativo (idêntico ao Pine)
        pos = momentum.where(momentum >= 0, 0.0).rolling(vidya_momentum).sum()
        neg = momentum.where(momentum < 0, 0.0).abs().rolling(vidya_momentum).sum()

        # Evitar divisão por zero no CMO
        denom = pos + neg
        abs_cmo = (100 * (pos - neg) / denom).abs().fillna(0) / 100
        
        alpha = 2 / (vidya_length + 1)
        vidya = np.zeros(n)
        
        # Inicialização fiel ao nz(vidya[1])
        # Usamos o primeiro close disponível para evitar que a linha comece em zero
        vidya[0] = closes[0]

        for i in range(1, n):
            # vidya := alpha * abs_cmo * src + (1 - alpha * abs_cmo) * nz(vidya[1])
            k = alpha * abs_cmo.iloc[i]
            vidya[i] = (k * closes[i]) + (1 - k) * vidya[i-1]

        # Suavização Final SMA 15 (como no script do BigBeluga)
        vidya_smooth = pd.Series(vidya).rolling(15).mean().bfill().to_numpy()

        # 2. ---- ATR Suavizado (RMA/Wilder) ----
        # Importante: O ATR do Pine usa RMA. Garante que a tua self.atr(200) usa RMA.
        atr_val = get_pine_atr(highs, lows, closes, atr_length)
        
        upper_band = vidya_smooth + (atr_val * band_distance)
        lower_band = vidya_smooth - (atr_val * band_distance)

        # 3. ---- Trend & Smoothed Line ----
        is_trend_up = np.full(n, False)
        for i in range(1, n):
            if closes[i] > upper_band[i]:
                is_trend_up[i] = True
            elif closes[i] < lower_band[i]:
                is_trend_up[i] = False
            else:
                is_trend_up[i] = is_trend_up[i-1]

        smoothed = np.full(n, np.nan)
        for i in range(1, n):
            # O Pine Script esconde a linha no momento do flip
            if is_trend_up[i] == is_trend_up[i-1]:
                smoothed[i] = lower_band[i] if is_trend_up[i] else upper_band[i]

        # 4. ---- Pivots & Volume (Idêntico à tua lógica, mas otimizado) ----
        ph = TvIndicatorsUtils.pivothigh(highs)
        pl = TvIndicatorsUtils.pivotlow(lows)

        up_trend_vol = np.zeros(n)
        down_trend_vol = np.zeros(n)
        
        u = 0.0
        d = 0.0
        
        for i in range(1, n):
            # No Pine: if ta.change(trend_cross_up) or ta.change(trend_cross_down)
            # Como trend_cross_up/down já são sinais de mudança de 1 barra:
            trend_flip = (is_trend_up[i] != is_trend_up[i-1])

            if trend_flip:
                u = 0.0
                d = 0.0
            
            # IMPORTANTE: No Pine, após o reset, ele verifica se acumula 
            # na mesma barra ou se a condição "not(ta.change...)" impede.
            # O BigBeluga usa: if not(trend_flip) -> acumula.
            if not trend_flip:
                if closes[i] > opens[i]:
                    u += float(volumes[i])
                elif closes[i] < opens[i]:
                    d += float(volumes[i])
            
            # Se for o candle do flip, u e d chegam aqui como 0.0
            up_trend_vol[i] = u
            down_trend_vol[i] = d

        # --- O GRANDE FILTRO ---
        # No Pine, o volume é 'na' enquanto o VIDYA não estabilizar.
        # Vamos zerar o Delta onde o VIDYA (SMA 15) ainda é NaN para bater com o TV.
        vidya_mask = ~np.isnan(vidya_smooth)
        
        avg_vol = (up_trend_vol + down_trend_vol) / 2
        delta_vol_pct = np.zeros(n)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # Só calcula onde a média de volume existe E o VIDYA está pronto
            mask = (avg_vol > 0) & vidya_mask
            delta_vol_pct[mask] = ((up_trend_vol[mask] - down_trend_vol[mask]) / avg_vol[mask]) * 100
            

        # 5. ---- Retest Logic ----
        retest = np.zeros(n)
        for i in range(1, n):
            if is_trend_up[i] and lows[i] <= lower_band[i]:
                retest[i] = 1 # Bullish Retest
            elif not is_trend_up[i] and highs[i] >= upper_band[i]:
                retest[i] = -1 # Bearish Retest

        return VolumaticVidya(
            vidya=vidya_smooth.tolist(),
            upper_band=upper_band.tolist(),
            lower_band=lower_band.tolist(),
            is_trend_up=is_trend_up.tolist(),
            smoothed=smoothed.tolist(),
            pivot_high=ph.tolist(),
            pivot_low=pl.tolist(),
            up_trend_volume=up_trend_vol.tolist(),
            down_trend_volume=down_trend_vol.tolist(),
            delta_volume_pct=delta_vol_pct.tolist(),
            retest=retest.tolist()
        )
    
    def smi(self, length_k=10, length_d=3, length_ema=3):
        """
        Calcula o Stochastic Momentum Index (SMI)
        
        Args:
            close (np.array ou list): preços de fechamento
            high (np.array ou list): máximos
            low (np.array ou list): mínimos
            length (int): período do cálculo
            smoothK (int): suavização do %K
            smoothD (int): suavização do %D
        Returns:
            smi (np.array): valor do SMI
            smi_signal (np.array): sinal suavizado do SMI (%D)
        """

        high = pd.Series(self.highs)
        low = pd.Series(self.lows)
        close = pd.Series(self.closes)
        n = len(close)
        direction = np.zeros(n)

        hh = high.rolling(length_k).max()
        ll = low.rolling(length_k).min()

        rel_range = close - (hh + ll) / 2
        range_hl = hh - ll

        def ema_ema(series, length):
            ema1 = series.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            return ema2

        smi_val = 200 * (ema_ema(rel_range, length_d) / ema_ema(range_hl, length_d))
        smi_signal = smi_val.ewm(span=length_ema, adjust=False).mean()

        for i in range(n):
            if smi_val[i] > smi_signal[i]:
                direction[i] = 1
            elif smi_val[i] < smi_signal[i]:
                direction[i] = -1

        return smi_val.values, smi_signal.values, direction

    
    def regime_filter(self, length=20, hma_len=15):
        # --- Funções Internas (Estilo Pine Script) ---
        def pine_wma(src, p):
            weights = np.arange(1, p + 1)
            return src.rolling(window=p).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

        def pine_hma(src, p):
            half_len = int(p / 2)
            sqrt_len = int(np.sqrt(p))
            wma_half = pine_wma(src, half_len)
            wma_full = pine_wma(src, p)
            diff = 2 * wma_half - wma_full
            return pine_wma(diff, sqrt_len)

        # --- Preparação de Dados ---
        h = np.array(self.highs, dtype='float64')
        l = np.array(self.lows, dtype='float64')
        c = np.array(self.closes, dtype='float64')
        v = np.array(self.volumes, dtype='float64')
        n = len(c)

        # 1. Delta Estimado (Filtro de Microestrutura para Futuros)
        # Evita que volume lateral na Hyperliquid gere sinais falsos
        body_range = np.abs(c - l) - np.abs(h - c)
        total_range = h - l + 0.000001
        delta = (body_range / total_range) * v
        cvd = pd.Series(delta).cumsum()

        # 2. Eficiência Fractal (Mede a "limpeza" da tendência)
        net_chg = abs(pd.Series(c).diff(length))
        sum_chg = pd.Series(abs(pd.Series(c).diff())).rolling(length).sum()
        efficiency = np.array((net_chg / (sum_chg + 0.000001)).fillna(0).values)

        # 3. Médias Móveis Hull (Preço e Volume Delta)
        hlc3 = pd.Series((h + l + c) / 3.0)
        hma_p = pine_hma(hlc3, hma_len).values
        hma_v = pine_hma(cvd, hma_len).values

        # --- Inicialização de Arrays de Saída ---
        trend = np.zeros(n)
        voltrend = np.zeros(n)
        coeff = 10.0 / length

        # 4. Loop de Regime (Cálculo de Força)
        # Começamos após o warm-up das médias
        start_idx = length + hma_len + int(np.sqrt(hma_len))
        
        for i in range(start_idx, n):
            t_sum, v_sum = 0, 0
            for j in range(length + 1):
                # Comparação de Preço
                if hma_p[i] > hma_p[i-j]: t_sum += 1
                else: t_sum -= 1
                
                # Comparação de Fluxo (Delta)
                if hma_v[i] > hma_v[i-j]: v_sum += 1
                else: v_sum -= 1
            
            trend[i] = t_sum * coeff
            voltrend[i] = v_sum * coeff

        # 5. Score de Confirmação (0-100)
        # Alinhamento: Trend e Voltrend no mesmo sinal?
        alignment = (trend * voltrend) > 0 
        
        # Força Bruta: Média das magnitudes
        raw_strength = (np.abs(trend) + np.abs(voltrend)) / 20.0
        
        # Cálculo do Score final ponderando Eficiência
        # Se alignment for False, o score é drasticamente reduzido (divergência)
        score = (raw_strength * 0.4 + efficiency * 0.6) * 100
        confirm_score = np.where(alignment, score, score * 0.2)
        confirm_score = np.clip(confirm_score, 0, 100)

        return {
            'trend': trend,
            'voltrend': voltrend,
            'efficiency': efficiency,
            'score': confirm_score
        }
    
    def find_swings(self, left=3, right=3):
        highs = np.array(self.highs)
        lows = np.array(self.lows)
        n = len(self.highs)
        swing_high = np.full(n, np.nan)
        swing_low = np.full(n, np.nan)

        for i in range(left, n - right):
            if highs[i] == max(highs[i-left:i+right+1]):
                swing_high[i] = highs[i]
            if lows[i] == min(lows[i-left:i+right+1]):
                swing_low[i] = lows[i]

        return swing_high, swing_low
    
    def market_structure(self):
        highs = np.array(self.highs)
        lows = np.array(self.lows)
        swing_high, swing_low = self.find_swings()

        last_HL = np.full(len(highs), np.nan)
        last_LH = np.full(len(highs), np.nan)

        current_HL = np.nan
        current_LH = np.nan

        for i in range(len(highs)):
            if not np.isnan(swing_low[i]):
                current_HL = swing_low[i]
            if not np.isnan(swing_high[i]):
                current_LH = swing_high[i]

            last_HL[i] = current_HL
            last_LH[i] = current_LH

        return last_HL, last_LH

    def market_structure_rsi(self):
        high = np.array(self.highs)
        low = np.array(self.lows)
        closes = np.array(self.closes)
        n = len(closes)

        direction = np.zeros(n)

        last_HL, last_LH = self.market_structure()

        for i in range(1, n):

            # Reversão para baixo (saída de LONG / entrada SHORT)
            if closes[i] < last_HL[i] and closes[i-1] >= last_HL[i-1]:
                direction[i] = -1

            # Reversão para cima (saída de SHORT / entrada LONG)
            elif closes[i] > last_LH[i] and closes[i-1] <= last_LH[i-1]:
                direction[i] = 1

        return direction
    

    def smart_money_flow_cloud(
        self,
        len_=34,
        basisType="EMA",
        almaOffset=0.85,
        almaSigma=6.0,
        basisSmooth=3,
        mfLen=24,
        mfSmooth=5,
        mfPower=1.2,
        atrLen=14,
        minMult=0.9,
        maxMult=2.2,
        dotCooldown=12  # Cooldown para retests
    ):
        n = len(self.closes)
        highs = self.highs
        lows = self.lows
        closes = self.closes
        volumes = self.volumes
        opens = self.opens

        # --- Output arrays ---
        bsO = np.full(n, np.nan)
        bsC = np.full(n, np.nan)
        bsMain = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        mfSm = np.full(n, np.nan)
        strength = np.full(n, np.nan)
        mult = np.full(n, np.nan)
        lastSignal = np.zeros(n, dtype=int)  # 1 = long, -1 = short
        bull_retest = np.full(n, False)
        bear_retest = np.full(n, False)

        # --- Helper: EMA recursiva ---
        def ema_step(prev, x, length):
            alpha = 2 / (length + 1)
            return alpha * x + (1 - alpha) * prev if not np.isnan(prev) else x

        # --- Helper: ALMA ---
        def alma(arr, idx, length, offset, sigma):
            if idx + 1 < length:
                return np.nan
            m = offset * (length - 1)
            s = length / sigma
            w = np.array([math.exp(-((i - m) ** 2) / (2 * s ** 2)) for i in range(length)])
            norm = np.sum(w)
            vals = arr[idx - length + 1 : idx + 1]
            return np.sum(vals * w) / norm

        # --- Helper: ATR Wilder ---
        atr_vals = np.full(n, np.nan)
        tr_prev = 0.0
        for i in range(n):
            if i == 0:
                tr = highs[i] - lows[i]
                atr_vals[i] = tr
                tr_prev = tr
            else:
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                atr_vals[i] = (tr_prev * (atrLen - 1) + tr) / atrLen
                tr_prev = atr_vals[i]

        # --- Buffers para money flow ---
        raw_buffer = deque(maxlen=mfLen)
        abs_buffer = deque(maxlen=mfLen)
        mf_prev_ema = np.nan

        # --- States para signals ---
        prev_signal = 0
        prev_signal2 = 0

        # --- EMA buffers para Basis smoothing ---
        ema_basis_o_prev = np.nan
        ema_basis_c_prev = np.nan

        # --- Estado para retests ---
        lastBearDotBar = -np.inf
        lastBullDotBar = -np.inf

        for i in range(n):
            # --- Calc MF ---
            clv = 0.0 if highs[i] == lows[i] else ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i])
            raw = clv * volumes[i]
            raw_buffer.append(raw)
            abs_buffer.append(abs(raw))
            num = sum(raw_buffer)
            den = sum(abs_buffer)
            mf = 0.0 if den == 0.0 else num / den

            # MF smoothed
            if mfSmooth > 1:
                mf_prev_ema = ema_step(mf_prev_ema, mf, mfSmooth)
                mfSm[i] = mf_prev_ema
            else:
                mfSm[i] = mf

            # MF strength & mult
            s = min(max(abs(mfSm[i]) ** mfPower, 0.0), 1.0)
            strength[i] = s
            mult[i] = minMult + (maxMult - minMult) * s

            # --- Calc Basis ---
            if basisType == "ALMA":
                bO_raw = alma(opens, i, len_, almaOffset, almaSigma)
                bC_raw = alma(closes, i, len_, almaOffset, almaSigma)
            else:  # EMA
                bO_raw = ema_step(ema_basis_o_prev, opens[i], len_)
                ema_basis_o_prev = bO_raw
                bC_raw = ema_step(ema_basis_c_prev, closes[i], len_)
                ema_basis_c_prev = bC_raw

            # Basis smoothing
            if basisSmooth > 1:
                if i == 0:
                    bsO[i] = bO_raw
                    bsC[i] = bC_raw
                else:
                    bsO[i] = ema_step(bsO[i-1], bO_raw, basisSmooth)
                    bsC[i] = ema_step(bsC[i-1], bC_raw, basisSmooth)
            else:
                bsO[i] = bO_raw
                bsC[i] = bC_raw

            bsMain[i] = bsC[i]

            # --- Bands ---
            upper[i] = bsMain[i] + atr_vals[i] * mult[i]
            lower[i] = bsMain[i] - atr_vals[i] * mult[i]

            # --- Signals ---
            longCond = i > 0 and closes[i-1] <= upper[i-1] and closes[i] > upper[i]
            shortCond = i > 0 and closes[i-1] >= lower[i-1] and closes[i] < lower[i]

            prev_signal2 = prev_signal
            if longCond:
                lastSignal[i] = 1
            elif shortCond:
                lastSignal[i] = -1
            else:
                lastSignal[i] = prev_signal if i > 0 else (1 if closes[i] >= bsMain[i] else -1)
            prev_signal = lastSignal[i]

            # --- Retests (dots) ---
            bullDotCond = lastSignal[i] == 1 and lows[i] < bsC[i]
            bearDotCond = lastSignal[i] == -1 and highs[i] > bsC[i]

            bullOk = bullDotCond and (dotCooldown == 0 or (i - lastBullDotBar) >= dotCooldown)
            bearOk = bearDotCond and (dotCooldown == 0 or (i - lastBearDotBar) >= dotCooldown)

            if bullOk:
                bull_retest[i] = True
                lastBullDotBar = i
            if bearOk:
                bear_retest[i] = True
                lastBearDotBar = i

        return {
            "basis": bsMain,
            "basis_open": bsO,
            "basis_close": bsC,
            "upper": upper,
            "lower": lower,
            "mf": mfSm,
            "strength": strength,
            "mult": mult,
            "signal": lastSignal,
            "bull_retest": bull_retest,
            "bear_retest": bear_retest
        }
    
    @staticmethod
    def crossover(a, b):
        # Shift para comparar o estado anterior
        a_prev = np.roll(a, 1)
        b_prev = np.roll(b, 1)
        # O primeiro elemento fica inválido após o roll
        mask = (a > b) & (a_prev <= b_prev)
        mask[0] = False
        return mask

    @staticmethod
    def barssince(condition):
        out = np.zeros(len(condition), dtype=int)
        count = 999999 # Valor alto inicial (equivalente ao nz no Pine)
        for i, val in enumerate(condition):
            if val:
                count = 0
            else:
                count += 1
            out[i] = count
        return out

    @staticmethod
    def highestbars(series, length):
        n = len(series)
        out = np.zeros(n)
        for i in range(n):
            start = max(0, i - length + 1)
            window = series[start:i+1]
            if len(window) == 0:
                out[i] = 0
                continue
            # PineScript: 0 é a barra atual, -1 é a anterior. 
            # argmax do invertido dá 0 para a mais recente, logo basta tornar negativo.
            out[i] = -np.argmax(window[::-1])
        return out

    @staticmethod
    def lowestbars(series, length):
        n = len(series)
        out = np.zeros(n)
        for i in range(n):
            start = max(0, i - length + 1)
            window = series[start:i+1]
            if len(window) == 0:
                out[i] = 0
                continue
            out[i] = -np.argmin(window[::-1])
        return out

    def smart_money_breakout_channels(self, length_norm=100, length_box=14, strong=True, overlap=False):
        highs, lows, closes, opens = np.array(self.highs), np.array(self.lows), np.array(self.closes), np.array(self.opens)
        n = len(closes)

        # --- 1. Cálculos de Normalização ---
        s_lows = pd.Series(lows)
        s_highs = pd.Series(highs)
        
        lowestLow = np.array(s_lows.rolling(length_norm, min_periods=1).min().values)
        highestHigh = np.array(s_highs.rolling(length_norm, min_periods=1).max().values)
        
        # Evitar divisão por zero
        denom = highestHigh - lowestLow
        denom[denom == 0] = 1e-9
        normalizedPrice = (closes - lowestLow) / denom
        
        # Volatilidade (Pine: ta.stdev(normalizedPrice, 14))
        vol = pd.Series(normalizedPrice).rolling(14, min_periods=1).std().values

        # --- 2. Detecção de Canais (Upper / Lower) ---
        hb = self.highestbars(vol, length_box + 1)
        lb = self.lowestbars(vol, length_box + 1)
        
        upper = (hb + length_box) / length_box
        lower = (lb + length_box) / length_box

        # --- 3. Duration e h/l dinâmicos ---
        # Pine: duration = math.max(nz(ta.barssince(ta.crossover(lower,upper))), 1)
        cross_low_up = self.crossover(lower, upper)
        duration = self.barssince(cross_low_up)
        duration = np.maximum(duration, 1)

        h_val = np.zeros(n)
        l_val = np.zeros(n)
        for i in range(n):
            d = int(duration[i])
            if d > 0:
                # d+1 para garantir que pegamos a barra onde começou a acumulação
                start = max(0, i - d) 
                h_val[i] = np.max(highs[start:i+1])
                l_val[i] = np.min(lows[start:i+1])

        # --- 4. Loop de Estado para Boxes Reais ---
        # Representamos as boxes ativas como uma lista de dicts
        active_boxes = [] 
        
        bull_break = np.zeros(n, dtype=bool)
        bear_break = np.zeros(n, dtype=bool)
        new_channel = np.zeros(n, dtype=bool)
        
        # Armazenar o estado das boxes para retorno (opcional)
        top_plot = np.full(n, np.nan)
        bot_plot = np.full(n, np.nan)

        cross_up_low = self.crossover(upper, lower)
        price_exec = (opens + closes) / 2 if strong else closes

        for i in range(n):
            # Tentar criar novo canal
            if cross_up_low[i] and duration[i] > 10:
                tNew, bNew = h_val[i], l_val[i]
                
                # Lógica f_can_create
                can_create = True
                if not overlap and len(active_boxes) > 0:
                    for box in active_boxes:
                        if (tNew > box['bottom']) and (bNew < box['top']):
                            can_create = False
                            break
                
                if can_create:
                    active_boxes.append({'top': tNew, 'bottom': bNew})
                    new_channel[i] = True

            # Verificar breakouts em todas as boxes ativas
            boxes_to_remove = []
            for idx, box in enumerate(active_boxes):
                if price_exec[i] > box['top']:
                    bull_break[i] = True
                    boxes_to_remove.append(idx)
                elif price_exec[i] < box['bottom']:
                    bear_break[i] = True
                    boxes_to_remove.append(idx)
            
            # Remover boxes que quebraram (do fim para o início para não quebrar o index)
            for idx in reversed(boxes_to_remove):
                active_boxes.pop(idx)

            # Para visualização, pegamos a box mais recente (se existir)
            if active_boxes:
                top_plot[i] = active_boxes[-1]['top']
                bot_plot[i] = active_boxes[-1]['bottom']

        return {
            'top': top_plot,
            'bottom': bot_plot,
            'bull_break': bull_break,
            'bear_break': bear_break,
            'new_channel': new_channel,
            'duration': duration
        }
    
    def directional_imbalance_index(self, length=10, period=70):
        highs = pd.Series(self.highs)
        lows = pd.Series(self.lows)

        # 1. Calcula o teto e o chão (equivalente ao ta.highest/lowest)
        upper = highs.rolling(window=length + 1).max()
        lower = lows.rolling(window=length + 1).min()

        # 2. Marca onde houve toque (1 se verdade, 0 se falso)
        # Usamos np.isclose para evitar erros de precisão de float
        up_hits = (highs == upper).astype(int)
        down_hits = (lows == lower).astype(int)

        # 3. Em vez de um loop de 70, usamos uma soma móvel (Rolling Sum)
        # Isto faz exatamente o que o teu 'for j in range(period)' fazia
        up_count = up_hits.rolling(window=period).sum()
        down_count = down_hits.rolling(window=period).sum()

        # 4. Cálculo das percentagens
        total = up_count + down_count
        
        # Evita divisão por zero de forma elegante
        bulls_perc = np.where(total > 0, (up_count / total) * 100, 50)
        bears_perc = np.where(total > 0, (down_count / total) * 100, 50)

        return {
            "upper": upper,
            "lower": lower,
            "up_count": up_count,
            "down_count": down_count,
            "bulls_perc": bulls_perc,
            "bears_perc": bears_perc
        }
    
    def adaptive_rsi_boswaves(self, rsi_len=18, smooth_len=20, adapt_lookback=1000):
        # 1. Preparação da Source (HMA 4 aplicada ao Close conforme o script)
        def pine_hma(src, p):
            def pine_wma(s, window):
                weights = np.arange(1, window + 1)
                return s.rolling(window=window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
            
            half_len = int(p / 2)
            sqrt_len = int(np.sqrt(p))
            diff = 2 * pine_wma(src, half_len) - pine_wma(src, p)
            return pine_wma(diff, sqrt_len)

        # Fonte base suavizada
        src_series = pd.Series(self.closes)
        base_src = pine_hma(src_series, 4)

        # 2. RSI Calculation
        def calculate_rsi(series, period):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        osc_raw = calculate_rsi(base_src, rsi_len)

        # 3. Smoothing (SMA por padrão no BOSWaves)
        osc = osc_raw.rolling(window=smooth_len).mean()

        # 4. Adaptive Thresholds (Percentis dinâmicos)
        # Usamos o rolling percentile para criar as bandas que "respiram" com o mercado
        upper_thr = osc.rolling(window=adapt_lookback).apply(lambda x: np.nanpercentile(x, 50), raw=True)
        lower_thr = osc.rolling(window=adapt_lookback).apply(lambda x: np.nanpercentile(x, 45), raw=True)

        # 5. Regime State (1 = Bull, -1 = Bear, 0 = Neutral)
        n = len(osc)
        regime = np.zeros(n)
        
        # Simulação da lógica de "isconfirmed" e retenção de estado do Pine
        for i in range(1, n):
            if osc[i] > upper_thr[i]:
                regime[i] = 1
            elif osc[i] < lower_thr[i]:
                regime[i] = -1
            else:
                regime[i] = regime[i-1] # Mantém o estado anterior (Hysteresis)

        return {
            'osc': osc.values,
            'upper_thr': upper_thr.values,
            'lower_thr': lower_thr.values,
            'regime': regime
        }
    
    def get_pivots(self, series, left=15, right=15):
        """Encontra picos (pivots) numa série temporal"""
        pivots = []
        for i in range(left, len(series) - right):
            is_high = True
            is_low = True
            for j in range(i - left, i + right + 1):
                if i == j: continue
                if series[j] >= series[i]: is_high = False
                if series[j] <= series[i]: is_low = False
            
            if is_high: pivots.append((i, 'high', series[i]))
            if is_low: pivots.append((i, 'low', series[i]))
        return pivots

    def adaptive_rsi_with_div(self, rsi_len=18, smooth_len=20, div_lookback=60):
        # 1. Obter o RSI Adaptativo (Usando a função anterior)
        data = self.adaptive_rsi_boswaves(rsi_len, smooth_len, 100)
        osc = data['osc']
        prices = np.array(self.closes)
        
        bull_div = np.zeros(len(osc))
        bear_div = np.zeros(len(osc))
        
        # 2. Localizar Pivots (usando janelas de 15 barras como no Pine)
        pivots = self.get_pivots(osc, left=15, right=15)
        
        # 3. Lógica de Divergência Regular
        # Bullish: Preço faz Low mais baixo, RSI faz Low mais alto
        # Bearish: Preço faz High mais alto, RSI faz High mais baixo
        for i in range(len(pivots) - 1):
            idx2, type2, val2 = pivots[i+1]
            idx1, type1, val1 = pivots[i]
            
            # Distância entre pivots deve estar no range (ex: 5 a 60 barras)
            if 5 <= (idx2 - idx1) <= div_lookback:
                if type1 == 'low' and type2 == 'low':
                    if prices[idx2] < prices[idx1] and val2 > val1:
                        bull_div[idx2] = 1 # Sinal confirmado no índice do pivot
                
                if type1 == 'high' and type2 == 'high':
                    if prices[idx2] > prices[idx1] and val2 < val1:
                        bear_div[idx2] = 1
                        
        data['bull_div'] = bull_div
        data['bear_div'] = bear_div
        return data

    def smoothed_heikin_ashi(self, len1=10, len2=10):
        # Converter para Pandas Series para usar o .ewm()
        closes = pd.Series(self.closes)
        opens = pd.Series(self.opens)
        highs = pd.Series(self.highs)
        lows = pd.Series(self.lows)

        # 1. Suavização Inicial (O que faltava!)
        # No Pine: o=ema(open,len), c=ema(close,len)...
        o_smooth = opens.ewm(span=len1, adjust=False).mean()
        c_smooth = closes.ewm(span=len1, adjust=False).mean()
        h_smooth = highs.ewm(span=len1, adjust=False).mean()
        l_smooth = lows.ewm(span=len1, adjust=False).mean()

        # 2. Cálculo Heikin Ashi (Usando os valores suavizados)
        haclose = (o_smooth + h_smooth + l_smooth + c_smooth) / 4
        
        haopen = np.zeros_like(haclose)
        # Valor inicial baseado nos suavizados
        haopen[0] = (o_smooth[0] + c_smooth[0]) / 2
        
        for i in range(1, len(haclose)):
            haopen[i] = (haopen[i-1] + haclose[i-1]) / 2
        
        # Importante: usar h_smooth e l_smooth aqui para manter a suavização
        hahigh = np.maximum(h_smooth, np.maximum(haopen, haclose))
        halow = np.minimum(l_smooth, np.minimum(haopen, haclose))

        # 3. Suavização Final (EMA das HA calculadas)
        o2 = pd.Series(haopen).ewm(span=len2, adjust=False).mean()
        c2 = pd.Series(haclose).ewm(span=len2, adjust=False).mean()
        h2 = pd.Series(hahigh).ewm(span=len2, adjust=False).mean()
        l2 = pd.Series(halow).ewm(span=len2, adjust=False).mean()

        # Tendência: 1 para Alta, -1 para Baixa
        trend = np.where(c2 > o2, 1, -1)
        
        return trend, o2, h2, l2, c2
    
    def standardized_macd_ha(self, fast=12, slow=26, sig_len=9):
        # 1. Cálculo do MACD Standardized
        ema_fast = pd.Series(self.closes).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(self.closes).ewm(span=slow, adjust=False).mean()
        atr_simulado = (pd.Series(self.highs) - pd.Series(self.lows)).ewm(span=slow, adjust=False).mean()
        
        # Esta é a fórmula mágica do indicador
        st_macd = (ema_fast - ema_slow) / atr_simulado * 100
        
        # 2. Transformação Heikin-Ashi do Oscilador
        # Criamos as velas HA baseadas no valor do st_macd
        o_macd = st_macd.shift(1)
        h_macd = np.maximum(st_macd, o_macd)
        l_macd = np.minimum(st_macd, o_macd)
        c_macd = st_macd
        
        # HA Recursivo para o Oscilador
        ha_c = (o_macd + h_macd + l_macd + c_macd) / 4
        ha_o = np.zeros_like(ha_c)
        ha_o[0] = (o_macd[0] + c_macd[0]) / 2
        for i in range(1, len(ha_c)):
            ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2
        
        # 3. Linha de Sinal (EMA do Close do HA)
        signal = pd.Series(ha_c).ewm(span=sig_len, adjust=False).mean()
        histogram = ha_c - signal
        
        return ha_o, ha_c, signal, histogram
    
    def regression_slope_oscillator(self, min_range=10, max_range=100, step=5, sig_line=7):
        """
        Calcula o Regression Slope Oscillator e retorna os gatilhos de cruzamento.
        
        Retorna: 
        - oscillator_values: List[float]
        - signal_line: List[float]
        - crossover_signals: List[int] (1: Reversal Up, -1: Reversal Down, 0: Nada)
        """
        closes = np.array(self.closes, dtype=float)
        n = len(closes)
        oscillator_values = np.zeros(n)
        
        # 1. Cálculo das Inclinações (Slopes)
        for i in range(max_range, n):
            slopes_at_i = []
            for length in range(min_range, max_range + 1, step):
                y = np.log(closes[i - length + 1 : i + 1])
                x = np.arange(1, length + 1)
                
                sum_x = np.sum(x)
                sum_y = np.sum(y)
                sum_xx = np.sum(x * x)
                sum_xy = np.sum(x * y)
                
                denominator = (length * sum_xx) - (sum_x * sum_x)
                if denominator != 0:
                    slope = (length * sum_xy - sum_x * sum_y) / denominator
                    slopes_at_i.append(slope * -1)
                else:
                    slopes_at_i.append(0.0)
            
            oscillator_values[i] = np.mean(slopes_at_i)
            
        # 2. Cálculo da Signal Line
        osc_series = pd.Series(oscillator_values)
        signal_line = osc_series.rolling(window=sig_line).mean().fillna(0).to_numpy()
        
        # 3. Identificação de Cruzamentos (Gatilhos)
        # 1: Reversal Up (Crossover quando osc < 0)
        # -1: Reversal Down (Crossunder quando osc > 0)
        signals = np.zeros(n)
        for i in range(1, n):
            # Crossover (Cruza para cima)
            if oscillator_values[i-1] < signal_line[i-1] and oscillator_values[i] > signal_line[i]:
                #if oscillator_values[i] < 0: # Condição de "Reversal Up" do BigBeluga
                    #signals[i] = 1
                signals[i] = -2 if oscillator_values[i] > 0 else -1
            
            # Crossunder (Cruza para baixo)
            elif oscillator_values[i-1] > signal_line[i-1] and oscillator_values[i] < signal_line[i]:
                #if oscillator_values[i] > 0: # Condição de "Reversal Down" do BigBeluga
                    #signals[i] = -1
                signals[i] = 2 if oscillator_values[i] < 0 else 1
        return oscillator_values.tolist(), signal_line.tolist(), signals.tolist()








        