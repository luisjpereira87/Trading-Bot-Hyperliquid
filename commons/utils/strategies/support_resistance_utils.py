from typing import List, Tuple

import numpy as np

from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper


class SupportResistanceUtils:

    @staticmethod
    def detect_support_resistance(
        candles: OhlcvWrapper,
        lookback: int = 50,
        tolerance_pct: float = 0.01  # 1%
    ) -> tuple[float, float]:
        highs = [c.high for c in candles.get_recent_closed(lookback)]
        lows = [c.low for c in candles.get_recent_closed(lookback)]

        resistance = max(highs)
        support = min(lows)

        return resistance, support
    
    @staticmethod
    def detect_multiple_support_resistance(
        candles: OhlcvWrapper,
        lookback: int = 50,
        tolerance_pct: float = 0.005
    ) -> tuple[list[float], list[float]]:
        """
        Detecta múltiplos níveis de suporte e resistência com base em extremos locais.
        Os níveis próximos são agrupados com base em `tolerance_pct`.

        :return: (lista de resistências, lista de suportes)
        """
        highs = [c.high for c in candles.get_recent_closed(lookback)]
        lows = [c.low for c in candles.get_recent_closed(lookback)]

        resistances = []
        supports = []

        for i in range(2, len(highs) - 2):
            # Máximo local
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                price = highs[i]
                if not any(abs(price - r) / price < tolerance_pct for r in resistances):
                    resistances.append(price)

            # Mínimo local
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                price = lows[i]
                if not any(abs(price - s) / price < tolerance_pct for s in supports):
                    supports.append(price)

        return sorted(resistances, reverse=True), sorted(supports)

    @staticmethod
    def get_distance_to_levels(ohlcv: OhlcvWrapper, price_ref: float, lookback: int = 50) -> tuple[float, float]:
        recent = ohlcv.get_recent_closed(lookback)

        highs = [candle.high for candle in recent]
        lows = [candle.low for candle in recent]

        resistance = max(highs)
        support = min(lows)
        price = price_ref  # ou self.ohlcv.get_current_candle().close

        dist_to_res = abs(resistance - price)
        dist_to_sup = abs(price - support)

        return dist_to_res, dist_to_sup
    
    @staticmethod
    def calculate_bands(ohlcv: OhlcvWrapper, multiplier):
        indicators = IndicatorsUtils(ohlcv)

        closes = ohlcv.closes

        atr = indicators.atr()
        upper_band = [closes[i] + multiplier * atr[i] for i in range(len(atr))]
        lower_band = [closes[i] - multiplier * atr[i] for i in range(len(atr))]

        return upper_band, lower_band
    
    @staticmethod
    def find_local_extrema_swings_psar(ohlcv: OhlcvWrapper, sequential: bool = True):
        closes = ohlcv.closes
        timestamps = ohlcv.timestamps
        psar = IndicatorsUtils(ohlcv).psar()

        high_pivots = []
        low_pivots = []
        pivots_high_index = []
        pivots_low_index = []

        last_pivot_type = None  # "high" ou "low"

        for i in range(1, len(psar)):
            # PSAR acima -> abaixo do preço = fundo
            if psar[i-1] > closes[i-1] and psar[i] < closes[i]:
                if (not sequential or last_pivot_type != "low"):
                    if not pivots_low_index or pivots_low_index[-1] != i:
                        low_pivots.append(timestamps[i])
                        pivots_low_index.append(i)
                        last_pivot_type = "low"

            # PSAR abaixo -> acima do preço = topo
            elif psar[i-1] < closes[i-1] and psar[i] > closes[i]:
                if (not sequential or last_pivot_type != "high"):
                    if not pivots_high_index or pivots_high_index[-1] != i:
                        high_pivots.append(timestamps[i])
                        pivots_high_index.append(i)
                        last_pivot_type = "high"

        return high_pivots, low_pivots, pivots_high_index, pivots_low_index
    
    @staticmethod
    def filter_close_pivots(pivots_idx, prices, min_distance=0.005):
        filtered = []
        if not pivots_idx:
            return filtered
        pivots_idx = sorted(pivots_idx)
        group = [pivots_idx[0]]

        for idx in pivots_idx[1:]:
            # Distância relativa no preço entre este pivot e último do grupo
            dist = abs(prices[idx] - prices[group[-1]]) / prices[group[-1]]
            if dist < min_distance:
                # Mantém só o pivot mais extremo do grupo
                if prices[idx] > prices[group[-1]]:
                    group[-1] = idx
            else:
                filtered.extend(group)
                group = [idx]
        filtered.extend(group)
        return filtered
    
    @staticmethod
    def filter_close_pivots_with_volume(pivots_idx, prices, volumes, avg_volume, min_distance=0.005):
        """
        Filtra pivots muito próximos, mantém só os mais extremos e com volume acima da média.

        pivots_idx: lista de índices dos pivots
        prices: lista de preços (highs ou lows)
        volumes: lista de volumes
        avg_volume: valor médio do volume de referência
        min_distance: distância mínima relativa no preço para separar pivots
        """
        filtered = []
        if not pivots_idx:
            return filtered

        pivots_idx = sorted(pivots_idx)
        group = []

        for idx in pivots_idx:
            # Ignora pivots com volume baixo
            if volumes[idx] < avg_volume:
                continue

            if not group:
                group = [idx]
                continue

            dist = abs(prices[idx] - prices[group[-1]]) / prices[group[-1]]
            if dist < min_distance:
                # Mantém só o pivot mais extremo no grupo
                if prices[idx] > prices[group[-1]]:
                    group[-1] = idx
            else:
                filtered.extend(group)
                group = [idx]

        filtered.extend(group)
        return filtered


    @staticmethod
    def find_pivots(ohlcv: OhlcvWrapper, left: int = 3, right: int = 3) -> Tuple[List[int], List[int]]:
        """
        Detecta pivôs locais de alta (topos) e baixa (fundos) no gráfico.
        Retorna dois arrays de índices: (pivots_high, pivots_low)
        """
        highs = ohlcv.highs
        lows = ohlcv.lows
        length = len(highs)
        pivots_high = []
        pivots_low = []

        for i in range(left, length - right):
            high_candidate = highs[i]
            is_pivot_high = all(high_candidate > highs[j] for j in range(i - left, i)) and \
                            all(high_candidate > highs[j] for j in range(i + 1, i + right + 1))
            if is_pivot_high:
                pivots_high.append(i)

            low_candidate = lows[i]
            is_pivot_low = all(low_candidate < lows[j] for j in range(i - left, i)) and \
                           all(low_candidate < lows[j] for j in range(i + 1, i + right + 1))
            if is_pivot_low:
                pivots_low.append(i)

        return pivots_high, pivots_low
    
    @staticmethod
    def linear_slope(values: List[float]) -> float:
        """
        Calcula a inclinação da reta (slope) que melhor ajusta os pontos fornecidos.
        Usa regressão linear simples.
        
        :param values: lista de valores (ex: preços ou bandas)
        :return: coeficiente angular da regressão linear (slope)
        """
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        y = np.array(values)
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0] # type: ignore
        return slope
    
    @staticmethod   
    def ratio_support_resistence(ohlcv: OhlcvWrapper) -> float:
        resistance, support = SupportResistanceUtils.detect_support_resistance(ohlcv, lookback=20, tolerance_pct=0.02)
        close_price = ohlcv.get_last_closed_candle().close

        channel_height = resistance - support

        channel_position = 0
        if channel_height > 0:
            # evitar divisão por zero, devolve penalização neutra (0) ou outro valor que faças sentido
            channel_position = (close_price - support) / channel_height
        return channel_position    
    
    @staticmethod
    def channel_position_normalized(ohlcv: OhlcvWrapper, multiplier: float) -> float:
        """
        Retorna a posição do preço dentro do canal normalizada entre 0 e 1.
        0 = preço na base do canal
        1 = preço no topo do canal
        """
        upper_band, lower_band = SupportResistanceUtils.calculate_bands(ohlcv, multiplier=multiplier)
        close = ohlcv.get_last_closed_candle().close

        if not upper_band or not lower_band or len(upper_band) < 1:
            return 0.0

        top = upper_band[-1]
        bottom = lower_band[-1]
        band_range = top - bottom
        if band_range == 0:
            return 0.0
        print("BANDS", top, bottom)
        # Distância relativa dentro do canal
        position = (close - bottom) / band_range

        # Limitar a [0, 1]
        return max(min(position, 1.0), 0.0)