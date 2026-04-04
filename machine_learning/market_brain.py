import numpy as np
import pandas as pd
import ta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

from commons.utils.indicators.custom_indicators_utils import CustomIndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper


class MarketBrain:

    @staticmethod
    def add_indicators(df: pd.DataFrame, is_training: bool = False):
        # Criamos uma cópia para não afetar o DF original
        df = df.copy()

        # --- INDICADORES TÉCNICOS ---
        # SuperTrend e outros (usando o teu wrapper)
        df_temp = df.copy()
        if pd.api.types.is_datetime64_any_dtype(df_temp['timestamp']):
            df_temp['timestamp'] = df_temp['timestamp'].astype('int64') // 10 ** 6

        custom_indicators = CustomIndicatorsUtils(OhlcvWrapper(df_temp.values.tolist()), mode='custom')
        _, direction, _, _, _, _, _ = custom_indicators.supertrend()
        final_scores, _ = custom_indicators.calculate_super_score()

        rsi, _ = custom_indicators.rsi()

        df["super_score"] = final_scores
        df["st_direction"] = direction
        df["ema9"] = df["close"].ewm(span=9).mean()
        df["ema21"] = df["close"].ewm(span=21).mean()
        df["ema200"] = df["close"].ewm(span=200).mean()

        # FEATURE REAL: Distância do preço para a média (Percentagem)
        df["dist_ema9"] = (df["close"] - df["ema9"]) / df["ema9"]
        df["dist_ema21"] = (df["close"] - df["ema21"]) / df["ema21"]
        df["dist_ema200"] = (df["close"] - df["ema200"]) / df["ema200"]

        df["macd"] = df["ema9"] - df["ema21"]

        # MACD já é uma diferença, mas para ser "scale-invariant", dividimos pelo preço
        df["macd_rel"] = (df["ema9"] - df["ema21"]) / df["close"]

        #df["rsi"] = RSIIndicator(df["close"]).rsi().shift(2)
        df["rsi"] = pd.Series(rsi).shift(2)

        df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()

        # ATR relativo (Volatilidade em %)
        df["atr_rel"] = df["atr"] / df["close"]

        stoch = StochasticOscillator(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        df["pct_change"] = df["close"].pct_change().shift(1)
        df['momentum'] = df['close'].pct_change(periods=3)
        df['roc'] = df['close'].pct_change(periods=10)

        df['mean_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['z_score'] = (df['close'] - df['mean_20']) / df['std_20']

        df['vol_ema'] = df['volume'].rolling(window=20).mean()
        df['relative_volume'] = (df['volume'] / df['vol_ema']).shift(2)
        df['bb_width'] = (df['ema21'] - df['ema9']) / df['ema21']

        df['ema_200'] = df['close'].ewm(span=200).mean()
        df['above_ema200'] = (df['close'] > df['ema_200']).astype(int)

        #df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        adx_i = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_i.adx()
        df['di_diff'] = adx_i.adx_pos() - adx_i.adx_neg()  # Positivo = Bulls no comando

        df['choppiness'] = MarketBrain.get_choppiness(df)

        df['bb20_up'] = ta.volatility.bollinger_hband(df['close'], window=20)
        df['bb80_up'] = ta.volatility.bollinger_hband(df['close'], window=80)
        df['bb_ratio'] = df['bb20_up'] / df['bb80_up']  # Se > 1.05, estamos em zona de perigo/euforia

        """
        cols_model = [
            "rsi", "macd", "adx",  # Os clássicos (Momento e Força)
            "choppiness",  # O GPS (Lateral vs Tendência)
            "z_score",  # A Estatística (Extremos)
            "relative_volume",  # A Confirmação (Dinheiro Real)
            "bb_width",  # A Volatilidade (Aperto/Explosão)
            "dist_ema200"  # O Contexto (Onde estamos no mapa?)
        ]
        """
        cols_model = [
            "rsi", "atr", "macd", "pct_change", "ema9", "ema21", "stoch_k", "stoch_d",
            "z_score", "relative_volume", "bb_width", "above_ema200", "momentum", "roc", "adx",
            "super_score", "st_direction", "dist_ema200"
        ]
        """
        cols_model = [
            # --- OS EXECUTORES (O que já tinhas e funciona) ---
            "rsi", "macd", "stoch_k", "ema9", "ema21", "z_score", "relative_volume",

            # --- OS GERENTES (Os novos para limpar o lixo) ---
            "adx",  # Para saber se a tendência tem força
            "choppiness",  # Para fugir de mercados laterais (o teu maior inimigo)
            "bb_ratio"  # Para detetar zonas de exaustão extrema
        ]
        """

        # --- LÓGICA DE TARGET (APENAS TREINO) ---
        if is_training:
            # Aumentamos para 1.2% para filtrar o ruído e sair dos 99%
            margin = 0.012

            df["future_return"] = df["close"].shift(-20) / df["close"] - 1

            # Criamos labels mais equilibrados
            df["label"] = np.select(
                [df["future_return"] > margin, df["future_return"] < -margin],
                [2, 0],
                default=1
            )

            # IMPORTANTE: Dropna deve ser a última coisa antes de separar as colunas
            df = df.dropna(subset=cols_model + ["label"])

            # DEBUG: Vamos ver se as classes estão equilibradas agora
            print(f"DEBUG - Distribuição Labels: {df['label'].value_counts(normalize=True)}")
        else:

            # 1. FAZEMOS FFILL APENAS NAS ÚLTIMAS 3 LINHAS (Limite de segurança)
            # Se faltar mais do que 3 candles de indicadores, algo está mesmo errado com os dados
            df[cols_model] = df[cols_model].ffill(limit=3)

            # 2. REMOVEMOS NANS APENAS DAS COLUNAS QUE O MODELO USA
            # Isto ignora os NaNs da coluna 'future_return' (que não existem no modo predição)
            # e foca-se apenas no warm-up inicial das EMAs/ADX.
            df = df.dropna(subset=cols_model)

        features = df[cols_model]
        #print(df.head(2))
        #print(df.tail(2))
        return df, features

    @staticmethod
    def get_choppiness(df, window=14):
        # True Range
        tr1 = pd.DataFrame(df['high'] - df['low'])
        tr2 = pd.DataFrame(abs(df['high'] - df['close'].shift(1)))
        tr3 = pd.DataFrame(abs(df['low'] - df['close'].shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr_sum = tr.rolling(window).sum()
        high_low_diff = df['high'].rolling(window).max() - df['low'].rolling(window).min()

        # Índice de 0 a 100 (Acima de 61 é lateral total / Abaixo de 38 é tendência forte)
        return 100 * np.log10(atr_sum / high_low_diff) / np.log10(window)