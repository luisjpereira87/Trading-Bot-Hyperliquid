import numpy as np
import pandas as pd
import ta
from ta.momentum import StochasticOscillator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

from commons.utils.indicators.custom_indicators_utils import CustomIndicatorsUtils
from commons.utils.indicators.tv_indicators_utils import TvIndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper


class MarketBrain:

    @staticmethod
    def add_indicators_(df: pd.DataFrame, is_training: bool = False):
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

        # df["rsi"] = RSIIndicator(df["close"]).rsi().shift(2)
        df["rsi"] = pd.Series(rsi)

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

        # df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        adx_i = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_i.adx()
        df['di_diff'] = adx_i.adx_pos() - adx_i.adx_neg()  # Positivo = Bulls no comando

        df['choppiness'] = MarketBrain.get_choppiness(df)

        df['bb20_up'] = ta.volatility.bollinger_hband(df['close'], window=20)
        df['bb80_up'] = ta.volatility.bollinger_hband(df['close'], window=80)
        df['bb_ratio'] = df['bb20_up'] / df['bb80_up']  # Se > 1.05, estamos em zona de perigo/euforia

        cols_model = [
            "rsi", "atr", "macd", "pct_change", "ema9", "ema21", "stoch_k", "stoch_d",
            "z_score", "relative_volume", "bb_width", "above_ema200", "momentum", "roc", "adx",
            "super_score", "st_direction", "dist_ema200"
        ]
        """
        cols_model = [
            "rsi", "atr", "relative_volume", "above_ema200", "adx", "dist_ema200"
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
        # print(df.head(2))
        # print(df.tail(2))
        return df, features

    @staticmethod
    def prepare_bayesian_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma indicadores técnicos em categorias (bins) para o Bayes.
        """
        df_bayes = df.copy()

        # 1. Aplicamos a discretização
        df_bayes['rsi_bin'] = pd.cut(df_bayes['rsi'], bins=[0, 30, 70, 100],
                                     labels=['OVERSOLD', 'NEUTRAL', 'OVERBOUGHT'])

        df_bayes['chop_bin'] = pd.cut(df_bayes['choppiness'], bins=[0, 38, 61, 100],
                                      labels=['TRENDING', 'NORMAL', 'CHOPPY'])

        # 2. Criamos colunas binárias simples
        df_bayes['trend_state'] = np.where(df_bayes['above_ema200'] == 1, 'BULL', 'BEAR')

        # 3. Retornamos apenas o que o Bayes precisa para o JSON
        return df_bayes[['rsi_bin', 'chop_bin', 'trend_state', 'label']]

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

    @staticmethod
    def add_indicators(df: pd.DataFrame, is_training: bool = False):
        df = df.copy()

        df_temp = df.copy()
        if pd.api.types.is_datetime64_any_dtype(df_temp['timestamp']):
            df_temp['timestamp'] = df_temp['timestamp'].astype('int64') // 10 ** 6

        custom_indicators = CustomIndicatorsUtils(OhlcvWrapper(df_temp.values.tolist()), mode='custom')
        _, direction, _, _, _, _, _ = custom_indicators.supertrend()
        final_scores, _ = custom_indicators.calculate_super_score()
        tv_indicators = TvIndicatorsUtils(OhlcvWrapper(df_temp.values.tolist()), mode='custom')

        df["super_score"] = final_scores
        df["st_direction"] = direction

        df["squeeze_index"] = tv_indicators.squeeze_index()

        # --- 1. MÉTRICAS DE MOMENTUM (Para Tendência) ---
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        # Relação de Médias (9 vs 50) - Indica se o "comboio" está a andar
        df['ema_gap'] = (ta.trend.ema_indicator(df['close'], 9) / ta.trend.ema_indicator(df['close'], 50)) - 1

        # --- 2. MÉTRICAS DE VOLATILIDADE (Para Reversão/Rompimento) ---
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        df['bb_pband'] = bb.bollinger_pband()  # 1.0 = Topo da banda, 0.0 = Fundo
        df['atr_norm'] = ta.volatility.average_true_range(df['high'], df['low'], df['close']) / df['close']

        # --- 3. MÉTRICAS DE FORÇA (Volume) ---
        df['vol_shock'] = df['volume'] / df['volume'].rolling(20).mean()

        df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()

        df['ema_200'] = df['close'].ewm(span=200).mean()
        df['above_ema200'] = (df['close'] > df['ema_200']).astype(int)

        df["dist_ema200"] = (df["close"] - df["ema_200"]) / df["ema_200"]

        df['mean_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['z_score'] = (df['close'] - df['mean_20']) / df['std_20']

        df['momentum'] = df['close'].pct_change(periods=3)
        adx_i = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_i.adx()

        df['price_change'] = df['close'].diff()
        df['consecutive_downs'] = (df['price_change'] < 0).astype(int).rolling(window=2).sum()

        # 2. Força da queda (Vela atual + Anterior)
        df['last_2_candles_sum'] = df['close'].pct_change(periods=2)

        # 3. Filtro de Volume na queda
        df['down_volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean()) & (df['price_change'] < 0)

        df['rsi_ema'] = df['rsi'].rolling(window=14).mean()

        # Opção A: Distância (valor contínuo - o LightGBM adora isto)
        df['rsi_above_ema'] = df['rsi'] - df['rsi_ema']

        df['price_velocity'] = df['close'].diff(2) / df['atr']

        # 1. Calcula as EMAs
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()

        # 2. Diferença Relativa (Feature Contínua - O LGBM adora isto)
        # Positivo = EMA9 acima | Negativo = EMA9 abaixo
        df['ema_spread'] = (df['ema9'] - df['ema21']) / df['ema21'] * 100

        # 3. Sinal de Cruzamento (O "Momento" do Cross)
        # Comparamos o estado atual com o anterior (shift)
        df['ema_cross_up'] = ((df['ema9'] > df['ema21']) & (df['ema9'].shift(1) <= df['ema21'].shift(1))).astype(int)
        df['ema_cross_down'] = ((df['ema9'] < df['ema21']) & (df['ema9'].shift(1) >= df['ema21'].shift(1))).astype(int)

        df['rsi_sma'] = df['rsi'].rolling(window=7).mean()
        df['rsi_trend'] = df['rsi'] - df['rsi_sma']  # Se positivo, o momentum está a acelerar

        # df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']

        df['vol_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()

        # --- DETEÇÃO DE ESTRUTURA (SWINGS & BREAKS) ---
        bs_df = MarketBrain.breakout_structure(df)
        df['break_up'] = bs_df['break_up']
        df['break_down'] = bs_df['break_down']
        df['higher_high'] = bs_df['higher_high']
        df['lower_low'] = bs_df['lower_low']

        # --- DETEÇÃO DE ZONAS EXTREMAS BB ---
        bb_df = MarketBrain.bb_extreme_zones(df)
        df['double_bb_signal'] = bb_df['double_bb_signal']
        df['bb_squeeze_diff_low'] = bb_df['bb_squeeze_diff_low']
        df['bb20_low_reverting'] = bb_df['bb20_low_reverting']

        sf_df = MarketBrain.add_sanity_filters(df)
        df['ema_macro_slope'] = sf_df['ema_macro_slope']
        df['adx'] = sf_df['adx']
        df['dmi_plus'] = sf_df['dmi_plus']
        df['dmi_minus'] = sf_df['dmi_minus']
        df['dist_upper_bb80'] = sf_df['dist_upper_bb80']
        df['dist_lower_bb80'] = sf_df['dist_lower_bb80']
        df['filter_no_buy'] = sf_df['filter_no_buy']
        df['filter_no_sell'] = sf_df['filter_no_sell']

        df['top_exhaustion_break'] = ((df['double_bb_signal'] == 1) & (df['lower_low'] == 1)).astype(int)
        df['bottom_exhaustion_break'] = ((df['double_bb_signal'] == 2) & (df['higher_high'] == 1)).astype(int)

        df_co = MarketBrain.add_custom_oscillators(df)

        df["super_score"] = df_co['super_score']
        df["sq_index"] = df_co['super_score']
        df["regime_filter_trend"] = df_co['super_score']
        df["regime_filter_voltrend"] = df_co['regime_filter_voltrend']
        df["regime_filter_efficiency"] = df_co['regime_filter_efficiency']

        df["andean_bull"] = df_co['andean_bull']
        df["andean_bear"] = df_co['andean_bear']
        df["andean_signal"] = df_co['andean_signal']
        df["regression_oscillator"] = df_co['regression_oscillator']
        df["regression_signal"] = df_co['regression_signal']

        cols_model = ["rsi", "ema_gap", "bb_pband", "atr_norm", "vol_shock", "atr",
                      "rsi_above_ema", "price_velocity", "ema_spread", "ema_cross_up", "ema_cross_down"
                      ]
        """

        cols_model = [
            "rsi_trend", "vol_zscore",  # As novas "estrelas"
            "ema_gap", "atr_norm", "rsi_above_ema",
            "price_velocity", "ema_spread", "ema_cross_up", "ema_cross_down",
            "adx", "dmi_plus", "dmi_minus", "dist_upper_bb80"
        ]
        """
        """
        cols_model = [
            "break_up", "break_down",  # Gatilhos de quebra
            "higher_high", "lower_low",  # Contexto de tendência
            "rsi_above_ema", "ema_gap",  # Momentum
            "atr_norm", "vol_shock",  # Volatilidade
            "price_velocity", "ema_spread",
            "ema_cross_up", "ema_cross_down",
            "double_bb_signal", "bb_squeeze_diff_low", "bb20_low_reverting",
            "ema_macro_slope", "adx", "dmi_plus", "dmi_minus", "dist_upper_bb80", "dist_lower_bb80",
            "filter_no_buy", "filter_no_sell",
        ]
        """
        """
        # Contexto de Regressão
        df['reg_spread'] = df['regression_oscillator'] - df['regression_signal']

        # Contexto de Andean (Onde está a força real?)
        df['andean_bull_spread'] = df['andean_bull'] - df['andean_signal']
        df['andean_bear_spread'] = df['andean_bear'] - df['andean_signal']

        # Relação Direta entre Bull e Bear (Quem ganha a luta?)
        df['andean_diff'] = df['andean_bull'] - df['andean_bear']

        # Contexto de 'Mergulho' (O valor atual em relação ao passado recente)
        df['reg_zscore'] = (df['regression_oscillator'] - df['regression_oscillator'].rolling(20).mean()) / df[
            'regression_oscillator'].rolling(20).std()
        """
        """
        cols_model = [
            "reg_spread",
            "andean_bull_spread", "andean_bear_spread", "andean_diff",
            "atr",
            # "atr_norm"  # Mantém este como "âncora" de volatilidade
        ]
        """

        # 1. Trata os infinitos que os indicadores podem ter gerado
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 2. Remove ou preenche os valores nulos (NaN)
        # Se uma coluna nova precisa de X velas para aquecer, ela terá NaNs no topo
        df.fillna(method='ffill', inplace=True)  # Repete o último valor válido
        df.fillna(0, inplace=True)  # Se ainda houver NaNs (no início), coloca 0

        cols_train = [c for c in cols_model if c != 'atr']
        if is_training:
            window = 12

            # Definimos o alvo real que o teu bot persegue
            min_profit_pct = 0.02  # 2%

            for i in range(len(df) - window):
                price_entry = df['close'].iloc[i]

                # Definimos os patamares de preço para este trade específico
                target_buy = price_entry * (1 + min_profit_pct)
                target_sell = price_entry * (1 - min_profit_pct)
                stop_buy = price_entry - (df['atr'].iloc[i] * 1.0)  # Stop inicial continua no ATR
                stop_sell = price_entry + (df['atr'].iloc[i] * 1.0)

                future_segment = df.iloc[i + 1: i + window + 1]
                max_high = future_segment['high'].max()
                min_low = future_segment['low'].min()

                # Lógica de COMPRA: O preço subiu 2% antes de bater no SL?
                if (max_high >= target_buy) and (min_low > stop_buy):
                    df.at[df.index[i], 'label'] = 2

                # Lógica de VENDA: O preço caiu 2% antes de bater no SL?
                elif (min_low <= target_sell) and (max_high < stop_sell):
                    df.at[df.index[i], 'label'] = 0

                else:
                    df.at[df.index[i], 'label'] = 1

        return df, df[cols_train]

    @staticmethod
    def find_best_params(df: pd.DataFrame, features: list):
        import lightgbm as lgb
        from sklearn.metrics import f1_score

        # --- 1. CONFIGURAÇÃO DA BUSCA ---
        profit_test = [0.015, 0.02, 0.025]
        window_test = [10, 12, 16]

        best_f1 = -1
        best_p = 0.02
        best_w = 12

        print("🔍 Otimizando alvo e janela para encontrar a melhor Precision...")

        for p_test in profit_test:
            for w_test in window_test:
                # Criamos uma cópia temporária para rotular
                temp_df = df.copy()
                temp_df['label'] = 1

                # --- TEU BLOCO DE LABELING (VERSÃO TESTE) ---
                for i in range(len(temp_df) - w_test):
                    price_entry = temp_df['close'].iloc[i]
                    target_buy = price_entry * (1 + p_test)
                    target_sell = price_entry * (1 - p_test)
                    stop_buy = price_entry - (temp_df['atr'].iloc[i] * 1.0)
                    stop_sell = price_entry + (temp_df['atr'].iloc[i] * 1.0)

                    future_segment = temp_df.iloc[i + 1: i + w_test + 1]
                    max_high = future_segment['high'].max()
                    min_low = future_segment['low'].min()

                    if (max_high >= target_buy) and (min_low > stop_buy):
                        temp_df.at[temp_df.index[i], 'label'] = 2
                    elif (min_low <= target_sell) and (max_high < stop_sell):
                        temp_df.at[temp_df.index[i], 'label'] = 0

                # --- TREINO RÁPIDO DE VALIDAÇÃO ---
                # Usamos os últimos 20% para validar se esta configuração funciona
                df_clean = temp_df.dropna()
                split = int(len(df_clean) * 0.8)
                train_data = df_clean.iloc[:split]
                test_data = df_clean.iloc[split:]

                if len(train_data['label'].unique()) < 3: continue  # Pula se não houver Buy/Sell suficientes

                clf = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.1, verbose=-1)
                clf.fit(train_data[features], train_data['label'])

                preds = clf.predict(test_data[features])
                # F1-Score Macro apenas para as classes 0 e 2 (ignora o Hold no cálculo de performance)
                f1 = f1_score(test_data['label'], preds, labels=[0, 2], average='macro', zero_division=0)

                num_signals = (preds != 1).sum()
                print(f"  > Teste {p_test * 100}%/{w_test}v: F1-Score = {f1:.4f} | Sinais = {num_signals}")

                if f1 > best_f1 and num_signals > 5:  # Garante que existem sinais reais
                    best_f1 = f1
                    best_p = p_test
                    best_w = w_test

        return best_p, best_w, best_f1

    @staticmethod
    def kalman_filter(data):
        data = np.array(data)
        n = len(data)

        # Parâmetros conforme o script Pine
        process_noise = 0.2
        measurement_error = 2.0
        error_est = 1.0

        # Inicialização
        estimate = data[0] if n > 0 else 0
        kalman_values = np.zeros(n)

        for i in range(n):
            # No Pine: prediction := estimate
            prediction = estimate

            # Cálculo do ganho de Kalman
            kalman_gain = error_est / (error_est + measurement_error)

            # Atualização da estimativa
            estimate = prediction + kalman_gain * (data[i] - prediction)

            # Atualização do erro da estimativa
            error_est = (1 - kalman_gain) * error_est + process_noise

            kalman_values[i] = estimate

        return kalman_values

    @staticmethod
    def breakout_structure(df: pd.DataFrame):
        # 1. Identificar Swing Highs (Topos locais)
        # Um topo é quando a vela atual é maior que as 2 anteriores e as 2 seguintes
        df['is_swing_high'] = (
                (df['high'] > df['high'].shift(1)) &
                (df['high'] > df['high'].shift(2)) &
                (df['high'] > df['high'].shift(-1)) &
                (df['high'] > df['high'].shift(-2))
        ).astype(int)

        # 2. Identificar Swing Lows (Fundos locais)
        df['is_swing_low'] = (
                (df['low'] < df['low'].shift(1)) &
                (df['low'] < df['low'].shift(2)) &
                (df['low'] < df['low'].shift(-1)) &
                (df['low'] < df['low'].shift(-2))
        ).astype(int)

        # 3. Mapear o valor do ÚLTIMO topo e fundo confirmado
        # Usamos ffill() para que o robô saiba sempre onde está a "barreira" a ser batida
        df['last_high'] = df['high'].where(df['is_swing_high'] == 1).ffill()
        df['last_low'] = df['low'].where(df['is_swing_low'] == 1).ffill()

        # 4. Quebras de Estrutura (BOS - Break of Structure)
        # Bullish Break: Fecho atual acima do último topo
        df['break_up'] = (df['close'] > df['last_high'].shift(1)).astype(int)

        # Bearish Break: Fecho atual abaixo do último fundo
        df['break_down'] = (df['close'] < df['last_low'].shift(1)).astype(int)

        # 5. Tendência de Estrutura (Higher Highs / Lower Lows)
        # Útil para o modelo saber se estamos em "escada" de alta ou baixa
        df['higher_high'] = (df['last_high'] > df['last_high'].shift(5)).astype(int)
        df['lower_low'] = (df['last_low'] < df['last_low'].shift(5)).astype(int)

        return df

    @staticmethod
    def bb_extreme_zones(df: pd.DataFrame):
        # 1. Base e Volatilidade
        df['ltf_basis'] = MarketBrain.kalman_filter(df['close'])
        std_20 = df['close'].rolling(20).std()
        std_80 = df['close'].rolling(80).std()

        # 2. BB Curta (20) com Kalman
        df['bb20_up'] = df['ltf_basis'] + (2.0 * std_20)
        df['bb20_low'] = df['ltf_basis'] - (2.0 * std_20)

        # 3. BB Longa (80)
        # Calculamos a base da 80 (pode ser SMA ou outro Kalman mais lento)
        bb80_basis = df['close'].rolling(80).mean()
        bb80_up_raw = bb80_basis + (2.25 * std_80)
        bb80_low_raw = bb80_basis - (2.25 * std_80)

        # 4. Suavização das Bandas Longas (para evitar sinais nervosos)
        df['bb80_up'] = bb80_up_raw.rolling(window=20).mean()
        df['bb80_low'] = bb80_low_raw.rolling(window=20).mean()

        # 5. Features de Exaustão (Onde a mágica acontece)
        # Diferença entre as bandas (Squeeze/Expansion extremo)
        df['bb_squeeze_diff_low'] = df['bb80_low'] - df['bb20_low']
        df['bb_squeeze_diff_high'] = df['bb20_up'] - df['bb80_up']

        # 6. Sinais de Setup (Binários para o Modelo)
        # Compra: BB80 Low acima da BB20 Low E preço abaixo da BB80
        df['extreme_buy_setup'] = ((df['bb80_low'] > df['bb20_low']) & (df['low'] < df['bb80_low'])).astype(int)

        # Venda: BB80 Up abaixo da BB20 Up E preço acima da BB80
        df['extreme_sell_setup'] = ((df['bb80_up'] < df['bb20_up']) & (df['high'] > df['bb80_up'])).astype(int)

        # 7. Inclinação da BB20 (Filtro de "faca a cair")
        df['bb20_low_reverting'] = (df['bb20_low'] > df['bb20_low'].shift(1)).astype(int)
        df['bb20_up_reverting'] = (df['bb20_up'] < df['bb20_up'].shift(1)).astype(int)

        # 8. Filtro de RSI (Vetorizado)
        # Marcamos 1 se o RSI esteve em zona extrema nas últimas 3 velas
        df['rsi_extreme_low'] = df['rsi'].rolling(3).apply(lambda x: (x < 30).any()).fillna(0).astype(int)
        df['rsi_extreme_high'] = df['rsi'].rolling(3).apply(lambda x: (x > 70).any()).fillna(0).astype(int)

        # 9. CONFLUÊNCIA FINAL (A feature que resume o teu indicador)
        df['double_bb_signal'] = 0
        df.loc[(df['extreme_buy_setup'] == 1) & (df['rsi_extreme_low'] == 1) & (
                df['close'] > df['ltf_basis']), 'double_bb_signal'] = 1
        df.loc[(df['extreme_sell_setup'] == 1) & (df['rsi_extreme_high'] == 1) & (
                df['close'] < df['ltf_basis']), 'double_bb_signal'] = -1

        return df

    @staticmethod
    def add_sanity_filters(df: pd.DataFrame):
        # 1. TENDÊNCIA MACRO (O "Navio")
        # Usamos uma EMA lenta (ex: 100 ou 200) para ver a inclinação
        df['ema_macro'] = df['close'].ewm(span=100, adjust=False).mean()
        # Slope: Se positivo, tendência de alta. Se negativo, queda.
        df['ema_macro_slope'] = df['ema_macro'].diff(3)

        # 2. FORÇA DA TENDÊNCIA (ADX)
        # Evita operar contra-tendência quando o "sangue" corre forte
        adx_i = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_i.adx()
        df['dmi_plus'] = adx_i.adx_pos()
        df['dmi_minus'] = adx_i.adx_neg()

        # 3. DISTÂNCIA ÀS BANDAS (Localização)
        # Mede onde o preço está em relação ao "teto" e "chão" da BB80
        # Se o valor for muito pequeno, estamos colados ao topo (Perigo para BUY)
        df['dist_upper_bb80'] = (df['bb80_up'] - df['close']) / df['close']
        df['dist_lower_bb80'] = (df['close'] - df['bb80_low']) / df['close']

        # 4. FILTROS BINÁRIOS (Para ajudar o modelo a decidir)
        # Proibição de Buy se estivermos em queda forte e longe da média
        df['filter_no_buy'] = ((df['ema_macro_slope'] < 0) & (df['dmi_minus'] > df['dmi_plus'])).astype(int)
        # Proibição de Sell se estivermos em tendência de alta forte
        df['filter_no_sell'] = ((df['ema_macro_slope'] > 0) & (df['dmi_plus'] > df['dmi_minus'])).astype(int)

        return df

    @staticmethod
    def add_custom_oscillators(df: pd.DataFrame):
        df_temp = df.copy()
        if pd.api.types.is_datetime64_any_dtype(df_temp['timestamp']):
            df_temp['timestamp'] = df_temp['timestamp'].astype('int64') // 10 ** 6

        custom_indicators = CustomIndicatorsUtils(OhlcvWrapper(df_temp.values.tolist()), mode='custom')
        _, direction, _, _, _, _, _ = custom_indicators.supertrend()
        final_scores, _ = custom_indicators.calculate_super_score()

        tv_indicators = TvIndicatorsUtils(OhlcvWrapper(df_temp.values.tolist()), mode='custom')
        sq_index = tv_indicators.squeeze_index()

        regime_filter = tv_indicators.regime_filter()
        trend = regime_filter['trend']
        voltrend = regime_filter['voltrend']
        efficiency = regime_filter['efficiency']

        andean = tv_indicators.calculate_andean_oscillator()
        andean_bull = andean['bull']
        andean_bear = andean['bear']
        andean_signal = andean['signal']

        oscillator_values, signal_line, _, _, _ = tv_indicators.regression_slope_oscillator()

        df["super_score"] = final_scores

        df["sq_index"] = sq_index

        df["regime_filter_trend"] = trend
        df["regime_filter_voltrend"] = voltrend
        df["regime_filter_efficiency"] = efficiency

        df["andean_bull"] = andean_bull
        df["andean_bear"] = andean_bear
        df["andean_signal"] = andean_signal

        df["regression_oscillator"] = oscillator_values
        df["regression_signal"] = signal_line

        return df
