import numpy as np
import pandas as pd
import ta
from ta.momentum import StochasticOscillator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange
from ta.volume import ChaikinMoneyFlowIndicator

from commons.utils.indicators.custom_indicators_utils import CustomIndicatorsUtils
from commons.utils.indicators.tv_indicators_utils import TvIndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper


class MarketBrain:

    @staticmethod
    def add_indicators_(df: pd.DataFrame, is_training: bool = False):
        df = df.copy()
        df_temp = df.copy()
        if pd.api.types.is_datetime64_any_dtype(df_temp['timestamp']):
            df_temp['timestamp'] = df_temp['timestamp'].astype('int64') // 10 ** 6

        custom_indicators = CustomIndicatorsUtils(OhlcvWrapper(df_temp.values.tolist()), mode='custom')
        tv_indicators = TvIndicatorsUtils(OhlcvWrapper(df_temp.values.tolist()), mode='custom')
        supertrend, trend, final_upperband, final_lowerband, supertrend_smooth, direction, perf_score = custom_indicators.supertrend()

        df['squeeze_index'] = tv_indicators.squeeze_index()
        df['st_direction'] = direction
        df['st_lowerband'] = final_lowerband
        df['st_upperband'] = final_upperband

        # --- 1. MÉTRICAS DE MOMENTUM (Para Tendência) ---
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

        df['price_velocity'] = df['close'].diff(2) / df['atr']

        # 1. Calcula as EMAs
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()

        # 2. Diferença Relativa (Feature Contínua - O LGBM adora isto)
        # Positivo = EMA9 acima | Negativo = EMA9 abaixo
        df['ema_spread'] = (df['ema9'] - df['ema21']) / df['ema21'] * 100

        df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema100'] = ta.trend.ema_indicator(df['close'], window=100)
        df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)

        # Gaps das Médias Longas (Distância)
        df['gap_9_21'] = (df['ema9'] - df['ema21']) / df['ema21']
        df['gap_21_50'] = (df['ema21'] - df['ema50']) / df['ema50']
        df['gap_50_200'] = (df['ema50'] - df['ema200']) / df['ema200']
        df['gap_100_200'] = (df['ema100'] - df['ema200']) / df['ema200']

        # --- 2. Momentum de Cruzamento (As Ondas) ---
        # MACD Cross
        macd = ta.trend.MACD(df['close'])
        df['macd_line'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_cross_gap'] = df['macd_line'] - df['macd_signal']

        # Stochastic Cross
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['stoch_cross_gap'] = df['stoch_k'] - df['stoch_d']

        # --- 3. RSI Cross (O Gatilho) ---
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_ema'] = df['rsi'].rolling(window=14).mean()
        df['rsi_cross_gap'] = df['rsi'] - df['rsi_ema']

        window = 10
        direction = abs(df['close'] - df['close'].shift(window))

        # 2. Movimento Total (Soma das variações individuais de cada vela)
        # Calculamos a diferença absoluta entre cada vela e somamos na janela
        volatility = df['close'].diff().abs().rolling(window=window).sum()

        # 3. Efficiency Ratio
        # Adicionamos 1e-9 para evitar divisão por zero se o preço ficar parado
        df['efficiency_ratio'] = direction / (volatility + 1e-9)

        # 2. Distância Negativa (Gap)
        # Se o preço cruzar a EMA9 para baixo com força, o gap fica negativo rápido
        df['price_ema_gap'] = (df['close'] - df['ema9']) / df['ema9']

        # 1. Média Móvel (Middle Band)
        df['bb_middle'] = df['close'].rolling(window=20).mean()

        # 2. Desvio Padrão
        df['bb_std'] = df['close'].rolling(window=20).std()

        # 3. Bandas Superior e Inferior
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

        # 4. Bollinger Bandwidth (O filtro de mercado lateral)
        # Mede a distância entre as bandas em percentagem
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        psar_df = ta.trend.PSARIndicator(df['high'], df['low'], df['close'], step=0.02, max_step=0.2)
        df['psar'] = psar_df.psar()

        df['rsi_psar'] = 0  # Estado Neutro
        # Engolfo de Alta: Vela verde atual engole o corpo da vermelha anterior
        df.loc[(df['close'] > df['psar']) & (df['rsi'] > 50) & (df['rsi'] > df['rsi_ema']), 'rsi_psar'] = 1
        # Engolfo de Baixa: Vela vermelha atual engole o corpo da verde anterior
        df.loc[(df['close'] < df['psar']) & (df['rsi'] < 50) & (df['rsi'] < df['rsi_ema']), 'rsi_psar'] = -1

        # Resistência: O preço mais alto das últimas X velas
        df['resistance'] = df['high'].rolling(window=window).max()
        # Suporte: O preço mais baixo das últimas X velas
        df['support'] = df['low'].rolling(window=window).min()

        # DISTÂNCIA RELATIVA (A feature que o ML realmente ama)
        # Em vez de preços brutos, damos a % de distância
        df['dist_to_res'] = (df['resistance'] - df['close']) / df['close']
        df['dist_to_sup'] = (df['close'] - df['support']) / df['close']

        # POSIÇÃO DENTRO DO RANGE (0 a 1)
        # 0 = No suporte | 1 = Na resistência
        df['range_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])

        # Identificar o momento exato onde as linhas se cruzam
        df['ema_cross_short'] = (df['ema9'] > df['ema21']).astype(int)

        # Contar quantas velas consecutivas estão no mesmo estado
        # Isto cria uma contagem: ex: +1, +2, +3 numa tendência de alta, ou -1, -2, -3 numa de baixa
        df['ema_cross_age'] = df['ema_cross_short'].groupby(
            (df['ema_cross_short'] != df['ema_cross_short'].shift()).cumsum()).cumcount() + 1

        # Ajustamos para que tendências de baixa fiquem negativas
        df['ema_cross_age'] = np.where(df['ema_cross_short'] == 1, df['ema_cross_age'], -df['ema_cross_age'])

        # 1. CÁLCULOS ANATÓMICOS DA VELA (Base para Price Action)
        candle_body = (df['close'] - df['open']).abs()
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)

        # 2. FEATURE DE REVERSÃO 1: Rejeição de Pavio (Pinbar)
        df['pa_reversion_pavio'] = 0  # Estado Neutro (Hold)
        # Rejeição de Queda: Pavio inferior é o dobro do corpo E fechou acima da Supertrend inferior
        df.loc[(lower_shadow > candle_body * 2) & (df['close'] > df['st_lowerband']), 'pa_reversion_pavio'] = 1
        # Rejeição de Alta: Pavio superior é o dobro do corpo E fechou abaixo da Supertrend superior
        df.loc[(upper_shadow > candle_body * 2) & (df['close'] < df['st_upperband']), 'pa_reversion_pavio'] = -1

        # 3. FEATURE DE REVERSÃO 2: Engolfo Violento
        df['pa_engulfing'] = 0  # Estado Neutro
        # Engolfo de Alta: Vela verde atual engole o corpo da vermelha anterior
        df.loc[(df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) &
               (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)), 'pa_engulfing'] = 1
        # Engolfo de Baixa: Vela vermelha atual engole o corpo da verde anterior
        df.loc[(df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)) &
               (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1)), 'pa_engulfing'] = -1

        # 4. CONVERTER PARA TIPO CATEGÓRICO (Fundamental para o LightGBM)
        df['pa_reversion_pavio'] = df['pa_reversion_pavio'].astype('category')
        df['pa_engulfing'] = df['pa_engulfing'].astype('category')

        # Descobre o suporte das últimas 100 velas
        df['suporte_100'] = df['low'].rolling(window=100).min()
        # Calcula a distância percentual do preço atual para esse suporte
        df['dist_to_support'] = (df['close'] - df['suporte_100']) / df['suporte_100']

        # Largura das bandas normalizada pelo preço (percentual)
        df['bb_width_norm'] = df['bb_width'] / df['close']

        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

        # 1. Alinhamento de Médias Móveis Longas (Mapeia a tendência do dia/semana)
        # Se o preço estiver acima da EMA 200, a tendência é estritamente de alta
        df['macro_trend_alignment'] = np.where(df['close'] > df['ema_200'], 1, -1)

        # 2. Direção do Preço em Janelas Largas (Ex: Força acumulada nas últimas 48 velas)
        df['macro_momentum'] = (df['close'] - df['close'].shift(48)) / df['close'].shift(48)

        # Verifica se a vela atual tem um mínimo maior que a anterior (True/False -> 1/0)
        df['hl_check'] = (df['low'] > df['low'].shift(1)).astype(int)

        # Conta a soma consecutiva das últimas 5 velas
        df['higher_lows_streak'] = df['hl_check'].rolling(5).sum()

        df['price_slope'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)

        # Se o volume de hoje for maior que o de ontem, e o de ontem maior que o do dia anterior
        df['volume_ramp_up'] = (
                (df['volume'] > df['volume'].shift(1)) &
                (df['volume'].shift(1) > df['volume'].shift(2))
        ).astype(int)

        cols_model = [
            'atr',
            'atr_norm',
            # 'rsi',  # Nível atual
            # 'rsi_ema',
            'rsi_cross_gap',

            # --- TREND (Estrutura) ---
            # 'ema_spread',  # Spread 9 vs 21
            'gap_50_200',  # Direção macro (O oceano)

            # --- QUALIDADE (Anti-Lixo) ---
            # 'efficiency_ratio',  # Filtro de ruído
            # 'bb_pband',  # Posição relativa às bandas (Para não comprar o topo)
            # 'price_ema_gap',
            'adx',
            'bb_width',

            # 'dist_to_support',
            # 'bb_width_norm'
            'volume_ratio',
            # 'squeeze_index',
            'macro_trend_alignment',
            'macro_momentum',
            # 'higher_lows_streak',
            'price_slope',
            # 'volume_ramp_up'
        ]

        # 1. Primeiro normalizas a largura das bandas pelo preço (essencial para o caldeirão multi-par)
        # df['bb_width_norm'] = (df['bb_high'] - df['bb_low']) / df['close']

        # 2. Calcula a taxa de expansão (se for positivo, estão a abrir; se for negativo, a fechar)
        df['bb_expansion_rate'] = (df['bb_width_norm'] - df['bb_width_norm'].shift(1)) / df['bb_width_norm'].shift(1)
        df['bb_expansion_rate'] = df['bb_expansion_rate'].fillna(0)

        """
        cols_model = [
            'atr',
            # 'atr_norm',
            # 'rsi',  # Nível atual
            # 'rsi_ema',
            'rsi_cross_gap',
            # 'gap_50_200',
            # 'price_ema_gap',
            # 'bb_expansion_rate'
            # 'ema_spread'
        ]
        """

        # 1. Trata os infinitos que os indicadores podem ter gerado
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Primeiro preenchemos para a frente (ffill) os indicadores que precisam de tempo
        df.ffill(inplace=True)
        # Depois preenchemos com 0 o início do dataset onde não há histórico anterior
        df.fillna(0, inplace=True)

        cols_to_exclude = ['st_lowerband', 'st_upperband', 'atr',
                           'label']
        cols_train = [c for c in cols_model if c not in cols_to_exclude]
        if is_training:

            df_discovery = df.iloc[:int(len(df) * 0.8)].copy()
            best_w, best_p, _ = MarketBrain.find_best_params(df_discovery, cols_train)
            # window = 12

            # Definimos o alvo real que o teu bot persegue
            # min_profit_pct = 0.02  # 2%

            window = best_w
            min_profit_pct = best_p

            # low_volatility = custom_indicators.detect_low_volatility(slope_threshold=min_profit_pct)

            if 'label' not in df.columns:
                df['label'] = 1  # Inicializa tudo como HOLD (1)

            for i in range(len(df) - window):
                # 1. FILTROS DE VETO (SMART LABELING)
                # is_low_volatility = low_volatility[i]

                # Se o mercado estiver "morto", forçamos HOLD e passamos à frente
                """
                if is_low_volatility:  # or current_er < 0.30 or not is_volatily_ok:
                    df.iloc[i, df.columns.get_loc('label')] = 1
                    continue
                """
                immediate_slope = (df['close'].iloc[i] - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1]

                price_entry = df['close'].iloc[i]

                # Take Profit baseado no min_profit_pct do discovery ou Rácio RR
                target_buy = price_entry * (1 + min_profit_pct)
                target_sell = price_entry * (1 - min_profit_pct)

                future_segment = df.iloc[i + 1: i + window + 1]
                max_high = future_segment['high'].max()
                min_low = future_segment['low'].min()

                folga = 0.0005
                prev_low = df['low'].iloc[i - 1]
                prev_high = df['high'].iloc[i - 1]

                # 3. LÓGICA DE COMPRA
                # Condição: Bateu no alvo ANTES de furar a Supertrend (sl_price)
                if immediate_slope > 0.001:
                    # sl_price = df['st_lowerband'].iloc[i]
                    sl_price = prev_low * (1.0 - folga)
                    if (max_high >= target_buy) and (min_low > sl_price):
                        df.iloc[i, df.columns.get_loc('label')] = 2

                # 4. LÓGICA DE VENDA
                elif immediate_slope < -0.001:
                    # sl_price = df['st_upperband'].iloc[i]
                    sl_price = prev_high * (1.0 + folga)
                    if (min_low <= target_sell) and (max_high < sl_price):
                        df.iloc[i, df.columns.get_loc('label')] = 0

                else:
                    df.iloc[i, df.columns.get_loc('label')] = 1

        return df, df[cols_train]

    @staticmethod
    def add_indicators(df: pd.DataFrame, is_training: bool = False):
        df = df.copy()

        # 🚨 1. GARANTIR ÍNDICE LIMPO LOGO À ENTRADA
        df.reset_index(drop=True, inplace=True)

        df_temp = df.copy()
        if pd.api.types.is_datetime64_any_dtype(df_temp['timestamp']):
            df_temp['timestamp'] = df_temp['timestamp'].astype('int64') // 10 ** 6

        # --- INDICADORES BASE ---
        custom_indicators = CustomIndicatorsUtils(OhlcvWrapper(df_temp.values.tolist()), mode='custom')
        tv_indicators = TvIndicatorsUtils(OhlcvWrapper(df_temp.values.tolist()), mode='custom')
        supertrend, trend, final_upperband, final_lowerband, supertrend_smooth, direction, perf_score = custom_indicators.supertrend()

        df['squeeze_index'] = tv_indicators.squeeze_index()
        df['st_direction'] = direction
        df['st_lowerband'] = final_lowerband
        df['st_upperband'] = final_upperband

        df_pivots = MarketBrain.add_luxalgo_pivots(df)
        df['dist_to_lux_pivot_high'] = df_pivots['dist_to_lux_pivot_high']
        df['dist_to_lux_pivot_low'] = df_pivots['dist_to_lux_pivot_low']

        # --- INDICADORES COMPLEMENTARES ---
        df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
        df['fast_rsi'] = ta.momentum.rsi(df['close'], window=9)
        df['macro_rsi'] = ta.momentum.rsi(df['close'], window=25)

        df['roc_fast'] = df['close'].pct_change(periods=3) * 100
        df['roc_mid'] = df['close'].pct_change(periods=7) * 100

        volume_ma = df['volume'].rolling(20).mean()
        df['volume_climax'] = df['volume'] / np.where(volume_ma == 0, 1, volume_ma)

        # --- INVOCAR O TEU MOTOR SMART MONEY CHANNELS ---
        smc_results = tv_indicators.smart_money_breakout_channels(
            length_norm=100,
            length_box=14,
            strong=True,
            overlap=False
        )

        # Injetar os resultados estruturados diretamente no DataFrame
        df['line_high_val'] = smc_results['top']
        df['line_low_val'] = smc_results['bottom']
        df['breakout_up'] = smc_results['bull_break'].astype(int)
        df['breakout_down'] = smc_results['bear_break'].astype(int)
        df['new_channel'] = smc_results['new_channel'].astype(int)
        df['box_duration'] = smc_results['duration']

        # --- ENGENHARIA DE FEATURES PARA O LIGHTGBM ---
        df['line_high_val'] = df['line_high_val'].ffill()
        df['line_low_val'] = df['line_low_val'].ffill()

        df['line_high_val'] = df['line_high_val'].fillna(df['close'])
        df['line_low_val'] = df['line_low_val'].fillna(df['close'])

        df['wedge_formation'] = df['line_high_val'] - df['line_low_val']
        df['dist_to_upper_trendline'] = (df['line_high_val'] - df['close']) / df['close']
        df['dist_to_lower_trendline'] = (df['close'] - df['line_low_val']) / df['close']

        # ==========================================================
        # NOVAS DIMENSÕES DE ANÁLISE (ALÉM DOS BREAKOUTS)
        # ==========================================================

        # 1. Dimensão de Exaustão / Reversão à Média (Bandas de Bollinger)
        indicator_bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
        df["bb_high"] = indicator_bb.bollinger_hband()
        df["bb_low"] = indicator_bb.bollinger_lband()
        # Calculamos a distância percentual às bandas (esta é a feature real para o ML)
        df["dist_to_bb_high"] = (df["bb_high"] - df["close"]) / df["close"]
        df["dist_to_bb_low"] = (df["close"] - df["bb_low"]) / df["close"]
        df["bb_width"] = indicator_bb.bollinger_wband()
        df["bb_expansion"] = df["bb_width"] / df["bb_width"].shift(1)

        # 2. Dimensão de Força de Tendência (ADX)
        df["adx"] = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14).adx()
        df["adx"] = df["adx"].fillna(20)  # Tratar NaNs iniciais do ADX

        # 3. Dimensão de Fluxo de Dinheiro Real (Chaikin Money Flow)

        df["cmf"] = ChaikinMoneyFlowIndicator(high=df["high"], low=df["low"], close=df["close"],
                                              volume=df["volume"], window=20).chaikin_money_flow()
        df["cmf"] = df["cmf"].fillna(0)

        # --- DIMENSÃO INSTITUCIONAL (Filtro de Tendência Macro Sol/Eth) ---
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)

        # 🌟 FEATURE 1: O Alinhamento Macro (O Filtro Visual do Print)
        # Se der > 0, o mercado está em Bull Market Macro. Se der < 0, Bear Market agressivo.
        df['macro_trend_alignment'] = (df['ema_50'] - df['ema_200']) / df['close'] * 100

        # 🌟 FEATURE 2: Distância do Preço para a EMA 200 (A Gravidade)
        # Mostra ao modelo se o preço está demasiado esticado longe da média institucional
        df['dist_to_ema_200'] = (df['close'] - df['ema_200']) / df['close'] * 100
        df['dist_to_ema_50'] = (df['close'] - df['ema_50']) / df['close'] * 100

        # ==========================================
        # TOQUE DE INTELIGÊNCIA 1: FEATURES DE MOMENTUM (LAG)
        # ==========================================
        for lag in [1, 3, 5]:
            df[f'dist_upper_lag_{lag}'] = df['dist_to_upper_trendline'].shift(lag)
            df[f'dist_low_lag_{lag}'] = df['dist_to_lower_trendline'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['macro_rsi'].shift(lag)
            df[f'adx_lag_{lag}'] = df['adx'].shift(lag)
            df[f'cmf_lag_{lag}'] = df['cmf'].shift(lag)
            df[f'bb_high_lag_{lag}'] = df['dist_to_bb_high'].shift(lag)
            df[f'bb_low_lag_{lag}'] = df['dist_to_bb_low'].shift(lag)

        # 🚀 CORREÇÃO DO BUG EM PRODUÇÃO:
        # Se for treino, limpamos os NaNs históricos do início do gráfico (as primeiras 25 linhas)
        if is_training:
            df.dropna(subset=['dist_upper_lag_5', 'atr', 'macro_rsi'], inplace=True)
            df.reset_index(drop=True, inplace=True)
        else:
            # Em produção, NÃO eliminamos linhas. Usamos Forward Fill para garantir que o ruído
            # de cálculo ou atrasos na API não apagam o nosso candle de decisão atual.
            df.ffill(inplace=True)
            # Se ainda sobrarem NaNs absolutos no início da janela de produção, metemos zero
            df.fillna(0, inplace=True)

        # Lista final de colunas de treino (Adiciona aqui 'squeeze_index' ou 'st_direction' se quiseres que o modelo as use)
        cols_base = [
            'wedge_formation', 'dist_to_upper_trendline', 'dist_to_lower_trendline',
            'breakout_up', 'breakout_down', 'volume_climax', 'macro_rsi', 'fast_rsi', 'atr',
            'new_channel', 'box_duration', 'roc_fast', 'roc_mid', 'cmf', 'adx', 'bb_expansion',
            'dist_to_lux_pivot_high', 'dist_to_lux_pivot_low'
        ]

        # Os teus lags já incluem os novos indicadores (BB, ADX, CMF). Está perfeito!
        cols_lags = [f'dist_upper_lag_{l}' for l in [1, 3, 5]] + \
                    [f'dist_low_lag_{l}' for l in [1, 3, 5]] + \
                    [f'rsi_lag_{l}' for l in [1, 3, 5]] + \
                    [f'adx_lag_{l}' for l in [1, 3]] + \
                    [f'cmf_lag_{l}' for l in [1, 3]] + \
                    [f'bb_high_lag_{l}' for l in [1, 3]] + \
                    [f'bb_low_lag_{l}' for l in [1, 3]]

        cols_train = cols_base + cols_lags

        # 🚨 SOLUÇÃO DO BUG DE NANs: Garantir que a coluna de pesos existe SEMPRE
        df['sample_weight'] = 1.0
        df['label'] = 1  # Default: HOLD

        # --- SMART LABELING ---
        if is_training:
            df_discovery = df.iloc[:int(len(df) * 0.8)].copy()
            best_w, best_p, _ = MarketBrain.find_best_params(df_discovery, cols_train)

            window = best_w
            min_profit_pct = best_p

            for i in range(len(df) - window):
                price_entry = df['close'].iloc[i]
                target_buy = price_entry * (1 + min_profit_pct)
                target_sell = price_entry * (1 - min_profit_pct)

                future_segment = df.iloc[i + 1: i + window + 1]
                max_high = future_segment['high'].max()
                min_low = future_segment['low'].min()

                folga = 0.0005
                prev_low = df['low'].iloc[i - 1]
                prev_high = df['high'].iloc[i - 1]

                sl_price_buy = prev_low * (1.0 - folga)
                sl_price_sell = prev_high * (1.0 + folga)

                move_up = (max_high - price_entry) / price_entry
                move_down = (price_entry - min_low) / price_entry

                # 🚨 CORREÇÃO CRÍTICA: Labeling Puro de Alvo vs Stop (A rampa imediata foi removida daqui)
                # Se atingir o Alvo de Alta antes de violar o Stop anterior, é COMPRA (2)
                if (max_high >= target_buy) and (min_low > sl_price_buy):
                    df.iloc[i, df.columns.get_loc('label')] = 2
                    # Atribui o peso proporcional à força da subida real
                    df.iloc[i, df.columns.get_loc('sample_weight')] = 1.0 + (move_up * 100)

                # Se atingir o Alvo de Baixa antes de violar o Stop anterior, é VENDA (0)
                elif (min_low <= target_sell) and (max_high < sl_price_sell):
                    df.iloc[i, df.columns.get_loc('label')] = 0
                    # Atribui o peso proporcional à força da queda real
                    df.iloc[i, df.columns.get_loc('sample_weight')] = 1.0 + (move_down * 100)

        return df, df[cols_train]

    @staticmethod
    def find_best_params(df: pd.DataFrame, features: list):
        import lightgbm as lgb
        from sklearn.metrics import f1_score
        import numpy as np

        # --- 1. CONFIGURAÇÃO DA BUSCA ---
        # profit_test = [0.015, 0.02, 0.025, 0.030]
        profit_test = [0.01, 0.015, 0.02, 0.025]
        window_test = [8, 10, 12, 16, 20, 24]

        # profit_test = [0.015, 0.025, 0.035, 0.045]  # Alvos de até 4.5% para aproveitar as explosões da SOL
        # window_test = [16, 24, 36, 48]

        best_f1 = -1
        best_p = 0.02
        best_w = 12

        # Extraímos os valores para arrays NumPy para velocidade extrema (e evitar overhead de memória)
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        atr_values = df['atr'].values

        n_total = len(df)

        print("🔍 Iniciando Busca de Hiper-parâmetros de Labeling...")

        for p_test in profit_test:
            for w_test in window_test:
                # Criamos apenas um array de labels (economiza RAM comparado a copiar o DF)
                labels_temp = np.ones(n_total, dtype=int)

                # --- LOOP DE LABELING OTIMIZADO (NUMPY SLICING) ---
                for i in range(n_total - w_test):
                    price_entry = close_prices[i]
                    target_buy = price_entry * (1 + p_test)
                    target_sell = price_entry * (1 - p_test)

                    # Usamos 1.2x ATR para dar margem ao ruído
                    stop_buy = price_entry - (atr_values[i] * 1.2)
                    stop_sell = price_entry + (atr_values[i] * 1.2)

                    # Slices rápidos do futuro
                    f_high = high_prices[i + 1: i + w_test + 1]
                    f_low = low_prices[i + 1: i + w_test + 1]

                    # Lógica de validação do movimento
                    if (f_high.max() >= target_buy) and (f_low.min() > stop_buy):
                        labels_temp[i] = 2
                    elif (f_low.min() <= target_sell) and (f_high.max() < stop_sell):
                        labels_temp[i] = 0

                # --- TREINO RÁPIDO DE VALIDAÇÃO ---
                # Temporariamente adicionamos ao DF original para facilitar o split
                df['temp_label'] = labels_temp

                # Removemos NaNs apenas para o treino
                df_clean = df.dropna(subset=features + ['temp_label'])

                split = int(len(df_clean) * 0.8)
                train_data = df_clean.iloc[:split]
                val_data = df_clean.iloc[split:]

                # Verificação de segurança: precisamos das 3 classes para o modelo aprender algo útil
                if len(train_data['temp_label'].unique()) < 3:
                    df.drop(columns=['temp_label'], inplace=True)
                    continue

                # Modelo ultra-rápido para teste (árvores rasas)
                clf = lgb.LGBMClassifier(
                    n_estimators=50,
                    learning_rate=0.1,
                    max_depth=4,
                    num_leaves=15,
                    verbose=-1
                )

                clf.fit(train_data[features], train_data['temp_label'])

                # Avaliação
                preds = clf.predict(val_data[features])

                # 1. Focamos no sucesso das operações Reais (Buy=0 e Sell=2)
                f1 = f1_score(val_data['temp_label'], preds, labels=[0, 2], average='macro', zero_division=0)

                # 2. Contagem de sinais REAIS gerados pelo labeling no set de treino e validação
                real_signals_train = (train_data['temp_label'] != 1).sum()
                num_predictions = (preds != 1).sum()  # Quantos sinais o modelo disparou na validação

                print(
                    f"  > Alvo {p_test * 100}% | Janela {w_test}: F1={f1:.4f} | Sinais Reais Treino={real_signals_train} | Predições={num_predictions}")

                # ----------------------------------------------------------------
                # CRITÉRIO QUANT DE SEGURANÇA:
                # Exigimos pelo menos 200 sinais reais no treino para o modelo ser confiável
                # E exigimos que o modelo faça pelo menos 15 predições para avaliar a precisão
                # ----------------------------------------------------------------
                if real_signals_train >= 200 and num_predictions >= 15:
                    if f1 > best_f1:
                        best_f1 = f1
                        best_p = p_test
                        best_w = w_test

                # Limpeza da coluna temporária para o próximo loop
                df.drop(columns=['temp_label'], inplace=True)

        print(f"✅ Melhor combinação encontrada: Lucro {best_p * 100}% em {best_w} velas (F1: {best_f1:.4f})")
        return best_w, best_p, best_f1

    @staticmethod
    def add_luxalgo_pivots(df, length=50):
        # 1. Encontrar Topos e Fundos usando apenas dados passados (Rolling Windows)
        # O sinal é deslocado por 'length' para simular o atraso real do mercado ao vivo
        df['rolling_max'] = df['high'].shift(length).rolling(window=length * 2 + 1, center=True).max()
        df['rolling_min'] = df['low'].shift(length).rolling(window=length * 2 + 1, center=True).min()

        # 2. Registar o preço exato do último pivô confirmado (sem olhar para o futuro)
        # Criamos uma estrutura que "carrega" o último nível para as velas da frente
        last_ph = np.nan
        last_pl = np.nan
        ph_series = []
        pl_series = []

        for i in range(len(df)):
            # Se o preço de há 'length' velas atrás era o maior da janela de 100 velas
            if df['high'].iloc[max(0, i - length)] == df['rolling_max'].iloc[max(0, i - length)]:
                last_ph = df['high'].iloc[max(0, i - length)]
            if df['low'].iloc[max(0, i - length)] == df['rolling_min'].iloc[max(0, i - length)]:
                last_pl = df['low'].iloc[max(0, i - length)]

            ph_series.append(last_ph)
            pl_series.append(last_pl)

        df['lux_last_pivot_high'] = ph_series
        df['lux_last_pivot_low'] = pl_series

        # 3. Métrica de Ouro para o LightGBM: Distância percentual até ao pivô
        # É isto que o modelo vai ler para saber se está perto de uma reversão fantasma
        df['dist_to_lux_pivot_high'] = (df['lux_last_pivot_high'] - df['close']) / df['close']
        df['dist_to_lux_pivot_low'] = (df['close'] - df['lux_last_pivot_low']) / df['close']

        # Limpar colunas temporárias para não carregar o dataset
        df.drop(columns=['rolling_max', 'rolling_min'], inplace=True)

        return df
