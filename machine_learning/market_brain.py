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
    def add_indicators(df: pd.DataFrame, is_training: bool = False):
        df = df.copy()
        df_temp = df.copy()
        if pd.api.types.is_datetime64_any_dtype(df_temp['timestamp']):
            df_temp['timestamp'] = df_temp['timestamp'].astype('int64') // 10 ** 6

        custom_indicators = CustomIndicatorsUtils(OhlcvWrapper(df_temp.values.tolist()), mode='custom')
        tv_indicators = TvIndicatorsUtils(OhlcvWrapper(df_temp.values.tolist()), mode='custom')
        supertrend, trend, final_upperband, final_lowerband, supertrend_smooth, direction, perf_score = custom_indicators.supertrend()

        low_volatility = custom_indicators.detect_low_volatility()
        # df['low_volatility'] = low_volatility

        df['st_direction'] = direction
        df['st_lowerband'] = final_lowerband
        df['st_upperband'] = final_upperband

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

        df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema100'] = ta.trend.ema_indicator(df['close'], window=100)
        df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)

        # Gaps das Médias Longas (Distância)
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

        # psar_df = ta.trend.PSARIndicator(df['high'], df['low'], df['close'], step=0.02, max_step=0.2)
        # df['psar'] = psar_df.psar()

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

        cols_model = [
            'atr',
            # --- MOMENTUM (Gatilhos Rápidos) ---
            # 'ema9_slope',  # Inclinação para entrar ANTES do cross
            # 'rsi_velocity',  # Rapidez da mudança de força
            'rsi',  # Nível atual
            'rsi_ema',

            # --- TREND (Estrutura) ---
            # 'ema_spread',  # Spread 9 vs 21
            'gap_50_200',  # Direção macro (O oceano)

            # --- QUALIDADE (Anti-Lixo) ---
            'efficiency_ratio',  # Filtro de ruído
            'bb_pband',  # Posição relativa às bandas (Para não comprar o topo)
            # 'downward_pressure',
            'price_ema_gap',
            'adx',
            # 'adx_slope',
            'bb_width',
            # 'psar',
            # 'st_direction',
            'st_lowerband',
            'st_upperband',
            # 'squeeze_index',
            # 'andean_diff',
            # 'andean_neutral_gap',
            # 'low_volatility'
        ]

        # 1. Trata os infinitos que os indicadores podem ter gerado
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Primeiro preenchemos para a frente (ffill) os indicadores que precisam de tempo
        df.ffill(inplace=True)
        # Depois preenchemos com 0 o início do dataset onde não há histórico anterior
        df.fillna(0, inplace=True)

        cols_to_exclude = ['st_lowerband', 'st_upperband', 'psar',
                           'label', 'atr', 'low_volatility']
        cols_train = [c for c in cols_model if c not in cols_to_exclude]
        if is_training:

            df_discovery = df.iloc[:int(len(df) * 0.8)].copy()
            best_w, best_p, _ = MarketBrain.find_best_params(df_discovery, cols_train)
            # window = 12

            # Definimos o alvo real que o teu bot persegue
            # min_profit_pct = 0.02  # 2%

            window = best_w
            min_profit_pct = best_p

            low_volatility = custom_indicators.detect_low_volatility(slope_threshold=min_profit_pct)

            # can_buy_series, can_sell_series = MarketBrain.get_consolidated_signals(df)

            if 'label' not in df.columns:
                df['label'] = 1  # Inicializa tudo como HOLD (1)

            for i in range(len(df) - window):
                # 1. FILTROS DE VETO (SMART LABELING)
                # current_adx = df['adx'].iloc[i]
                # current_er = df['efficiency_ratio'].iloc[i]
                # is_volatily_ok = bool(df['bb_width'].iloc[i] > 0.008)
                # is_low_volatility = df['low_volatility'].iloc[i]
                is_low_volatility = low_volatility[i]

                # NOVO: Filtro de Direção pela Supertrend (st_direction: 1 Alta, -1 Baixa)
                # is_bullish_regime = bool(df['st_direction'].iloc[i] == 1)
                # is_bearish_regime = bool(df['st_direction'].iloc[i] == -1)

                # Filtros de Momentum mantidos
                # momentum_up = bool(df['rsi'].iloc[i] > df['rsi_ema'].iloc[i])
                # momentum_down = bool(df['rsi'].iloc[i] < df['rsi_ema'].iloc[i])

                immediate_slope = (df['close'].iloc[i] - df['close'].iloc[i - 3]) / df['close'].iloc[i - 3]

                # Se o mercado estiver "morto", forçamos HOLD e passamos à frente
                if is_low_volatility:  # or current_er < 0.30 or not is_volatily_ok:
                    df.iloc[i, df.columns.get_loc('label')] = 1
                    continue

                price_entry = df['close'].iloc[i]

                # Take Profit baseado no min_profit_pct do discovery ou Rácio RR
                target_buy = price_entry * (1 + min_profit_pct)
                target_sell = price_entry * (1 - min_profit_pct)

                future_segment = df.iloc[i + 1: i + window + 1]
                max_high = future_segment['high'].max()
                min_low = future_segment['low'].min()

                # Pegamos o sinal do Agrupador para este índice
                # final_buy = can_buy_series.iloc[i]
                # final_sell = can_sell_series.iloc[i]

                # found_signal = False

                # 3. LÓGICA DE COMPRA
                # Condição: Bateu no alvo ANTES de furar a Supertrend (sl_price)
                if immediate_slope > 0.001:
                    sl_price = df['st_lowerband'].iloc[i]
                    if (max_high >= target_buy) and (min_low > sl_price):
                        # df.at[df.index[i], 'label'] = 2
                        df.iloc[i, df.columns.get_loc('label')] = 2
                        found_signal = True

                # 4. LÓGICA DE VENDA
                elif immediate_slope < -0.001:
                    sl_price = df['st_upperband'].iloc[i]
                    if (min_low <= target_sell) and (max_high < sl_price):
                        df.iloc[i, df.columns.get_loc('label')] = 0

                else:
                    df.iloc[i, df.columns.get_loc('label')] = 1

        return df, df[cols_train]

    @staticmethod
    def get_consolidated_signals(df):
        """
        Versão VETORIAL: Usa & em vez de 'and' para processar o DataFrame todo.
        """
        # Condições de Compra (PSAR + RSI)
        # É OBRIGATÓRIO usar parênteses em cada comparação individual
        buy_psar = (df['close'] > df['psar']) & (df['rsi'] > 50) & (df['rsi'] > df['rsi_ema'])

        # Condições de Venda (PSAR + RSI)
        sell_psar = (df['close'] < df['psar']) & (df['rsi'] < 50) & (df['rsi'] < df['rsi_ema'])

        # Se quiseres adicionar a Supertrend personalizada aqui:
        # final_buy = buy_psar & (df['close'] > df['st_lower'])
        # final_sell = sell_psar & (df['close'] < df['st_upper'])

        return buy_psar, sell_psar

    @staticmethod
    def find_best_params(df: pd.DataFrame, features: list):
        import lightgbm as lgb
        from sklearn.metrics import f1_score
        import numpy as np

        # --- 1. CONFIGURAÇÃO DA BUSCA ---
        profit_test = [0.01, 0.015, 0.02, 0.025]
        window_test = [8, 10, 12, 16]

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

                # Focamos no sucesso das operações Reais (Buy e Sell)
                f1 = f1_score(val_data['temp_label'], preds, labels=[0, 2], average='macro', zero_division=0)
                num_signals = (preds != 1).sum()

                print(f"  > Alvo {p_test * 100}% | Janela {w_test}: F1={f1:.4f} | Sinais={num_signals}")

                # Critério de seleção: melhor F1 com um mínimo de sinais para evitar sorte
                if f1 > best_f1 and num_signals > 5:
                    best_f1 = f1
                    best_p = p_test
                    best_w = w_test

                # Limpeza da coluna temporária para o próximo loop
                df.drop(columns=['temp_label'], inplace=True)

        print(f"✅ Melhor combinação encontrada: Lucro {best_p * 100}% em {best_w} velas (F1: {best_f1:.4f})")
        return best_w, best_p, best_f1

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
