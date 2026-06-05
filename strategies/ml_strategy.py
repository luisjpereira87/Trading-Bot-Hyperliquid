import json
import logging
import os
from typing import List

import joblib
import numpy as np
import pandas as pd
import ta.momentum

from commons.enums.mode_enum import ModeEnum
from commons.models.metadata_dclass import Metadata
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from commons.utils.paths import get_model_path, get_scaler_path, get_bayesian_path, get_metadat_json_path
from machine_learning.market_brain import MarketBrain
from trading_bot.exchange_base import ExchangeBase

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ou '3'
from tensorflow.keras.models import load_model  # type: ignore

from commons.enums.ml_model_enum import MLModelType
from commons.enums.signal_enum import Signal
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from machine_learning.ml_train_pipeline import MLTrainer


class MLStrategy(StrategyBase):
    def __init__(self, exchange: ExchangeBase, model_type=MLModelType.RANDOM_FOREST, is_combined_model=True):
        super().__init__()

        self.exchange = exchange
        self.aggressive_mode = ModeEnum.CONSERVATIVE
        self.model_type = model_type
        self.ohlcv: OhlcvWrapper
        self.symbol = None
        self.model = None
        self.bayesian_model = None
        self.scaler = None
        self.last_train_len = 0
        self.confidence_threshold = 0.3
        self.price_ref: float = 0.0
        self.exchange_name = None

        self.model_loaded = False
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, "trading-bot", "machine_learning", "models")
        os.makedirs(self.model_dir, exist_ok=True)
        # self.model_path = get_model_path(self.model_type.value, self.exchange.get_name(), ".pkl")
        # self.keras_model_path = get_model_path(self.model_type.value, self.exchange.get_name(), ".keras")
        # self.scaler_model_path = get_scaler_path(self.model_type.value, self.exchange.get_name())
        self.data_dir = "data"
        self.image_path = "img/imagem.png"
        # self.BAYESIAN_PATH = get_bayesian_path(self.exchange.get_name())

        self.is_combined_model = is_combined_model

        self.metadata: (Metadata | None) = None

    def required_init(self, ohlcv: OhlcvWrapper, ohlcv_higher: (OhlcvWrapper | None), symbol: str, price_ref: float):
        self.ohlcv = ohlcv
        self.symbol = symbol
        self.ohlcv_higher = ohlcv_higher
        self.price_ref = price_ref

    def set_params(self, params: StrategyParams):
        self.mode = params.mode
        self.multiplier = params.multiplier

    def set_candles(self, ohlcv):
        self.ohlcv = ohlcv

    def set_higher_timeframe_candles(self, ohlcv_higher: List[list]):
        # TODO document why this method is empty
        pass

    def get_sl_tp(self):
        # TODO document why this method is empty
        pass

    async def initialize(self, model_type: MLModelType):
        if self.model_loaded:
            return

        symbol_clean = self.symbol.replace("/", "_").replace(":", "_")
        if self.is_combined_model:
            keras_model_path = get_model_path(self.model_type.value, self.exchange_name, "MASTER", ".keras")
            model_path = get_model_path(self.model_type.value, self.exchange_name, "MASTER", ".pkl")
            scaler_model_path = get_scaler_path(self.model_type.value, self.exchange_name, "MASTER")
            metadata = get_metadat_json_path(self.model_type.value, self.exchange_name, "MASTER")
        else:
            keras_model_path = get_model_path(self.model_type.value, self.exchange_name, symbol_clean, ".keras")
            model_path = get_model_path(self.model_type.value, self.exchange_name, symbol_clean, ".pkl")
            scaler_model_path = get_scaler_path(self.model_type.value, self.exchange_name, symbol_clean)
            metadata = get_metadat_json_path(self.model_type.value, self.exchange_name, symbol_clean)

        # Define qual o caminho do ficheiro principal para validar a data
        target_path = keras_model_path if model_type == MLModelType.LSTM else model_path

        # Lógica de Re-treino: Existe e tem menos de 7 dias?
        model_exists = os.path.exists(target_path)

        # Se o modelo existe e é recente, carregamos
        if model_exists and metadata and metadata.is_fresh:
            if model_type == MLModelType.LSTM:
                self.model = load_model(keras_model_path)
                logging.info("📥 Modelo LSTM carregado e atualizado.")
            else:
                self.model = joblib.load(model_path)
                # self.bayesian_model = self.load_bayesian_model()
                logging.info(f"📥 Modelo {self.model_type.value} carregado e atualizado.")

            self.scaler = joblib.load(scaler_model_path)
            self.metadata = metadata
            self.model_loaded = True

        # Caso contrário (não existe ou está expirado), treinamos
        else:
            reason = "inexistente" if not model_exists else "expirado"
            logging.warning(f"⚠️ Modelo {model_type.value} {reason}. A executar MLTrainer...")

            mlTrainer = MLTrainer(self.exchange, model_type, False, False)
            await mlTrainer.run()

            # Após o treino, carregamos os novos ficheiros
            if model_type == MLModelType.LSTM:
                self.model = load_model(keras_model_path)
            else:
                self.model = joblib.load(model_path)
                # self.bayesian_model = self.load_bayesian_model()

            if self.is_combined_model:
                self.metadata = get_metadat_json_path(self.model_type.value, self.exchange_name, "MASTER")
            else:
                self.metadata = get_metadat_json_path(self.model_type.value, self.exchange_name, symbol_clean)

            self.scaler = joblib.load(scaler_model_path)
            self.model_loaded = True
            logging.info(f"✅ Modelo {model_type.value} treinado e carregado com sucesso.")

    def load_bayesian_model(self):
        symbol_clean = self.symbol.replace("/", "_").replace(":", "_")
        bayesian_path = get_bayesian_path(self.exchange.get_name(), symbol_clean)
        # 1. Debug do Path
        print(f"🔍 A tentar carregar de: {bayesian_path}")

        # 2. Verificação de segurança do Path
        if not bayesian_path or not os.path.exists(bayesian_path):
            logging.error(f"❌ Path Inválido ou Inexistente: {bayesian_path}")
            return None

        # 3. Singleton Pattern: Só carrega se ainda não estiver na memória
        if bayesian_path is not None:
            try:
                with open(bayesian_path, 'r') as f:
                    # IMPORTANTE: Guardar no self para as próximas chamadas!
                    self.bayesian_model = json.load(f)
                    logging.info("🧠 Supervisor Bayesiano carregado com sucesso.")
            except Exception as e:
                logging.error(f"❌ Erro ao ler ficheiro JSON: {e}")
                self.bayesian_model = None

        return self.bayesian_model

    def create_lstm_sequences(self, df: pd.DataFrame, window_size: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Converte o DataFrame em janelas sequenciais para LSTM.
        Retorna X (3D numpy array) e y (1D array).
        """
        df = self.calculate_features(df).dropna()

        # Criar labels (exemplo): 2=buy, 1=hold, 0=sell baseado em retorno futuro (3 períodos)
        df['future_return'] = df['close'].shift(-3) / df['close'] - 1
        conditions = [
            (df['future_return'] > 0.003),
            (df['future_return'] < -0.003)
        ]
        choices = [2, 0]
        df['label'] = np.select(conditions, choices, default=1)
        df = df.dropna()

        features = df[['rsi', 'atr', 'macd', 'pct_change', 'ema9', 'ema21', 'stoch_k', 'stoch_d']].values
        labels = df['label'].values

        X, y = [], []
        for i in range(len(features) - window_size):
            X.append(features[i:i + window_size])
            y.append(labels[i + window_size])
        X = np.array(X)
        y = np.array(y)
        return X, y

    def fetch_ohlcv(self, data: OhlcvWrapper) -> pd.DataFrame:
        df = pd.DataFrame(data.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df, features = MarketBrain.add_indicators(df)

        return df

    def compute_sl_tp(self, price: float, atr: float, confidence: float, direction: Signal) -> tuple[
        float | None, float | None]:
        if direction == Signal.HOLD or direction is None:
            return None, None

        # 1. Inverter a Confiança: Se confio muito, aceito menos erro (Stop mais curto)
        # Se confidence for 1.0 (máx), fator é 0.8. Se for 0.0 (mín), fator é 1.2.
        risk_factor = 1.2 - (confidence * 0.4)

        # 2. Distâncias Realistas (Equilibrar a Respiração)
        # Subimos o SL para 2.0 * ATR para sair do ruído (deixar a SOL respirar)
        # Baixamos o TP para 2.5 * ATR (mais fácil de atingir em reversões)
        sl_distance = atr * 2.0 * risk_factor
        tp_distance = atr * 2.5 * risk_factor

        # 3. Hard Cap Adaptativo por Ativo
        # Em vez de 1.5% fixo, podes usar 2.5% para SOL/ETH e 1.2% para BTC
        # Ou simplesmente confiar mais no ATR e menos no % fixo.
        max_sl_pct = 0.025 if "SOL" in self.symbol else 0.015
        sl_distance = min(sl_distance, price * max_sl_pct)

        if direction == Signal.BUY:
            sl, tp = price - sl_distance, price + tp_distance
        else:
            sl, tp = price + sl_distance, price - tp_distance

        return round(float(sl), 4), round(float(tp), 4)

    def supertrend_sl_tp_(self, price: float, lowerband: float, upperband: float, signal: Signal):

        if signal == Signal.BUY:
            sl = lowerband  # SL no ponto mais baixo da banda
            tp = upperband + (upperband - sl) * 0.5

        elif signal == Signal.SELL:
            sl = upperband  # SL no ponto mais alto da banda
            tp = lowerband - (sl - lowerband) * 0.5

        risk = abs(price - sl)
        reward = abs(tp - price)

        # RR Mínimo aceitável (ex: 1:1.2 ou 1:1.5)
        min_rr = 1.2

        if reward < (risk * min_rr):
            # Em vez de encurtar o SL (perigoso), esticamos o TP
            # ou invalidamos o trade se estivermos muito perto da banda
            tp = price + (risk * min_rr) if signal == Signal.BUY else price - (risk * min_rr)

            # Validar se o novo TP faz sentido contra a resistência (opcional)
            # Se o novo TP for absurdamente longe, talvez seja melhor não entrar.

        return round(float(sl), 4), round(float(tp), 4)

    def supertrend_sl_tp_(self, price: float, lowerband: float, upperband: float, signal: Signal):
        # Se o sinal for HOLD, nem vale a pena calcular nada
        if signal == Signal.HOLD:
            return None, None

        # 1. Definir os alvos originais baseados nas bandas da Supertrend
        if signal == Signal.BUY:
            sl = lowerband  # SL na banda inferior (suporte)
            tp = upperband + (upperband - sl) * 0.5

        elif signal == Signal.SELL:
            sl = upperband  # SL na banda superior (resistência)
            tp = lowerband - (sl - lowerband) * 0.5
        else:
            return None, None

        # 2. Calcular o Risco e o Retorno brutos do trade
        risk = abs(price - sl)
        reward = abs(tp - price)

        # Evitar divisão por zero se as bandas estiverem literalmente em cima do preço
        if risk == 0:
            return None, None

        # 3. FILTRO DE ELITE: Validar o Rácio Risco/Retorno Real
        min_rr = 1.2

        # Se o prémio (reward) não pagar o risco com a margem necessária,
        # significa que as bandas estão demasiado coladas (mercado lateral). Não entramos!
        if reward < (risk * min_rr):
            # DEBUG para saberes porque é que o bot ficou de mãos no bolso
            # print(f"  [FILTRO RR] Trade rejeitado. Reward ({reward:.2f}) menor que o Risco Mínimo ({risk * min_rr:.2f})")
            return None, None

        return round(float(sl), 4), round(float(tp), 4)

    def supertrend_sl_tp(self, price: float, df_window: pd.DataFrame, signal: Signal):
        """
        df_window: Passa as últimas velas do teu DataFrame para extrair os máximos/mínimos
        """
        if signal == Signal.HOLD or len(df_window) < 2:
            return None, None

        # Extrair os dados da vela anterior (índice -2) e da atual (-1)
        prev_low = df_window['low'].iloc[-2]
        prev_high = df_window['high'].iloc[-2]
        prev_candle_size = prev_high - prev_low
        atr_prev = df_window['atr'].iloc[-2]
        """
        MAX_CANDLE_FACTOR = 3.0
        if prev_candle_size > (atr_prev * MAX_CANDLE_FACTOR):
            # print(f"🚨 [BLOQUEIO GIGANTE] Abortado. A vela anterior (SL) é uma anomalia.")
            return None, None
        """
        # 1. DEFINIÇÃO DO STOP LOSS E TAKE PROFIT (MÉTODO PRICE ACTION ESTREITO)
        # Damos uma folga cirúrgica de 0.05% para evitar violinadas de ruído
        folga = 0.0005

        if signal == Signal.BUY:
            # O SL acompanha o preço: fica logo abaixo do mínimo da vela anterior
            sl = prev_low * (1.0 - folga)
            # O TP mantém-se agressivo para ir buscar os teus +2.0% ou mais
            # Exemplo: Alvo fixo de 2.2% para garantir que surfamos a tendência macro
            tp = price * 1.022

        elif signal == Signal.SELL:
            # O SL acompanha o preço: fica logo acima do máximo da vela anterior
            sl = prev_high * (1.0 + folga)
            # Alvo fixo de short de 2.2%
            tp = price * (1.0 - 0.022)
        else:
            return None, None

        """
        risk_percent = abs(price - sl) / price

        # O teu limite prático e direto baseado nos teus 15 USDC:
        if risk_percent > 0.005:  # 0.5%
            return None, None
        """
        # 2. VALIDAÇÃO MATEMÁTICA DO RISCO/RETORNO (R:R)
        risk = abs(price - sl)
        reward = abs(tp - price)

        if risk == 0:
            return None, None

        # Exigimos que o prémio seja pelo menos 1.5 vezes maior que o risco
        # Graças ao SL curto na vela anterior, este filtro vai aceitar muito mais trades de elite!
        min_rr = 1.5

        if reward < (risk * min_rr):
            # Se mesmo com o SL curto o risco for muito grande (vela anterior gigante), abortamos
            return None, None

        return round(float(sl), 4), round(float(tp), 4)

    def is_exhaustion_zone(self, df_atual: pd.DataFrame, signal: Signal) -> bool:
        """
        Filtro de Reversão do Luís:
        - Bloqueia Shorts em sobrevenda extrema (RSI < 30) e Longs em sobrecompra (RSI > 70).
        - Só aceita trades contra-tendência se o preço estiver LONGE das EMAs (vácuo).
        - Se o preço estiver perto ou a furar as EMAs, o contra-tendência é bloqueado.
        """
        if len(df_atual) < 200:
            return False

        df_local = df_atual.copy()

        # Garantia das EMAs (vamos usar a EMA 50 e EMA 200 para medir o vácuo)
        if 'ema_50' not in df_local.columns:
            df_local['ema_50'] = df_local['close'].ewm(span=50, adjust=False).mean()
        if 'ema_200' not in df_local.columns:
            df_local['ema_200'] = df_local['close'].ewm(span=200, adjust=False).mean()
        if 'rsi' not in df_local.columns:
            df_local['rsi'] = ta.momentum.rsi(df_local['close'], window=14)

        last_candle = df_local.iloc[-1]
        close_atual = last_candle['close']
        ema50 = last_candle['ema_50']
        ema200 = last_candle['ema_200']

        # Definir uma distância mínima de segurança para considerar "longe" (ex: 1.5% ou 2% da média)
        # Podes ajustar este threshold percentual conforme o comportamento da Solana
        distancia_minima_pct = 0.015

        # --- 🟢 VALIDAR GATILHO DE BUY (Procura Reversão para Alta) ---
        if signal == Signal.BUY:
            # 1. Proteção estática: Não comprar topo absoluto
            if last_candle['rsi'] > 70:
                return True  # 🛑 BLOQUEIA

            # Se o preço está ACIMA da EMA 200, está a favor da tendência macro. Caminho livre.
            if close_atual > ema200:
                return False  # ✅ AUTORIZA

            # Se o preço está ABAIXO da EMA 200 (Contra-tendência), validamos a distância:
            else:
                # Calcula a distância para a EMA mais próxima (EMA 50 ou 200)
                dist_ema50 = (ema50 - close_atual) / close_atual

                # Se a distância for pequena, significa que o preço está perto ou a furar a média. Perigoso!
                if dist_ema50 < distancia_minima_pct:
                    return True  # 🛑 BLOQUEIA (Está colado ou a furar a média)

                return False  # ✅ AUTORIZA (Está isolado cá em baixo, longe das EMAs, bom para reversão)

        # --- 🔴 VALIDAR GATILHO DE SELL (Procura Reversão para Baixa) ---
        if signal == Signal.SELL:
            # 1. Proteção estática: Salva os teus prejuízos dos fundos (1738/1789)
            if last_candle['rsi'] < 30:
                return True  # 🛑 BLOQUEIA

            # Se o preço está ABAIXO da EMA 200, está a favor da tendência de queda. Caminho livre.
            if close_atual < ema200:
                return False  # ✅ AUTORIZA

            # Se o preço está ACIMA da EMA 200 (Contra-tendência - Caso do Index 1734):
            else:
                # Calcula a distância para a EMA 50 ou 200
                dist_ema50 = (close_atual - ema50) / ema50

                # Se o preço estiver colado à média ou mesmo em cima dela a furar, não queremos shortar
                if dist_ema50 < distancia_minima_pct:
                    return True  # 🛑 BLOQUEIA (A furar ou perto demais da média)

                return False  # ✅ AUTORIZA (Está esticado lá no topo, longe de tudo, ótimo para apanhar o início do Short)

        return False

    def predict_signal(self, df: pd.DataFrame) -> SignalResult:
        if self.model is None or self.metadata is None:
            logging.warning("⚠️ Modelo ou metadata não carregado.")
            return SignalResult(Signal.HOLD)

        # 1. Gerar indicadores (UMA ÚNICA VEZ)
        # O MarketBrain já faz o dropna() internamente, então recebemos dados limpos
        df_with_ind, features = MarketBrain.add_indicators(df, False)

        if features.empty:
            logging.warning("⚠️ Features vazias após MarketBrain (falta histórico/warm-up).")
            return SignalResult(Signal.HOLD)

        # --- LÓGICA LSTM ---
        if self.model_type == MLModelType.LSTM:
            window_size = 30  # <--- TEM DE SER IGUAL AO TREINO

            if len(features) < window_size:
                logging.warning(f"⚠️ Precisamos de {window_size} velas, mas só temos {len(features)}.")
                return SignalResult(Signal.HOLD)

            if self.scaler is None:
                logging.error("❌ Scaler não encontrado para LSTM!")
                return SignalResult(Signal.HOLD)

            # Preparar a sequência (as últimas X velas)
            X_input_raw = features.tail(window_size).values
            X_input_scaled = self.scaler.transform(X_input_raw)
            X_input = np.expand_dims(X_input_scaled, axis=0)  # Shape (1, 30, n_features)

            proba = self.model.predict(X_input, verbose=0)[0]

        # --- LÓGICA RF / XGB / MLP ---
        else:
            # Pegar apenas a ÚLTIMA linha para predição em tempo real
            latest_features = features.iloc[[-1]]

            # Verificar se há NaNs na última linha (evita erro no predict)
            if latest_features.isnull().values.any():
                logging.warning("⚠️ Última linha contém NaNs. Aguardando mais dados.")
                return SignalResult(Signal.HOLD)

            features_scaled = self.scaler.transform(latest_features)

            proba = self.model.predict_proba(features_scaled)[0]

        # --- DECISÃO COMUM ---
        idx = np.argmax(proba)
        confidence = proba[idx]

        latest_row = df_with_ind.iloc[-1]

        close_price = latest_row['close']
        # print("AQUII", latest_row)
        # atr = latest_row['atr']
        st_lowerband = latest_row['st_lowerband']
        st_upperband = latest_row['st_upperband']

        close = latest_row['close']

        logging.info(
            f"🤖 [{self.model_type.value}] Prob: L:{proba[0]:.2f} | N:{proba[1]:.2f} | H:{proba[2]:.2f} (Conf: {confidence:.2f})")

        # if confidence > 0.5:
        if idx == 2 and confidence > (self.metadata.threshold_buy):  # ALTA
            # $sl, tp = self.compute_sl_tp(close_price, atr, confidence, Signal.BUY)
            sl, tp = self.supertrend_sl_tp(close_price, df_with_ind, Signal.BUY)

            if sl is None or tp is None:
                return SignalResult(Signal.HOLD, confidence=confidence)

            if self.is_exhaustion_zone(df, Signal.BUY):
                return SignalResult(Signal.HOLD, confidence=confidence)

            return SignalResult(Signal.BUY, sl, tp, confidence)
        elif idx == 0 and confidence > (self.metadata.threshold_sell):  # BAIXA
            sl, tp = self.supertrend_sl_tp(close_price, df_with_ind, Signal.SELL)
            # sl, tp = self.compute_sl_tp(close_price, atr, confidence, Signal.SELL)

            if sl is None or tp is None:
                return SignalResult(Signal.HOLD, confidence=confidence)

            if self.is_exhaustion_zone(df, Signal.SELL):
                return SignalResult(Signal.HOLD, confidence=confidence)

            return SignalResult(Signal.SELL, sl, tp, confidence)

        return SignalResult(Signal.HOLD, confidence=confidence)

    async def get_signal(self) -> SignalResult:
        if self.ohlcv is None and self.symbol is None:
            logging.error("Tem que executar em primeiro lugar o método required_init")

        if self.metadata is not None and self.metadata.is_fresh == False:
            self.model_loaded = False

        await self.initialize(self.model_type)
        df = self.fetch_ohlcv(self.ohlcv)
        df = df.iloc[:-1]
        result = self.predict_signal(df)
        logging.info(f"🚦 Sinal ML para {self.symbol}: {result}")
        return result
