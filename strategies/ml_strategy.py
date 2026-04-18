import json
import logging
import os
from datetime import datetime, timedelta
from typing import List

import joblib
import numpy as np
import pandas as pd

from commons.enums.mode_enum import ModeEnum
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from commons.utils.paths import get_model_path, get_scaler_path, get_bayesian_path
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
        self.confidence_threshold = 0.70
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

        if self.is_combined_model:
            keras_model_path = get_model_path(self.model_type.value, self.exchange_name, "MASTER", ".keras")
            model_path = get_model_path(self.model_type.value, self.exchange_name, "MASTER", ".pkl")
            scaler_model_path = get_scaler_path(self.model_type.value, self.exchange_name, "MASTER")
        else:
            symbol_clean = self.symbol.replace("/", "_").replace(":", "_")
            keras_model_path = get_model_path(self.model_type.value, self.exchange_name, symbol_clean, ".keras")
            model_path = get_model_path(self.model_type.value, self.exchange_name, symbol_clean, ".pkl")
            scaler_model_path = get_scaler_path(self.model_type.value, self.exchange_name, symbol_clean)

        # Define qual o caminho do ficheiro principal para validar a data
        target_path = keras_model_path if model_type == MLModelType.LSTM else model_path

        # Lógica de Re-treino: Existe e tem menos de 7 dias?
        model_exists = os.path.exists(target_path)
        is_fresh = False

        if model_exists:
            file_mod_time = os.path.getmtime(target_path)
            last_modified = datetime.fromtimestamp(file_mod_time)
            if datetime.now() < last_modified + timedelta(days=7):
                is_fresh = True
            else:
                logging.info(
                    f"📅 Modelo com mais de 7 dias ({last_modified.strftime('%Y-%m-%d')}). A forçar re-treino...")

        # Se o modelo existe e é recente, carregamos
        if model_exists and is_fresh:
            if model_type == MLModelType.LSTM:
                self.model = load_model(keras_model_path)
                logging.info("📥 Modelo LSTM carregado e atualizado.")
            else:
                self.model = joblib.load(model_path)
                # self.bayesian_model = self.load_bayesian_model()
                logging.info(f"📥 Modelo {self.model_type.value} carregado e atualizado.")

            self.scaler = joblib.load(scaler_model_path)
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

    def predict_signal(self, df: pd.DataFrame) -> SignalResult:
        if self.model is None:
            logging.warning("⚠️ Modelo não carregado.")
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
        atr = latest_row['atr']

        logging.info(
            f"🤖 [{self.model_type.value}] Prob: L:{proba[0]:.2f} | N:{proba[1]:.2f} | H:{proba[2]:.2f} (Conf: {confidence:.2f})")

        # bayes_confidence = self.get_final_confidence(confidence, latest_row)

        # 3. Decisão Final (A Média Ponderada)
        # O Bayes atua como um filtro de sanidade
        # final_score = (lgbm_buy_prob * 0.7) + (bayes_confidence * 0.3)

        # print(f"bayes_confidence= {bayes_confidence}")

        if confidence > self.confidence_threshold:
            if idx == 2:  # ALTA
                sl, tp = self.compute_sl_tp(close_price, atr, confidence, Signal.BUY)
                return SignalResult(Signal.BUY, sl, tp, confidence)
            elif idx == 0:  # BAIXA
                sl, tp = self.compute_sl_tp(close_price, atr, confidence, Signal.SELL)
                return SignalResult(Signal.SELL, sl, tp, confidence)

        return SignalResult(Signal.HOLD, confidence=confidence)

    async def get_signal(self) -> SignalResult:
        if self.ohlcv is None and self.symbol is None:
            logging.error("Tem que executar em primeiro lugar o método required_init")

        await self.initialize(self.model_type)
        df = self.fetch_ohlcv(self.ohlcv)
        result = self.predict_signal(df)
        logging.info(f"🚦 Sinal ML para {self.symbol}: {result}")
        return result

    def get_final_confidence(self, lgbm_prob_class_2: float, current_data: dict) -> float:
        """
        Aplica o Teorema de Bayes para ajustar a confiança do LightGBM.
        """
        model = self.bayesian_model
        print(model)
        prior = model['prior_win']

        # 1. Começamos com a "crença" do LightGBM
        # Se o modelo está muito confiante, o likelihood é alto
        likelihood_ratio = 1.0

        # 2. Ajustamos pelas 6 âncoras
        # Mapeamos o valor atual para o bin correspondente (mesma lógica do treino)
        mappings = {
            'rsi_bin': 'OVERSOLD' if current_data['rsi'] < 30 else (
                'OVERBOUGHT' if current_data['rsi'] > 70 else 'NEUTRAL'),
            'adx_bin': 'WEAK' if current_data['adx'] < 25 else ('STRONG' if current_data['adx'] < 50 else 'EXTREME'),
            'score_bin': 'LOW' if current_data['super_score'] < 3 else (
                'HIGH' if current_data['super_score'] > 7 else 'MID'),
            'chop_bin': 'TRENDING' if current_data['choppiness'] < 38 else (
                'CHOPPY' if current_data['choppiness'] > 61 else 'NORMAL'),
            'ema_bin': 'BULL' if current_data['above_ema200'] == 1 else 'BEAR'
            # ... adicionar relative_volume conforme a lógica de qcut
        }

        for feat, bin_val in mappings.items():
            # P(Win | Atributo) vinda do histórico
            p_attr = model['tables'].get(feat, {}).get(bin_val, prior)

            # Se P(Atributo) > Prior, este indicador ajuda. Se for <, ele penaliza.
            # Isto é uma simplificação robusta do ajuste bayesiano
            likelihood_ratio *= (p_attr / prior)

        # 3. Resultado Final: Confiança do Modelo * Força Estatística das Âncoras
        final_prob = lgbm_prob_class_2 * likelihood_ratio

        # Clipping para garantir que fica entre 0 e 1
        return max(0.0, min(1.0, final_prob))
