import logging
import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

from commons.enums.mode_enum import ModeEnum
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.indicators.custom_indicators_utils import CustomIndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from commons.utils.paths import get_model_path, get_scaler_path
from machine_learning.market_brain import MarketBrain
from trading_bot.exchange_base import ExchangeBase
from trading_bot.exchange_client import ExchangeClient

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ou '3'
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from xgboost import XGBClassifier

from commons.enums.ml_model_enum import MLModelType
from commons.enums.signal_enum import Signal
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from machine_learning.ml_train_pipeline import MLTrainer


class MLStrategy(StrategyBase):
    def __init__(self, exchange: ExchangeBase, model_type = MLModelType.RANDOM_FOREST):
        super().__init__()

        self.exchange = exchange
        self.aggressive_mode = ModeEnum.CONSERVATIVE
        self.model_type = model_type
        self.ohlcv: OhlcvWrapper
        self.symbol = None
        self.model = None
        self.scaler = None
        self.last_train_len = 0
        self.confidence_threshold = 0.70
        self.price_ref: float = 0.0

        self.model_loaded = False
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, "trading-bot","machine_learning", "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = get_model_path(self.model_type.value,self.exchange.get_name(), ".pkl")
        self.keras_model_path = get_model_path(self.model_type.value,self.exchange.get_name(), ".keras")
        self.scaler_model_path = get_scaler_path(self.model_type.value, self.exchange.get_name())
        self.data_dir = "data"
        self.image_path = "img/imagem.png"
    
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
       pass

    def get_sl_tp(self):
        pass

    async def initialize(self, model_type: MLModelType):
        if self.model_loaded == True:
            return

        if self.model_type == MLModelType.LSTM:
            if os.path.exists(self.keras_model_path):
                self.model = load_model(self.keras_model_path)
                self.scaler = joblib.load(self.scaler_model_path)
                self.model_loaded = True
                logging.info(f"📥 Modelo LSTM carregado de '{self.keras_model_path}'")
            else:
                logging.warning("⚠️ Modelo LSTM ainda não treinado, a executar treino...")
                mlTrainer = MLTrainer(self.exchange, model_type, False, False)
                await mlTrainer.run()
                logging.warning("✅ Modelo LSTM com treino finalizado")
                self.model = load_model(self.keras_model_path)  # Recarrega após treino
                self.scaler = joblib.load(self.scaler_model_path)
        else:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_model_path)
                self.model_loaded = True
                logging.info(f"📥 Modelo {self.model_type.value} carregado de '{self.model_path}'")
            else:
                logging.warning(f"⚠️ Modelo {self.model_type.value} ainda não treinado, a executar treino...")
                mlTrainer = MLTrainer(self.exchange, model_type, False, False)
                await mlTrainer.run()
                logging.warning(f"✅ Modelo {self.model_type.value} com treino finalizado")
                self.model = joblib.load(self.model_path)  # Recarrega após treino
                self.scaler = joblib.load(self.scaler_model_path)

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
            X.append(features[i:i+window_size])
            y.append(labels[i+window_size])
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
        """
        Calcula Stop Loss e Take Profit dinâmicos baseados no ATR e Confiança do Modelo.
        Estratégia: Stop curto e controlado (Hard Cap) com Take Profit longo (Fator 3.5).
        """
        if direction == Signal.HOLD or direction is None:
            return None, None

        # 1. Fator de Confiança (oscila entre 0.8 e 1.2 para não distorcer o ATR)
        risk_factor = 0.8 + (confidence * 0.4)

        # 2. Distâncias Base (ATR)
        # Stop Loss mais "apertado" (1.2) para proteger o capital
        # Take Profit mais "largo" (3.5) para deixar o lucro correr
        sl_distance = atr * 1.2 * risk_factor
        tp_distance = atr * 3.5 * risk_factor

        # 3. SEGURANÇA MÁXIMA (Hard Cap no Prejuízo)
        # Impede que o Stop Loss seja maior que 1.5% do preço de entrada,
        # mesmo que a volatilidade (ATR) esteja altíssima.
        max_sl_val = price * 0.015
        if sl_distance > max_sl_val:
            sl_distance = max_sl_val

        # 4. Cálculo dos níveis de preço
        if direction == Signal.BUY:
            sl = price - sl_distance
            tp = price + tp_distance
        elif direction == Signal.SELL:
            sl = price + sl_distance
            tp = price - tp_distance
        else:
            return None, None

        # Arredondamento para 4 casas decimais (padrão da maioria das exchanges)
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

        latest_row = df_with_ind.iloc[-2]
        close_price = latest_row['close']
        #print("AQUII", latest_row)
        atr = latest_row['atr']

        logging.info(
            f"🤖 [{self.model_type.value}] Prob: L:{proba[0]:.2f} | N:{proba[1]:.2f} | H:{proba[2]:.2f} (Conf: {confidence:.2f})")

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




