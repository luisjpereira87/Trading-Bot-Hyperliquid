import asyncio
import logging
import os
from enum import Enum
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange

from commons.enums.mode_enum import ModeEnum
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.ohlcv_wrapper import OhlcvWrapper
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
    def __init__(self, exchange :ExchangeClient, model_type = MLModelType.RANDOM_FOREST):
        super().__init__()

        self.exchange = exchange
        self.aggressive_mode = ModeEnum.CONSERVATIVE
        self.model_type = model_type
        self.ohlcv: OhlcvWrapper
        self.symbol = None
        self.model = None
        self.scaler = None
        self.last_train_len = 0
        self.confidence_threshold = 0.55
        self.price_ref: float = 0.0

        self.model_loaded = False
        
        self.model_dir = os.path.join("machine_learning", "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, f"modelo_{self.model_type.value.lower()}.pkl")
        self.keras_model_path = os.path.join(self.model_dir, f"modelo_{self.model_type.value.lower()}.keras")
        self.scaler_model_path = os.path.join(self.model_dir, f"modelo_{self.model_type.value.lower()}_scaler.pkl")
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
                mlTrainer = MLTrainer(model_type, False, False)
                await mlTrainer.run()
                logging.warning("✅ Modelo LSTM com treino finalizado")
                self.model = load_model(self.keras_model_path)  # Recarrega após treino
                self.scaler = joblib.load(self.scaler_model_path) 
        else:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.model_loaded = True
                logging.info(f"📥 Modelo {self.model_type.value} carregado de '{self.model_path}'")
            else:
                logging.warning(f"⚠️ Modelo {self.model_type.value} ainda não treinado, a executar treino...")
                mlTrainer = MLTrainer(model_type, False, False)
                await mlTrainer.run()
                logging.warning(f"✅ Modelo {self.model_type.value} com treino finalizado")

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
        df = df.copy()
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['macd'] = df['ema9'] - df['ema21']
        df['rsi'] = RSIIndicator(df['close']).rsi()
        df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['pct_change'] = df['close'].pct_change()
        return df

    def compute_sl_tp(self, price: float, atr: float, confidence: float, direction: Signal)-> tuple[float | None, float | None]:
        risk_factor = 1 + (confidence - 0.5) * 2
        sl_distance = atr * 1.5 * risk_factor
        tp_distance = atr * 2.5 * risk_factor

        if direction == Signal.BUY:
            sl = price - sl_distance
            tp = price + tp_distance
        elif direction == Signal.SELL:
            sl = price + sl_distance
            tp = price - tp_distance
        else:
            return None, None

        return round(sl, 4), round(tp, 4)

    def predict_signal(self, df: pd.DataFrame) -> SignalResult:
        if self.model is None:
            logging.warning("⚠️ Modelo ainda não treinado.")
            return SignalResult(Signal.HOLD, None, None, None)

        df = self.calculate_features(df).dropna()
        if self.model_type == MLModelType.LSTM:
            window_size = 10  # deve ser consistente com treino
            features = df[['rsi', 'atr', 'macd', 'pct_change', 'ema9', 'ema21', 'stoch_k', 'stoch_d']].values
            if len(features) < window_size:
                logging.warning("⚠️ Dados insuficientes para predição LSTM. Retornando HOLD.")
                return SignalResult(Signal.HOLD, None, None, None)
            
            if self.scaler is None:
                logging.warning("Scaler não carregado, não é possível fazer predição LSTM")
                return SignalResult(Signal.HOLD, None, None, None)
            
            X_input_raw = features[-window_size:]
            # Escalar os dados aqui, usando self.scaler
            X_input_scaled = self.scaler.transform(X_input_raw)  # assume scaler foi treinado com essas features na mesma ordem

            X_input = np.expand_dims(X_input_scaled, axis=0)  # shape (1, window_size, n_features)

            proba = self.model.predict(X_input)[0]  # output shape (3,) p[baixa, neutro, alta]
            logging.info(f"ML LSTM prob baixa: {proba[0]:.2f}, neutro: {proba[1]:.2f}, alta: {proba[2]:.2f}")

            idx = proba.argmax()
            confidence = proba[idx]
            latest = df.iloc[-1]
            close_price = latest['close']
            atr = latest['atr']

            logging.info(f"Classe prevista: {idx} (0=Baixa,1=Neutra,2=Alta), probabilidades: {proba}, confidence_threshold={self.confidence_threshold}")

            if self.aggressive_mode:
                if idx == 2:
                    sl, tp = self.compute_sl_tp(close_price, atr, confidence, Signal.BUY)
                    return SignalResult(Signal.BUY, sl, tp, confidence)
                elif idx == 0:
                    sl, tp = self.compute_sl_tp(close_price, atr, confidence, Signal.SELL)
                    return SignalResult(Signal.SELL, sl, tp, confidence)
            else:
                if idx == 2 and proba[2] > self.confidence_threshold:
                    sl, tp = self.compute_sl_tp(close_price, atr, confidence, Signal.BUY)
                    return SignalResult(Signal.BUY, sl, tp, confidence)
                elif idx == 0 and proba[0] > self.confidence_threshold:
                    sl, tp = self.compute_sl_tp(close_price, atr, confidence, Signal.SELL)
                    return SignalResult(Signal.SELL, sl, tp, confidence)

            return SignalResult(Signal.HOLD, None, None, confidence)
        else:
            # Seu código atual para RF, XGB, MLP (features 2D)
            latest = df.iloc[-1]
            features = pd.DataFrame([[
                latest['rsi'], latest['atr'], latest['macd'],
                latest['pct_change'], latest['ema9'], latest['ema21'],
                latest['stoch_k'], latest['stoch_d']
            ]], columns=['rsi', 'atr', 'macd', 'pct_change', 'ema9', 'ema21', 'stoch_k', 'stoch_d'])

            if features.isnull().values.any():
                logging.warning("⚠️ Features contêm NaNs. Retornando 'hold'.")
                return SignalResult(Signal.HOLD, None, None, None)

            proba = self.model.predict_proba(features)[0]
            logging.info(f"ML prob baixa: {proba[0]:.2f}, neutro: {proba[1]:.2f}, alta: {proba[2]:.2f}")

            idx = proba.argmax()
            confidence = proba[idx]
            close_price = latest['close']
            atr = latest['atr']

            if self.aggressive_mode:
                if idx == 2:
                    sl, tp = self.compute_sl_tp(close_price, atr, confidence, Signal.BUY)
                    return SignalResult(Signal.BUY, sl, tp, confidence, proba[2])
                elif idx == 0:
                    sl, tp = self.compute_sl_tp(close_price, atr, confidence, Signal.SELL)
                    return SignalResult(Signal.SELL, sl, tp, confidence, proba[0])
            else:
                if idx == 2 and proba[2] > self.confidence_threshold:
                    sl, tp = self.compute_sl_tp(close_price, atr, confidence, Signal.BUY)
                    return SignalResult(Signal.BUY, sl, tp, confidence, proba[2])
                elif idx == 0 and proba[0] > self.confidence_threshold:
                    sl, tp = self.compute_sl_tp(close_price, atr, confidence, Signal.SELL)
                    return SignalResult(Signal.SELL, sl, tp, confidence, proba[0])

            return SignalResult(Signal.HOLD, None, None, confidence, proba[1])

    async def get_signal(self) -> SignalResult:
        if self.ohlcv is None and self.symbol is None:
            logging.error("Tem que executar em primeiro lugar o método required_init")

        await self.initialize(self.model_type)
        df = self.fetch_ohlcv(self.ohlcv)
        result = self.predict_signal(df)
        logging.info(f"🚦 Sinal ML para {self.symbol}: {result}")
        return result




