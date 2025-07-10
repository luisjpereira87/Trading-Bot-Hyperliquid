import asyncio
import logging
import os
from enum import Enum

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
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier

from commons.enums.ml_model_enum import MLModelType
from commons.enums.signal_enum import Signal
from commons.models.signal_result import SignalResult
from commons.models.strategy_base import StrategyBase
from machine_learning.ml_train_pipeline import MLTrainer


class MLStrategy(StrategyBase):
    def __init__(self, exchange, symbol, timeframe='15m', train_interval=100,
                 plot_enabled=False, enable_training=False, aggressive_mode=False,
                 model_type=MLModelType.RANDOM_FOREST):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.train_interval = train_interval
        self.plot_enabled = plot_enabled
        self.enable_training = enable_training
        self.aggressive_mode = aggressive_mode
        self.model_type = model_type
        self.model = None
        self.last_train_len = 0
        self.confidence_threshold = 0.55
        

        self.model_dir = os.path.join("machine_learning", "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, f"modelo_{self.model_type.value.lower()}.pkl")
        self.keras_model_path = os.path.join(self.model_dir, f"modelo_{self.model_type.value.lower()}.keras")
        self.data_dir = "data"
        self.image_path = "img/imagem.png"
            

    async def initialize(self, model_type):
        if self.model_type == MLModelType.LSTM:
            if os.path.exists(self.keras_model_path):
                self.model = load_model(self.keras_model_path)
                logging.info(f"📥 Modelo LSTM carregado de '{self.keras_model_path}'")
            else:
                logging.warning("⚠️ Modelo LSTM ainda não treinado, a executar treino...")
                mlTrainer = MLTrainer(model_type, False, False)
                await mlTrainer.run()
                logging.warning("✅ Modelo LSTM com treino finalizado")
                self.model = load_model(self.keras_model_path)  # Recarrega após treino
        else:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logging.info(f"📥 Modelo {self.model_type.value} carregado de '{self.model_path}'")
            else:
                logging.warning(f"⚠️ Modelo {self.model_type.value} ainda não treinado, a executar treino...")
                mlTrainer = MLTrainer(model_type, False, False)
                await mlTrainer.run()
                logging.warning(f"✅ Modelo {self.model_type.value} com treino finalizado")

    def create_lstm_sequences(self, df, window_size=10):
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

    async def fetch_ohlcv(self, limit=500):
        data = await self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    
    def calculate_features(self, df):
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
    

    """
    def prepare_dataset(self, df):
        df = self.calculate_features(df)
        df['future_return'] = df['close'].shift(-3) / df['close'] - 1

        conditions = [
            (df['future_return'] > 0.003),
            (df['future_return'] < -0.003)
        ]
        choices = [2, 0]
        df['label'] = np.select(conditions, choices, default=1)

        df = df.dropna()

        features = df[['rsi', 'atr', 'macd', 'pct_change', 'ema9', 'ema21', 'stoch_k', 'stoch_d']]
        labels = df['label']

        smote = SMOTE(random_state=42)
        features_res, labels_res = smote.fit_resample(features, labels)

        return features_res, labels_res
    """

    """    
    def get_model_with_gridsearch(self):
        if self.model_type == MLModelType.RANDOM_FOREST:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 5, 10]
            }
            base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
        elif self.model_type == MLModelType.XGBOOST:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
            base_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        elif self.model_type == MLModelType.MLP:
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,)],
                'activation': ['relu', 'tanh'],
                'max_iter': [300]
            }
            base_model = MLPClassifier(random_state=42)
        else:
            raise ValueError("Modelo não suportado")

        return GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    """
    """
    def evaluate_model(self, model, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4)
        cm = confusion_matrix(y_test, y_pred)

        logging.info(f"📊 Avaliação do Modelo ({self.model_type.value}) - Acurácia: {acc:.4f}")
        logging.info(f"Matriz de Confusão:\n{cm}")
        logging.info(f"\nRelatório de Classificação:\n{report}")

        if self.plot_enabled:
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Baixa', 'Neutro', 'Alta'],
                        yticklabels=['Baixa', 'Neutro', 'Alta'])
            plt.xlabel('Predito')
            plt.ylabel('Real')
            plt.title('Matriz de Confusão')
            os.makedirs(os.path.dirname(self.image_path), exist_ok=True)
            plt.savefig(self.image_path)
            plt.close()

        return model
    """

    """
    def train(self, features, labels):
        model = self.get_model_with_gridsearch()
        self.model = self.evaluate_model(model, features, labels)
        self.last_train_len = len(features)

        # Salva o GridSearchCV completo para análise futura
        grid_path = self.model_path.replace(".pkl", "_grid.pkl")
        joblib.dump(self.model, grid_path)
        logging.info(f"💾 GridSearchCV completo salvo em '{grid_path}'")

        # Salva só o melhor estimador para uso prático
        joblib.dump(self.model.best_estimator_, self.model_path)
        logging.info(f"💾 Melhor modelo salvo em '{self.model_path}'")

    """

    def compute_sl_tp(self, price, atr, confidence, direction):
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

    def predict_signal(self, df) -> SignalResult:
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
            X_input = features[-window_size:]
            X_input = np.expand_dims(X_input, axis=0)  # shape (1, window_size, n_features)

            proba = self.model.predict(X_input)[0]  # output shape (3,) p[baixa, neutro, alta]
            logging.info(f"ML LSTM prob baixa: {proba[0]:.2f}, neutro: {proba[1]:.2f}, alta: {proba[2]:.2f}")

            idx = proba.argmax()
            confidence = proba[idx]
            latest = df.iloc[-1]
            close_price = latest['close']
            atr = latest['atr']

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
    """
    async def train_if_due(self, df):
        if not self.enable_training:
            return

        if len(df) - self.last_train_len >= self.train_interval:
            features, labels = self.prepare_dataset(df)
            self.train(features, labels)

            os.makedirs(self.data_dir, exist_ok=True)
            filename = os.path.join(self.data_dir, f"{self.symbol.replace('/', '_').replace(':', '_')}_{self.timeframe}_train.csv")
            df.to_csv(filename, index=False)
            logging.info(f"📁 Dataset salvo para treino em: {filename}")
    """

    async def get_signal(self) -> SignalResult:
        await self.initialize(self.model_type)
        df = await self.fetch_ohlcv()
        #await self.train_if_due(df)
        result = self.predict_signal(df)
        logging.info(f"🚦 Sinal ML para {self.symbol}: {result}")
        return result




