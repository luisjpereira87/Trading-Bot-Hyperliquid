import asyncio
import json
import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from ccxt.async_support import hyperliquid
from imblearn.over_sampling import SMOTE
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import compute_class_weight
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from xgboost import XGBClassifier

from commons.enums.ml_model_enum import MLModelType

logging.basicConfig(level=logging.INFO)


class MLTrainer:
    TIMEFRAME = "15m"
    CANDLE_LIMIT = 10000

    def __init__(self, model_type=MLModelType.RANDOM_FOREST, save_csv=False, save_img=False, use_gridsearch=False):
        self.model_type = model_type
        self.save_csv = save_csv
        self.save_img = save_img
        self.use_gridsearch = use_gridsearch

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.base_dir, "data")
        self.MODEL_PATH = os.path.join(self.base_dir, "models", f"modelo_{model_type.value.lower()}.pkl")
        self.IMG_DIR = os.path.join(self.base_dir, "img")

    def _init_model(self):
        if self.model_type == MLModelType.RANDOM_FOREST:
            return RandomForestClassifier(random_state=42)
        elif self.model_type == MLModelType.XGBOOST:
            return XGBClassifier(eval_metric='mlogloss', random_state=42)
        elif self.model_type == MLModelType.MLP:
            return MLPClassifier(max_iter=500, random_state=42)
        elif self.model_type == MLModelType.LSTM:
            # Para LSTM model, criaremos na train_and_save_model pois precisa input 3D
            return None
        else:
            raise ValueError("Modelo desconhecido")

    def _get_param_grid(self):
        if self.model_type == MLModelType.RANDOM_FOREST:
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'class_weight': [None, 'balanced']
            }
        elif self.model_type == MLModelType.XGBOOST:
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 1]
            }
        elif self.model_type == MLModelType.MLP:
            return {
                'hidden_layer_sizes': [(32,), (64, 32), (128, 64)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001]
            }
        else:
            return {}

    async def fetch_ohlcv(self, symbol):
        exchange = hyperliquid({
            "enableRateLimit": True,
            "testnet": True,
        })
        logging.info(f"🔁 Baixando {self.CANDLE_LIMIT} candles para {symbol}...")

        data = await exchange.fetch_ohlcv(symbol, timeframe=self.TIMEFRAME, limit=self.CANDLE_LIMIT)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        if self.save_csv:
            os.makedirs(self.DATA_DIR, exist_ok=True)
            csv_path = os.path.join(self.DATA_DIR, f"ohlcv_data_{symbol.replace('/', '_').replace(':', '_')}.csv")
            df.to_csv(csv_path, index=False)
            logging.info(f"✅ CSV salvo em: {csv_path}")

        await exchange.close()
        return df

    def prepare_dataset(self, df):
        df = df.copy()
        df["ema9"] = df["close"].ewm(span=9).mean()
        df["ema21"] = df["close"].ewm(span=21).mean()
        df["macd"] = df["ema9"] - df["ema21"]
        df["rsi"] = RSIIndicator(df["close"]).rsi()
        df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
        stoch = StochasticOscillator(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        df["pct_change"] = df["close"].pct_change()

        df["future_return"] = df["close"].shift(-3) / df["close"] - 1
        df["label"] = np.select(
            [df["future_return"] > 0.003, df["future_return"] < -0.003],
            [2, 0], default=1
        )
        df = df.dropna()

        features = df[["rsi", "atr", "macd", "pct_change", "ema9", "ema21", "stoch_k", "stoch_d"]]
        labels = df["label"]

        smote = SMOTE(random_state=42)
        features_res, labels_res = smote.fit_resample(features, labels)

        if self.save_csv:
            os.makedirs(self.DATA_DIR, exist_ok=True)
            dataset_path = os.path.join(self.DATA_DIR, f"dataset_{self.model_type.value.lower()}.csv")
            df_full = features_res.copy()
            df_full["label"] = labels_res
            df_full.to_csv(dataset_path, index=False)
            logging.info(f"✅ Dataset balanceado salvo em: {dataset_path}")

        return features_res, labels_res

    def build_lstm_model(self, input_shape, num_classes, lstm_units=50, dropout_rate=0.2):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units=lstm_units, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=lstm_units // 2, return_sequences=False))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        return model

    def create_sequences(self, X, y, window_size=20):
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size):
            X_seq.append(X[i:i+window_size])
            y_seq.append(y[i+window_size])
        return np.array(X_seq), np.array(y_seq)

    def load_lstm_model(self):
        model_path_keras = self.MODEL_PATH.replace(".pkl", ".h5")
        if not os.path.exists(model_path_keras):
            raise FileNotFoundError(f"Modelo LSTM não encontrado em {model_path_keras}")
        model = tf.keras.models.load_model(model_path_keras)
        logging.info(f"💾 Modelo LSTM carregado de: {model_path_keras}")
        return model

    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, model_path='best_lstm_model.keras'):
        # Normaliza os dados (fit no treino, aplica no treino e validação)
        scaler = MinMaxScaler()
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        scaler.fit(X_train_reshaped)
        X_train_scaled = scaler.transform(X_train_reshaped).reshape(n_samples, n_timesteps, n_features)
        X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape[0], n_timesteps, n_features)

        # One-hot encode das classes (se ainda não estiver feito)
        num_classes = len(np.unique(y_train))
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)

        model = self.build_lstm_model(input_shape=(n_timesteps, n_features), num_classes=num_classes)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            #ModelCheckpoint("best_lstm_model.h5", monitor='val_accuracy', save_best_only=True, mode='max')
        ]

        history = model.fit(
            X_train_scaled, y_train_cat,
            validation_data=(X_val_scaled, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2
        )

        # Avaliar no conjunto de validação
        y_val_pred_prob = model.predict(X_val_scaled)
        y_val_pred = np.argmax(y_val_pred_prob, axis=1)

        print("Classification Report (val):")
        print(classification_report(y_val, y_val_pred))

        if self.save_img:
            # Gráfico da evolução do treino
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Loss treino')
            plt.plot(history.history['val_loss'], label='Loss validação')
            plt.title('Loss durante o treino')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Accuracy treino')
            plt.plot(history.history['val_accuracy'], label='Accuracy validação')
            plt.title('Accuracy durante o treino')
            plt.legend()

            plt.show()

        return model, scaler, history

    # Exemplo de chamada (X_train, y_train, X_val, y_val devem ser numpy arrays já preparados)
    # model, scaler, history = train_lstm(X_train, y_train, X_val, y_val)

    def predict_lstm(self, model, features, seq_len=5):
        X = []
        for i in range(len(features) - seq_len):
            X.append(features.iloc[i:i+seq_len].values)
        X = np.array(X)
        preds_prob = model.predict(X)
        preds = np.argmax(preds_prob, axis=1)
        preds_aligned = np.concatenate([np.full(seq_len, np.nan), preds])
        return preds_aligned

    def train_and_save_model(self, features, labels):
        if self.model_type == MLModelType.LSTM:
            window_size = 10  # ajuste esse valor conforme fizer sentido para seu problema
            X_seq, y_seq = self.create_sequences(features.values, labels.values, window_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=42
            )
            model, scaler, history = self.train_lstm(X_train, y_train, X_val, y_val)
            os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
            model_path_keras = self.MODEL_PATH.replace(".pkl", ".keras")
            model.save(model_path_keras, include_optimizer=False)
            logging.info(f"✅ Modelo LSTM salvo em: {model_path_keras}")
            return model, history

        model = self._init_model()
        if self.use_gridsearch:
            param_grid = self._get_param_grid()
            if param_grid:
                logging.info("🔎 Iniciando GridSearchCV para otimizar hiperparâmetros...")
                gs = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
                gs.fit(features, labels)
                model = gs.best_estimator_
                logging.info(f"✅ Melhor modelo após GridSearch: {gs.best_params_}")

        model.fit(features, labels)
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        joblib.dump(model, self.MODEL_PATH)
        logging.info(f"✅ Modelo salvo em: {self.MODEL_PATH}")
        return model, None

    def evaluate_model(self, model, X_test, y_test):
        if self.model_type == MLModelType.LSTM:
            preds = self.predict_lstm(model, X_test)
            y_test_aligned = y_test.iloc[len(y_test) - len(preds):]
            mask = ~np.isnan(preds)
            preds_clean = preds[mask].astype(int)
            y_true = y_test_aligned[mask].astype(int)

            acc = accuracy_score(y_true, preds_clean)
            logging.info(f"Accuracy LSTM: {acc:.4f}")
            print(classification_report(y_true, preds_clean))
            self.plot_confusion_matrix(y_true, preds_clean)

            return acc

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logging.info(f"Accuracy {self.model_type.value}: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        self.plot_confusion_matrix(y_test, y_pred)
        return acc

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def load_pair_configs(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.path = os.path.join(self.base_dir, "config", "pairs.json")

        try:
            with open(self.path, "r") as f:
                pairs = json.load(f)
            return pairs
        except FileNotFoundError:
            logging.error(f"Arquivo de configuração {self.path} não encontrado.")
            return []

    async def run(self):
        # Mantido igual ao seu original!
        pairs = self.load_pair_configs()
        if not pairs:
            logging.warning("Nenhum par carregado da configuração.")
            return

        df_all = pd.DataFrame()
        for pair_cfg in pairs:
            symbol = pair_cfg.get("symbol")
            if not symbol:
                logging.warning("Par sem símbolo ignorado.")
                continue

            df = await self.fetch_ohlcv(symbol)
            df["symbol"] = symbol  # opcional para futura análise
            df_all = pd.concat([df_all, df], ignore_index=True)

        features, labels = self.prepare_dataset(df_all)
        self.train_and_save_model(features, labels)





