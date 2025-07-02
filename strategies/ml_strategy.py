import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange


class MLStrategy:
    def __init__(self, exchange, symbol, timeframe='15m', train_interval=100, plot_enabled=False, enable_training=False):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.train_interval = train_interval
        self.plot_enabled = plot_enabled
        self.enable_training = enable_training
        self.model = None
        self.last_train_len = 0
        self.confidence_threshold = 0.55

        self.model_dir = os.path.join("machine_learning", "models")
        self.model_path = os.path.join(self.model_dir, "modelo_rf.pkl")
        self.data_dir = "data"
        self.image_path = "img/imagem.png"

        logging.info(f"Tentando carregar modelo em: {self.model_path}")

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logging.info(f"📥 Modelo ML carregado com sucesso de '{self.model_path}'")
        else:
            logging.warning("⚠️ Modelo ainda não treinado.")

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

    def prepare_dataset(self, df):
        df = self.calculate_features(df)
        df['future_return'] = df['close'].shift(-3) / df['close'] - 1

        conditions = [
            (df['future_return'] > 0.003),
            (df['future_return'] < -0.003)
        ]
        choices = [2, 0]  # 2 = alta, 0 = baixa
        df['label'] = np.select(conditions, choices, default=1)  # 1 = neutro

        df = df.dropna()

        features = df[['rsi', 'atr', 'macd', 'pct_change', 'ema9', 'ema21', 'stoch_k', 'stoch_d']]
        labels = df['label']

        smote = SMOTE(random_state=42)
        features_res, labels_res = smote.fit_resample(features, labels)

        return features_res, labels_res

    def evaluate_model(self, model, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4)
        cm = confusion_matrix(y_test, y_pred)

        logging.info(f"📊 Avaliação do Modelo - Acurácia: {acc:.4f}")
        logging.info(f"Matriz de Confusão:\n{cm}")
        logging.info(f"\nRelatório de Classificação:\n{report}")

        if self.plot_enabled:
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Baixa', 'Neutro', 'Alta'],
                yticklabels=['Baixa', 'Neutro', 'Alta']
            )
            plt.xlabel('Predito')
            plt.ylabel('Real')
            plt.title('Matriz de Confusão')
            os.makedirs(os.path.dirname(self.image_path), exist_ok=True)
            plt.savefig(self.image_path)
            plt.close()

        return model

    def train(self, features, labels):
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.model = self.evaluate_model(model, features, labels)
        self.last_train_len = len(features)
        logging.info("✅ Modelo ML treinado e avaliado com sucesso!")

        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logging.info(f"💾 Modelo salvo em '{self.model_path}'")

    def predict_signal(self, df):
        if self.model is None:
            logging.warning("⚠️ Modelo ainda não treinado.")
            return "hold"

        df = self.calculate_features(df)
        latest = df.iloc[-1]

        features = pd.DataFrame([[
            latest['rsi'], latest['atr'], latest['macd'],
            latest['pct_change'], latest['ema9'], latest['ema21'],
            latest['stoch_k'], latest['stoch_d']
        ]], columns=['rsi', 'atr', 'macd', 'pct_change', 'ema9', 'ema21', 'stoch_k', 'stoch_d'])

        if features.isnull().values.any():
            logging.warning("⚠️ Features contêm NaNs. Retornando 'hold'.")
            return "hold"

        proba = self.model.predict_proba(features)[0]
        logging.info(f"ML prob baixa: {proba[0]:.2f}, neutro: {proba[1]:.2f}, alta: {proba[2]:.2f}")

        idx = proba.argmax()
        if idx == 2 and proba[2] > self.confidence_threshold:
            return "buy"
        elif idx == 0 and proba[0] > self.confidence_threshold:
            return "sell"
        else:
            return "hold"

    async def train_if_due(self, df):
        if not self.enable_training:
            logging.info("🛑 Treinamento desabilitado (enable_training=False). Pulando treino.")
            return

        if len(df) - self.last_train_len >= self.train_interval:
            logging.info(f"🔄 Treinamento devido. Treinando modelo com {len(df)} registros...")
            features, labels = self.prepare_dataset(df)
            self.train(features, labels)

            os.makedirs(self.data_dir, exist_ok=True)
            filename = os.path.join(self.data_dir, f"{self.symbol.replace('/', '_').replace(':', '_')}_{self.timeframe}_train.csv")
            df.to_csv(filename, index=False)
            logging.info(f"📁 Dataset salvo para treino em: {filename}")

    async def get_signal(self):
        df = await self.fetch_ohlcv()
        await self.train_if_due(df)
        signal = self.predict_signal(df)
        logging.info(f"🚦 Sinal ML para {self.symbol}: {signal}")
        return signal


