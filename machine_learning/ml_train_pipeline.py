import asyncio
import json
import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ccxt.async_support import hyperliquid
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange
from xgboost import XGBClassifier

from strategies.ml_strategy import MLModelType

logging.basicConfig(level=logging.INFO)

class MLTrainer:
    TIMEFRAME = "15m"
    CANDLE_LIMIT = 10000
    DATA_DIR = "data"
    MODEL_PATH = "models/modelo.pkl"  # Generalizado o nome do arquivo

    def __init__(self, model_type=MLModelType.RANDOM_FOREST):
        self.model_type = model_type
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
        else:
            raise ValueError("Modelo desconhecido")

    def _get_param_grid(self):
        # Grid de hiperparâmetros para GridSearchCV conforme o modelo
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

        return features_res, labels_res

    def train_and_save_model(self, features, labels):
        model = self._init_model()
        param_grid = self._get_param_grid()

        X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels, test_size=0.2, random_state=42)

        if param_grid:
            logging.info(f"🔍 Iniciando GridSearchCV para o modelo {self.model_type}...")
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            logging.info(f"✅ Melhor modelo encontrado: {grid_search.best_params_}")
        else:
            best_model = model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4)
        cm = confusion_matrix(y_test, y_pred)

        logging.info(f"📊 Acurácia: {acc:.4f}")
        print(report)

        os.makedirs(self.IMG_DIR, exist_ok=True)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Baixa", "Neutro", "Alta"], yticklabels=["Baixa", "Neutro", "Alta"])
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.title("Matriz de Confusão")
        plt.savefig(os.path.join(self.IMG_DIR, "imagem.png"))
        plt.close()

        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        joblib.dump(best_model, self.MODEL_PATH)
        logging.info(f"💾 Modelo salvo em: {self.MODEL_PATH}")

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


if __name__ == "__main__":
    trainer = MLTrainer(model_type=MLModelType.RANDOM_FOREST)  # ou 'xgboost' ou 'mlp'
    asyncio.run(trainer.run())



