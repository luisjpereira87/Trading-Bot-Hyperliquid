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
from sklearn.model_selection import train_test_split
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange

logging.basicConfig(level=logging.INFO)

# === CONFIGURAÇÕES ===
SYMBOL = "ETH/USDC:USDC"
TIMEFRAME = "15m"
CANDLE_LIMIT = 10000
CSV_PATH = f"data/ohlcv_data.csv"
MODEL_PATH = "models/modelo_rf.pkl"

# ========== FETCH CANDLES ==========
async def fetch_ohlcv():
    exchange = hyperliquid({
        "enableRateLimit": True,
        "testnet": True,
    })
    logging.info(f"🔁 Baixando {CANDLE_LIMIT} candles para {SYMBOL}...")

    data = await exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    os.makedirs("data", exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    logging.info(f"✅ CSV salvo em: {CSV_PATH}")

    # Fecha a sessão para liberar recursos
    await exchange.close()

    return df

# ========== PREPARA DATASET ==========
def prepare_dataset(df):
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

# ========== TREINA E AVALIA ==========
def train_and_save_model(features, labels):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels, test_size=0.2)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    logging.info(f"📊 Acurácia: {acc:.4f}")
    print(report)

    # Salva imagem da matriz de confusão
    os.makedirs("img", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Baixa", "Neutro", "Alta"], yticklabels=["Baixa", "Neutro", "Alta"])
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.savefig("img/imagem.png")
    plt.close()

    # Salva o modelo treinado
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logging.info(f"💾 Modelo salvo em: {MODEL_PATH}")

def load_pair_configs():
    # Caminho relativo à raiz do projeto
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # sobe 1 pasta para raiz
    path = os.path.join(base_dir, "config", "pairs.json")

    try:
        with open(path, "r") as f:
            pairs = json.load(f)
        return pairs
    except FileNotFoundError:
        import logging
        logging.error(f"Arquivo de configuração {path} não encontrado.")
        return []

# ========== MAIN ==========
async def main():
    pairs = load_pair_configs()
    if not pairs:
        logging.warning("Nenhum par carregado da configuração.")
        return

    for pair_cfg in pairs:
        symbol = pair_cfg.get("symbol")
        if not symbol:
            logging.warning("Par sem símbolo ignorado.")
            continue
        
        logging.info(f"📈 Processando par: {symbol}")
        global SYMBOL
        SYMBOL = symbol
        
        df = await fetch_ohlcv()
        features, labels = prepare_dataset(df)
        train_and_save_model(features, labels)

if __name__ == "__main__":
    asyncio.run(main())
