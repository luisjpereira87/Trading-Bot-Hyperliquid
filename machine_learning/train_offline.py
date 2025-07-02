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

logging.basicConfig(level=logging.INFO)


def prepare_dataset(df):
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

    df['future_return'] = df['close'].shift(-3) / df['close'] - 1
    df['label'] = np.select(
        [
            df['future_return'] > 0.003,
            df['future_return'] < -0.003
        ],
        [2, 0],
        default=1
    )

    df = df.dropna()

    features = df[['rsi', 'atr', 'macd', 'pct_change', 'ema9', 'ema21', 'stoch_k', 'stoch_d']]
    labels = df['label']

    smote = SMOTE(random_state=42)
    features_res, labels_res = smote.fit_resample(features, labels)

    return features_res, labels_res


def evaluate_model(model, features, labels, plot_enabled=True):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print("📊 Avaliação do Modelo")
    print(f"Acurácia: {acc:.4f}")
    print(report)

    if plot_enabled:
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
        os.makedirs("img", exist_ok=True)
        plt.savefig("img/imagem.png")
        plt.close()

    return model


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "ohlcv_data.csv")
    img_dir = os.path.join(base_dir, "machine_learning", "img")
    model_path = os.path.join(base_dir, "machine_learning", "models", "modelo_rf.pkl")

    if not os.path.exists(csv_path):
        logging.error(f"Arquivo CSV não encontrado em: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        logging.error("O CSV não contém todas as colunas necessárias: open, high, low, close, volume")
        return

    if df.empty:
        logging.error("O DataFrame está vazio. Verifique o conteúdo do CSV.")
        return

    logging.info("📥 CSV carregado com sucesso")
    features, labels = prepare_dataset(df)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    trained_model = evaluate_model(model, features, labels, plot_enabled=True)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(trained_model, model_path)
    logging.info(f"💾 Modelo treinado salvo em '{model_path}'")


if __name__ == "__main__":
    main()
