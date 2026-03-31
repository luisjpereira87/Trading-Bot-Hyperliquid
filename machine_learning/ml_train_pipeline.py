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
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

from commons.utils.indicators.custom_indicators_utils import CustomIndicatorsUtils
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ou '3'
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
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
            return RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', random_state=42)
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
            "testnet": False,
        })
        logging.info(f"🔁 Baixando {self.CANDLE_LIMIT} candles para {symbol}...")

        # 1. Carregar candles antigos
        since_timestamp = int(pd.Timestamp("2025-03-01").timestamp() * 1000)  # em ms
        old_data = await exchange.fetch_ohlcv(symbol, timeframe=self.TIMEFRAME, since=since_timestamp, limit=self.CANDLE_LIMIT)
        old_df = pd.DataFrame(old_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # 2. Buscar candles recentes da exchange (últimos 500 candles, por ex)
        since_timestamp1 = int(pd.Timestamp("2024-10-01").timestamp() * 1000) 
        new_data = await exchange.fetch_ohlcv(symbol, timeframe=self.TIMEFRAME, since=since_timestamp1, limit=self.CANDLE_LIMIT)
        new_df = pd.DataFrame(new_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # 3. Concatenar e remover duplicados
        df_total = pd.concat([old_df, new_df])
        df_total = df_total.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

        df = pd.DataFrame(df_total, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        if self.save_csv:
            os.makedirs(self.DATA_DIR, exist_ok=True)
            csv_path = os.path.join(self.DATA_DIR, f"ohlcv_data_{symbol.replace('/', '_').replace(':', '_')}.csv")
            df.to_csv(csv_path, index=False)
            logging.info(f"✅ CSV salvo em: {csv_path}")

        await exchange.close()


        return df

    def prepare_dataset(self, df):

        # Criamos uma cópia temporária para não estragar o DF original
        df_temp = df.copy()

        # Se o timestamp já for datetime, convertemos de volta para milissegundos (int)
        if pd.api.types.is_datetime64_any_dtype(df_temp['timestamp']):
            df_temp['timestamp'] = df_temp['timestamp'].astype('int64') // 10 ** 6

        custom_indicators = CustomIndicatorsUtils(OhlcvWrapper(df_temp.values.tolist()))
        supertrend, trend, final_upperband, final_lowerband, supertrend_smooth, direction, perf_score = custom_indicators.supertrend()

        final_scores, smooth_scores = custom_indicators.calculate_super_score()
        df = df.copy()

        # --- NOVIDADE: Injetar os teus indicadores no DF ---
        # Garantimos que os tamanhos batem (dropna pode ser necessário depois)
        df["super_score"] = final_scores
        df["st_direction"] = direction  # 1 para alta, -1 para baixa
        # --------------------------------------------------

        df["ema9"] = df["close"].ewm(span=9).mean()
        df["ema21"] = df["close"].ewm(span=21).mean()
        df["macd"] = df["ema9"] - df["ema21"]
        df["rsi"] = RSIIndicator(df["close"]).rsi()
        df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
        stoch = StochasticOscillator(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        df["pct_change"] = df["close"].pct_change()

        df["future_return"] = df["close"].shift(-20) / df["close"] - 1


        df["target_vol"] = df["atr"] * 2 / df["close"]
        df["label"] = np.select(
            [df["future_return"] > df["target_vol"], df["future_return"] < -df["target_vol"]],
            [2, 0], default=1
        )
        """
        df["label"] = np.select(
            [df["future_return"] > 0.006, df["future_return"] < -0.006],
            [2, 0], default=1
        )
        """
        df['momentum'] = df['close'].pct_change(periods=3)
        df['roc'] = df['close'].pct_change(periods=10)

        # 1. Z-Score (Distância da Média em Desvios Padrão)
        # Diz ao modelo se o preço está "caro" ou "barato" em relação à média
        df['mean_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['z_score'] = (df['close'] - df['mean_20']) / df['std_20']

        # 2. Volume Relativo
        # Reversões com volume baixo são geralmente falsas
        df['vol_ema'] = df['volume'].ewm(span=20).mean()
        df['relative_volume'] = df['volume'] / df['vol_ema']

        # 3. Bandas de Bollinger (Largura)
        # Ajuda o modelo a detetar "Squeezes" de volatilidade
        df['bb_width'] = (df['ema21'] - df['ema9']) / df['ema21']  # Simples spread de médias

        # 4. Tendência de Longo Prazo
        # O modelo precisa saber se a tendência macro é de alta ou baixa
        df['ema_200'] = df['close'].ewm(span=200).mean()
        df['above_ema200'] = (df['close'] > df['ema_200']).astype(int)

        # --- 2. Filtro de Lateralização (ADX) ---
        adx_inst = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_inst.adx()

        df = df.dropna()

        features = df[[
            "rsi", "atr", "macd", "pct_change", "ema9", "ema21", "stoch_k", "stoch_d",
            "z_score", "relative_volume", "bb_width", "above_ema200", "momentum", "roc", "adx",
            "super_score", "st_direction"
        ]]
        labels = df["label"]

        print(df[['close', 'rsi', 'macd', 'ema9', 'ema21', 'stoch_k', 'stoch_d']].tail(15))
        
        #smote = SMOTE(random_state=42)
        #features_res, labels_res = smote.fit_resample(features, labels) # type: ignore

        if self.save_csv:
            os.makedirs(self.DATA_DIR, exist_ok=True)
            dataset_path = os.path.join(self.DATA_DIR, f"dataset_{self.model_type.value.lower()}.csv")
            #df_full = features_res.copy()
            #df_full["label"] = labels_res
            df.to_csv(dataset_path, index=False)
            logging.info(f"✅ Dataset balanceado salvo em: {dataset_path}")

        return features, labels

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
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model

    def create_sequences(self, X, y, window_size=30):
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size):
            X_seq.append(X[i:i+window_size])
            y_seq.append(y[i+window_size])
        return np.array(X_seq), np.array(y_seq)

    def load_lstm_model(self):
        model_path_keras = self.MODEL_PATH.replace(".pkl", ".h5")
        if not os.path.exists(model_path_keras):
            raise FileNotFoundError(f"Modelo LSTM não encontrado em {model_path_keras}")
        model = tf.keras.models.load_model(model_path_keras) # type: ignore
        logging.info(f"💾 Modelo LSTM carregado de: {model_path_keras}")
        return model

    def train_lstm(self, X_train, y_train, X_val, y_val, class_weight=None):
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

        # Calcula class weights para lidar com desequilíbrio
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))

        model = self.build_lstm_model(input_shape=(n_timesteps, n_features), num_classes=num_classes)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            #ModelCheckpoint("best_lstm_model.h5", monitor='val_accuracy', save_best_only=True, mode='max')
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            class_weight=class_weight_dict,
            verbose=1,
            callbacks=callbacks
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

    def predict_lstm(self, model, features, seq_len=5, confidence_threshold=0.7):
        X = []
        for i in range(len(features) - seq_len):
            X.append(features.iloc[i:i+seq_len].values)
        X = np.array(X)

        preds_prob = model.predict(X)
        preds = np.argmax(preds_prob, axis=1)
        max_probs = np.max(preds_prob, axis=1)

        # Aplicar threshold: se prob < threshold, define como 'hold' (exemplo: classe 0)
        preds_thresholded = []
        for prob, pred in zip(max_probs, preds):
            if prob >= confidence_threshold:
                preds_thresholded.append(pred)
            else:
                preds_thresholded.append(0)  # hold ou neutro

        preds_aligned = np.concatenate([np.full(seq_len, np.nan), preds_thresholded])
        return preds_aligned

    def train_and_save_model(self, features, labels):
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)

        # --- ALTERAÇÃO 1: SPLIT TEMPORAL (Para todos os modelos) ---
        # Em vez de random_state, cortamos os últimos 20% dos dados para validação
        split_idx = int(len(features) * 0.8)
        if self.model_type == MLModelType.LSTM:
             # Cria sequências
            window_size = 30
            X_seq, y_seq = self.create_sequences(features.values, labels.values, window_size)

            # Split de treino/validação
            X_train, X_val, y_train, y_val = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=42
            )

            # Mostra distribuição das classes
            unique, counts = np.unique(y_train, return_counts=True)
            class_distribution = dict(zip(unique, counts))
            logging.info(f"📊 Distribuição das classes no treino (LSTM): {class_distribution}")

            # Calcula pesos inversamente proporcionais à frequência
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = {i: w for i, w in enumerate(class_weights)}
            logging.info(f"⚖️ Pesos das classes: {class_weight_dict}")

            # Treinamento com pesos
            model, scaler, history = self.train_lstm(X_train, y_train, X_val, y_val, class_weight=class_weight_dict)

            # Salvamento do modelo e scaler
            model_path_keras = self.MODEL_PATH.replace(".pkl", ".keras")
            scaler_path = self.MODEL_PATH.replace(".pkl", "_scaler.pkl")

            model.save(model_path_keras, include_optimizer=True)
            joblib.dump(scaler, scaler_path)

            logging.info(f"💾 Modelo LSTM salvo em: {model_path_keras}")
            logging.info(f"💾 Scaler salvo em: {scaler_path}")
            return

        # ALTERAR AQUI: Split temporal para RF/XGB (Sem baralhar os candles)
        X_train_raw, X_val = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train_raw, y_val = labels.iloc[:split_idx], labels.iloc[split_idx:]

        # ALTERAR AQUI: Aplicar o SMOTE APENAS no treino
        logging.info("⚖️ Aplicando SMOTE apenas nos dados de treino...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)

        logging.info(f"📊 Distribuição após SMOTE (Treino): {np.bincount(y_train)}")
        logging.info(f"📊 Distribuição Real (Validação): {np.bincount(y_val)}")

        # Modelos clássicos (RF, XGB, MLP)
        model = self._init_model()
        param_grid = self._get_param_grid()

        if model is not None:

            if self.use_gridsearch and param_grid:
                logging.info(f"🔎 Iniciando GridSearchCV para {self.model_type.value}...")
                grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                logging.info(f"✅ Melhor modelo: {grid.best_params_}")
            else:
                model.fit(X_train, y_train)

            # Avaliação
            y_val_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_val_pred)
            logging.info(f"Accuracy na validação: {acc:.4f}")
            logging.info("Classification Report:")
            logging.info(f"\n {classification_report(y_val, y_val_pred)}")

        # Salvar modelo
        joblib.dump(model, self.MODEL_PATH)
        logging.info(f"💾 Modelo salvo em: {self.MODEL_PATH}")

        if self.save_img:
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['hold', 'buy', 'sell'] , yticklabels=['hold', 'buy', 'sell'])# type: ignore
            plt.xlabel("Predito")
            plt.ylabel("Verdadeiro")
            plt.title("Matriz de Confusão")
            os.makedirs(self.IMG_DIR, exist_ok=True)
            img_path = os.path.join(self.IMG_DIR, f"confusion_matrix_{self.model_type.value.lower()}.png")
            plt.savefig(img_path)
            logging.info(f"🖼️ Matriz de confusão salva em: {img_path}")
            plt.close()


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





