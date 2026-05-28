import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ccxt.async_support import binance
from lightgbm import LGBMClassifier
from seaborn import cm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import compute_class_weight

from commons.enums.timeframe_enum import TimeframeEnum
from commons.utils.config_loader import load_pair_configs
from commons.utils.paths import get_model_path, get_bayesian_path
from machine_learning.market_brain import MarketBrain
from trading_bot.exchange_base import ExchangeBase

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
    TIMEFRAME = TimeframeEnum.M15
    CANDLE_LIMIT = 10000

    def __init__(self, exchange: ExchangeBase, model_type=MLModelType.RANDOM_FOREST, save_csv=False, save_img=False,
                 use_gridsearch=False):
        self.model_type = model_type
        self.save_csv = save_csv
        self.save_img = save_img
        self.use_gridsearch = use_gridsearch
        self.exchange = exchange
        self.exchange_name = None

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.base_dir, "data")
        # self.MODEL_PATH = get_model_path(self.model_type.value, self.exchange_name)
        # self.BAYESIAN_PATH = get_bayesian_path(self.exchange_name)
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
        elif self.model_type == MLModelType.LIGHTGBM:
            custom_weights = {0: 15.0, 1: 1.0, 2: 15.0}

            """
            return LGBMClassifier(
                n_estimators=200,  # Menos árvores para não saturar
                learning_rate=0.03,
                num_leaves=15,  # Árvores rasas (essencial para poucos dados)
                max_depth=4,  # Força a generalização
                # min_child_samples=150,  # Cada regra tem de cobrir pelo menos 150 candles
                class_weight='balanced',
                reg_alpha=0.5,  # Regularização L1
                reg_lambda=0.5,  # Regularização L2
                random_state=42,
                # min_child_samples=100
                min_child_samples=30
            )

            """
            return LGBMClassifier(
                n_estimators=200,
                # learning_rate=0.015,
                learning_rate=0.02,
                # num_leaves=15,  # 🚨 Domado!
                num_leaves=31,  # 🧠 Cérebro mais detalhado
                # max_depth=4,  # 🚨 Domado!
                max_depth=6,  # Deixa as árvores crescerem livremente
                # min_child_samples=150,  # Exige 150 candles para criar uma regra
                min_child_samples=30,  # 🎯 Exige poucas amostras para criar regras
                feature_fraction=0.7,  # Não deixa usar todas as colunas ao mesmo tempo
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
            """
            return LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                class_weight='balanced',
                random_state=42,
                importance_type='gain',

                min_child_samples=50
            )
            """
            """
            return LGBMClassifier(
                n_estimators=1000,  # Mais árvores para refinamento
                learning_rate=0.02,  # Passo mais lento para maior precisão
                num_leaves=63,  # Aumenta a complexidade das decisões
                # num_leaves=31,
                max_depth=10,  # Evita que as árvores cresçam demasiado
                class_weight='balanced',  # Mantém para não ignorar sinais
                reg_alpha=0.1,  # L1 Regularization (limpa features inúteis)
                reg_lambda=0.1,  # L2 Regularization (evita pesos extremos)
                random_state=42,
                importance_type='gain',
                verbosity=-1,
                # min_data_in_leaf=100
            )
            """
            """
            return LGBMClassifier(
                n_estimators=500,  # Voltamos aos 500, mas com profundidade controlada
                learning_rate=0.02,  # Baixamos a velocidade para ele ser mais "atento"
                num_leaves=64,  # Aumentamos para ele conseguir distinguir melhor os detalhes
                max_depth=10,  # Permitimos árvores mais profundas
                min_child_samples=20,  # Garante que cada "regra" tem pelo menos 20 exemplos
                class_weight='balanced',
                random_state=42,
                importance_type='gain',

                # Mantemos o "Cadeado Java"
                n_jobs=1,
                deterministic=True,
                force_row_wise=True,
                bagging_seed=42,
                feature_fraction_seed=42
            )
            """
        else:
            raise ValueError("Modelo desconhecido")

    def _get_param_grid(self):
        if self.model_type == MLModelType.RANDOM_FOREST:
            return {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],  # 🚨 Antes tinhas até 20 (overfitting puro)
                'min_samples_leaf': [50, 150],  # Obriga as folhas a serem robustas
                'class_weight': ['balanced', None]
            }

        elif self.model_type == MLModelType.XGBOOST:
            return {
                'n_estimators': [100, 200],
                'max_depth': [3, 4, 5],  # 🚨 Árvores rasas para o BTC são muito melhores
                'learning_rate': [0.01, 0.03],  # Passos mais lentos e precisos
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8]  # Seleção aleatória de colunas
            }

        elif self.model_type == MLModelType.MLP:
            return {
                'hidden_layer_sizes': [(32,), (32, 16)],  # Redes mais simples para não decorar ruído
                'activation': ['tanh'],  # Tanh costuma ser mais estável em finanças
                'alpha': [0.001, 0.01],  # Regularização forte L2
                'learning_rate_init': [0.005]
            }

        elif self.model_type == MLModelType.LIGHTGBM:
            return {
                'n_estimators': [100, 300],
                'learning_rate': [0.01, 0.03],
                # 'num_leaves': [7, 15],  # 🚨 CRÍTICO: Antes tinhas até 50! Agora limitamos o crescimento
                'num_leaves': [15, 31],
                'max_depth': [3, 4],  # Controla a profundidade vertical
                # 'min_child_samples': [100, 200],  # 🚨 CRÍTICO: Equivalente ao min_data_in_leaf
                'min_child_samples': [20, 40],
                'feature_fraction': [0.7, 0.8]
            }
        else:
            return {}

    def _get_param_grid_(self):
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
        elif self.model_type == MLModelType.LIGHTGBM:
            return {
                'n_estimators': [100, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [20, 31, 50],
                'min_child_samples': [10, 20]
            }
        else:
            return {}

    async def fetch_ohlcv(self, symbol):
        limit_total = 5000
        binance_max_limit = 1000  # Limite real da API Binance
        ms_per_candle = 15 * 60 * 1000

        now_ms = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)

        # Definimos os pontos de partida para as 10 janelas
        windows = [
            (f"period_{i}", now_ms - (i * limit_total * ms_per_candle))
            for i in range(1, 16)
        ]

        all_dfs = []
        exchange = binance({"enableRateLimit": True, "testnet": False})

        try:
            for name, start_time in windows:
                window_data = []
                current_since = int(start_time)

                # --- NOVO LOOP DE PAGINAÇÃO ---
                while len(window_data) < limit_total:
                    # Calcula quanto falta para chegar aos 1500
                    remaining = limit_total - len(window_data)
                    # Pede o mínimo entre o que falta e o máximo da Binance
                    fetch_limit = min(remaining, binance_max_limit)

                    partial_data = await exchange.fetch_ohlcv(
                        symbol, self.TIMEFRAME, since=current_since, limit=fetch_limit
                    )

                    if not partial_data:
                        break

                    window_data.extend(partial_data)
                    # Atualiza o current_since para o milissegundo após o último candle recebido
                    current_since = partial_data[-1][0] + 1

                    # Pequena pausa para não stressar a API (opcional com enableRateLimit)
                    if len(window_data) < limit_total:
                        await asyncio.sleep(0.1)
                # ------------------------------

                df_temp = pd.DataFrame(window_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

                if len(df_temp) < limit_total:
                    logging.warning(f"⚠️ {symbol} - Janela '{name}' incompleta: {len(df_temp)}/{limit_total}")

                all_dfs.append(df_temp)

                real_start = pd.to_datetime(df_temp['timestamp'].iloc[0], unit='ms').strftime('%Y-%m-%d %H:%M')
                real_end = pd.to_datetime(df_temp['timestamp'].iloc[-1], unit='ms').strftime('%Y-%m-%d %H:%M')
                logging.info(f"✅ {symbol} [{name}] - Total: {len(df_temp)} candles (Paginado)")

        finally:
            await exchange.close()

        final_df = pd.DataFrame(pd.concat(all_dfs, ignore_index=True)).drop_duplicates(
            subset=['timestamp']).sort_values('timestamp')
        return final_df

    def prepare_dataset(self, df):
        # Calcula indicadores e labels apenas para esta moeda
        df_with_ind, features_raw = MarketBrain.add_indicators(df, is_training=True)

        # MarketBrain.plot_trading_channels(df_with_ind, num_candles=400)

        # GARANTIA ANTI-BATOTA:
        # Forçamos o modelo a ver APENAS as colunas que decidimos
        # features = features_raw[cols_model].copy()

        # Fundamental: Reset do índice para o modelo não decorar a "linha"
        features = features_raw.reset_index(drop=True)
        labels = df_with_ind["label"].reset_index(drop=True)
        sample_weight = df_with_ind["sample_weight"].reset_index(drop=True)

        return features, labels, sample_weight, df_with_ind

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
            X_seq.append(X[i:i + window_size])
            y_seq.append(y[i + window_size])
        return np.array(X_seq), np.array(y_seq)

    def load_lstm_model(self, symbol):
        model_path = get_model_path(self.model_type.value, self.exchange_name, symbol)

        model_path_keras = model_path.replace(".pkl", ".h5")
        if not os.path.exists(model_path_keras):
            raise FileNotFoundError(f"Modelo LSTM não encontrado em {model_path_keras}")
        model = tf.keras.models.load_model(model_path_keras)  # type: ignore
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
            # ModelCheckpoint("best_lstm_model.h5", monitor='val_accuracy', save_best_only=True, mode='max')
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
            X.append(features.iloc[i:i + seq_len].values)
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

    def train_bayesian_supervisor(self, df: pd.DataFrame, symbol):
        """
        Treina a camada de auditoria Bayesiana com as 6 âncoras do MarketBrain.
        """
        # 1. DISCRETIZAÇÃO (Criação dos Bins)
        # Criamos categorias lógicas para os teus indicadores
        symbol_clean = symbol.replace("/", "_").replace(":", "_")

        bayesian_path = get_bayesian_path(self.exchange_name, symbol_clean)

        # RSI: Exaustão
        df['rsi_bin'] = pd.cut(df['rsi'], bins=[0, 30, 70, 100], labels=['OVERSOLD', 'NEUTRAL', 'OVERBOUGHT'])

        # ADX: Força da Tendência
        df['adx_bin'] = pd.cut(df['adx'], bins=[0, 25, 50, 100], labels=['WEAK', 'STRONG', 'EXTREME'])

        # Super Score: O teu algoritmo (0-10)
        df['score_bin'] = pd.cut(df['super_score'], bins=[-1, 3, 7, 11], labels=['LOW', 'MID', 'HIGH'])

        # Choppiness: Qualidade do Movimento (<38 tendência, >61 lateral)
        df['chop_bin'] = pd.cut(df['choppiness'], bins=[0, 38, 61, 100], labels=['TRENDING', 'NORMAL', 'CHOPPY'])

        # Relative Volume: Confirmação (qcut para dividir em 3 partes iguais de frequência)
        df['vol_bin'] = pd.qcut(df['relative_volume'], q=3, labels=['LOW_VOL', 'AVG_VOL', 'HIGH_VOL'])

        # Above EMA200: Contexto (Já é 0 ou 1)
        df['ema_bin'] = df['above_ema200'].map({1: 'BULL', 0: 'BEAR'})

        # 2. CÁLCULO DAS PROBABILIDADES (Prior e Likelihoods)
        target_label = 2  # Focamos no teu sinal de COMPRA (1.2% de lucro)
        prior_win = float((df['label'] == target_label).mean())

        anchor_features = ['rsi_bin', 'adx_bin', 'score_bin', 'chop_bin', 'vol_bin', 'ema_bin']
        bayes_tables = {}

        for feat in anchor_features:
            # Calculamos a P(Win | Categoria)
            # Ex: Qual a % de vezes que o label foi 2 quando o rsi_bin foi OVERSOLD?
            prob_table = df.groupby(feat, observed=True)['label'].apply(
                lambda x: (x == target_label).mean()
            ).to_dict()
            bayes_tables[feat] = prob_table

        # 3. EXPORTAÇÃO DO "CÉREBRO"
        bayes_model = {
            "prior_win": prior_win,
            "tables": bayes_tables,
            "features_mapped": anchor_features,
            "updated_at": pd.Timestamp.now().isoformat()
        }

        # Guardamos no caminho que definiste no initialize
        with open(bayesian_path, 'w') as f:
            json.dump(bayes_model, f, indent=4)

        logging.info(f"🧠 Supervisor Bayesiano treinado com {len(df)} candles. Prior: {prior_win:.2%}")

    def train_and_save_model_(self, features, labels, symbol):
        # --- NOVIDADE: LIMPEZA DE NANs E SINCRONIZAÇÃO ---
        # 1. Garante que labels e features têm o mesmo índice e remove NaNs de ambos
        # Isso evita o erro "Input contains NaN" se a label de 2% falhou em algumas linhas
        combined = pd.DataFrame(pd.concat([features, labels], axis=1)).dropna()

        if len(combined) < 100:
            logging.error(f"❌ Dados insuficientes para {symbol} após limpeza. Pulando treino.")
            return

        # Separa novamente após a limpeza
        features = combined.iloc[:, :-1]
        labels = combined.iloc[:, -1]
        # -----------------------------------------------

        symbol_clean = symbol.replace("/", "_").replace(":", "_")
        model_path = get_model_path(self.model_type.value, self.exchange_name, symbol_clean)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        scaler_path = model_path.replace(".pkl", "_scaler.pkl")

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
            model_path_keras = model_path.replace(".pkl", ".keras")
            scaler_path = model_path.replace(".pkl", "_scaler.pkl")

            model.save(model_path_keras, include_optimizer=True)
            joblib.dump(scaler, scaler_path)

            logging.info(f"💾 Modelo LSTM salvo em: {model_path_keras}")
            logging.info(f"💾 Scaler salvo em: {scaler_path}")
            return

        # ALTERAR AQUI: Split temporal para RF/XGB (Sem baralhar os candles)
        X_train_raw, X_val_raw = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_val = labels.iloc[:split_idx], labels.iloc[split_idx:]

        scaler = StandardScaler()

        # O Scaler aprende no treino e aplica-se na validação
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        # Guardamos logo o scaler para não esquecer
        joblib.dump(scaler, scaler_path)
        logging.info(f"💾 Scaler salvo em: {scaler_path}")

        # Modelos clássicos (RF, XGB, MLP)
        model = self._init_model()
        param_grid = self._get_param_grid()

        if model is not None:

            if self.use_gridsearch and param_grid:
                logging.info(f"🔎 Iniciando GridSearchCV para {self.model_type.value}...")
                grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1, scoring='f1_macro')
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                logging.info(f"✅ Melhor modelo: {grid.best_params_}")
            else:
                logging.info(f"🚀 Treinando {self.model_type.value} com parâmetros padrão...")
                print(f"DEBUG - Features usadas: {features.columns.tolist()}")
                print(f"🔥 COLUNAS QUE ESTÃO A IR PARA O TREINO: {X_train_raw.columns.tolist()}")
                logging.info(f"Shape Treino: {X_train.shape}, Shape Validação: {X_val.shape}")
                # print(X_train.columns.tolist())
                model.fit(X_train, y_train)

                # Cria um "hash" das importâncias das colunas
                feat_imp = model.feature_importances_
                imp_hash = hashlib.md5(feat_imp.tobytes()).hexdigest()
                logging.info(f"🛡️ Model Signature (Hash): {imp_hash}")

            # Avaliação
            y_val_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_val_pred)
            logging.info(f"Accuracy na validação: {acc:.4f}")
            logging.info("Classification Report:")
            logging.info(f"\n {classification_report(y_val, y_val_pred)}")

        # Salvar modelo
        joblib.dump(model, model_path)

        # --- NOVIDADE: CÁLCULO E SALVAMENTO DE METADADOS (THRESHOLDS) ---
        metadata_path = model_path.replace(".pkl", "_metadata.json")

        # Pegamos as probabilidades na validação para calibrar o threshold
        y_val_probs = model.predict_proba(X_val)
        # y_val_probs terá o formato: [prob_sell, prob_hold, prob_buy] (classes 0, 1, 2)

        # Calculamos o melhor threshold para Sell (0) e Buy (2)
        t_sell = self.find_best_threshold(y_val.values, y_val_probs, 0)
        t_buy = self.find_best_threshold(y_val.values, y_val_probs, 2)

        metadata = {
            "symbol": symbol,
            "threshold_sell": round(t_sell, 3),
            "threshold_buy": round(t_buy, 3),
            "efficiency_min": 0.35,  # Teu filtro de ruído padrão
            "val_accuracy": round(float(acc), 4),
            "f1_macro": round(f1_score(y_val, y_val_pred, average='macro'), 4),
            "features_count": len(features.columns.tolist()),
            "model_signature": imp_hash,
            "timestamp": datetime.now().timestamp(),
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        logging.info(f"📊 Metadados e Thresholds salvos em: {metadata_path}")
        logging.info(f"🎯 Thresholds Otimizados -> BUY: {t_buy} | SELL: {t_sell}")
        # ---------------------------------------------------------------

        logging.info(f"💾 Modelo salvo em: {model_path}")

        """
        importances = model.feature_importances_
        # feature_names = features
        for name, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
            print(f"{name}: {imp}")
        """

        # --- CORREÇÃO: IMPRIMIR IMPORTÂNCIA DAS FEATURES EM QUALQUER CENÁRIO ---
        try:
            # Garantir que pegamos nas importâncias quer o modelo venha do GridSearch ou direto
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_importances_'):
                importances = model.best_estimator_.feature_importances_
            else:
                importances = None

            if importances is not None:
                logging.info("📊 --- RANKING DE IMPORTÂNCIA DOS INDICADORES ---")
                # Criamos uma lista ordenada para ver quem manda no modelo
                feature_ranking = sorted(zip(features.columns, importances), key=lambda x: x[1], reverse=True)

                for name, imp in feature_ranking:
                    # Mostra a percentagem de importância de cada indicador
                    print(f"🔹 {name:<30}: {imp:.4f}")
                logging.info("------------------------------------------------")
            else:
                logging.warning("⚠️ Não foi possível extrair a importância das features para este tipo de modelo.")

        except Exception as e:
            logging.error(f"❌ Erro ao gerar o ranking de indicadores: {str(e)}")

        if self.save_img:
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['hold', 'buy', 'sell'],
                        yticklabels=['hold', 'buy', 'sell'])  # type: ignore
            plt.xlabel("Predito")
            plt.ylabel("Verdadeiro")
            plt.title("Matriz de Confusão")
            os.makedirs(self.IMG_DIR, exist_ok=True)
            img_path = os.path.join(self.IMG_DIR, f"confusion_matrix_{self.model_type.value.lower()}.png")
            plt.savefig(img_path)
            logging.info(f"🖼️ Matriz de confusão salva em: {img_path}")
            plt.close()

    def train_and_save_model_old(self, features, labels, sample_weights, symbol):
        # --- NOVIDADE: LIMPEZA DE NANs E SINCRONIZAÇÃO COMPLETA COM PESOS ---
        # Colocamos os pesos no mesmo caldeirão de limpeza para garantir sincronia absoluta de índices
        combined = pd.DataFrame(pd.concat([features, labels, sample_weights], axis=1)).dropna()

        if len(combined) < 100:
            logging.error(f"❌ Dados insuficientes para {symbol} após limpeza. Pulando treino.")
            return

        # Separa novamente após a limpeza (features, labels e agora pesos)
        features = combined.iloc[:, :-2]  # Todas as colunas de indicadores e lags
        labels = combined.iloc[:, -2]  # A penúltima coluna (label)
        weights = combined.iloc[:, -1]  # A última coluna (sample_weight)
        # ---------------------------------------------------------------------

        symbol_clean = symbol.replace("/", "_").replace(":", "_")
        model_path = get_model_path(self.model_type.value, self.exchange_name, symbol_clean)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        scaler_path = model_path.replace(".pkl", "_scaler.pkl")

        # --- ALTERAÇÃO 1: SPLIT TEMPORAL (Para todos os modelos) ---
        # Cortamos os últimos 20% dos dados para validação (Sem baralhar os candles)
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
            model_path_keras = model_path.replace(".pkl", ".keras")
            scaler_path = model_path.replace(".pkl", "_scaler.pkl")

            model.save(model_path_keras, include_optimizer=True)
            joblib.dump(scaler, scaler_path)

            logging.info(f"💾 Modelo LSTM salvo em: {model_path_keras}")
            logging.info(f"💾 Scaler salvo em: {scaler_path}")
            return

        # --- SPLIT TEMPORAL PARA MODELOS CLÁSSICOS (RF, XGB, MLP) ---
        X_train_raw, X_val_raw = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_val = labels.iloc[:split_idx], labels.iloc[split_idx:]
        w_train = weights.iloc[:split_idx]  # Pesos que dão super-importância às grandes tendências

        scaler = StandardScaler()

        # O Scaler aprende no treino e aplica-se na validação
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        # Guardamos logo o scaler para não esquecer
        joblib.dump(scaler, scaler_path)
        logging.info(f"💾 Scaler salvo em: {scaler_path}")

        # Inicializar os Modelos clássicos
        model = self._init_model()
        param_grid = self._get_param_grid()

        if model is not None:

            if self.use_gridsearch and param_grid:
                logging.info(f"🔎 Iniciando GridSearchCV para {self.model_type.value}...")
                grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1, scoring='f1_macro')

                # Treino do GridSearch injetando os pesos financeiros
                grid.fit(X_train, y_train, sample_weight=w_train.values)
                model = grid.best_estimator_
                logging.info(f"✅ Melhor modelo: {grid.best_params_}")
            else:
                logging.info(f"🚀 Treinando {self.model_type.value} com parâmetros padrão e sample_weights...")
                print(f"DEBUG - Features usadas: {features.columns.tolist()}")
                print(f"🔥 COLUNAS QUE ESTÃO A IR PARA O TREINO: {X_train_raw.columns.tolist()}")
                logging.info(f"Shape Treino: {X_train.shape}, Shape Validação: {X_val.shape}")

                # 🔥 O ponto crítico: o modelo aprende a priorizar os movimentos com base nos pesos
                model.fit(X_train, y_train, sample_weight=w_train.values)

                # Cria um "hash" das importâncias das colunas
                feat_imp = model.feature_importances_
                imp_hash = hashlib.md5(feat_imp.tobytes()).hexdigest()
                logging.info(f"🛡️ Model Signature (Hash): {imp_hash}")

            # Avaliação na Validação (A validação corre SEM pesos para sabermos a performance real em mercado limpo)
            y_val_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_val_pred)
            logging.info(f"Accuracy na validação: {acc:.4f}")
            logging.info("Classification Report:")
            logging.info(f"\n {classification_report(y_val, y_val_pred)}")

        # Salvar modelo
        joblib.dump(model, model_path)

        # --- CÁLCULO E SALVAMENTO DE METADADOS (THRESHOLDS) ---
        metadata_path = model_path.replace(".pkl", "_metadata.json")

        # Pegamos as probabilidades na validação para calibrar o threshold
        y_val_probs = model.predict_proba(X_val)

        # Calculamos o melhor threshold para Sell (0) e Buy (2)
        t_otimo = self.find_best_threshold(y_val.values, y_val_probs)
        t_sell = t_otimo
        t_buy = t_otimo

        metadata = {
            "symbol": symbol,
            "threshold_sell": round(t_sell, 3),
            "threshold_buy": round(t_buy, 3),
            "efficiency_min": 0.35,
            "val_accuracy": round(float(acc), 4),
            "f1_macro": round(f1_score(y_val, y_val_pred, average='macro'), 4),
            "features_count": len(features.columns.tolist()),
            "model_signature": imp_hash if 'imp_hash' in locals() else "N/A",
            "timestamp": datetime.now().timestamp(),
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        logging.info(f"📊 Metadados e Thresholds salvos em: {metadata_path}")
        logging.info(f"🎯 Thresholds Otimizados -> BUY: {t_buy} | SELL: {t_sell}")

    def train_and_save_model(self, features, labels, sample_weights, symbol):
        # --- LIMPEZA DE NANs E SINCRONIZAÇÃO COMPLETA COM PESOS ---
        combined = pd.DataFrame(pd.concat([features, labels, sample_weights], axis=1)).dropna()

        if len(combined) < 100:
            logging.error(f"❌ Dados insuficientes para {symbol} após limpeza. Pulando treino.")
            return

        # Separa novamente após a limpeza (features, labels e pesos)
        features = combined.iloc[:, :-2]  # Todas as colunas de indicadores e lags
        labels = combined.iloc[:, -2]  # A penúltima coluna (label)
        weights = combined.iloc[:, -1]  # A última coluna (sample_weight)
        # ---------------------------------------------------------------------

        symbol_clean = symbol.replace("/", "_").replace(":", "_")
        model_path = get_model_path(self.model_type.value, self.exchange_name, symbol_clean)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        scaler_path = model_path.replace(".pkl", "_scaler.pkl")

        # Cortamos os últimos 20% dos dados para a validação final/calibração
        split_idx = int(len(features) * 0.8)

        # =========================================================================
        # 🧠 FLUXO 1: SE FOR LSTM (Blindado contra Data Leakage)
        # =========================================================================
        if self.model_type == MLModelType.LSTM:
            window_size = 30

            # 🚨 CORREÇÃO: O split da LSTM TEM de ser estritamente cronológico antes de criar as sequências
            X_train_raw, X_val_raw = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train_raw, y_val_raw = labels.iloc[:split_idx], labels.iloc[split_idx:]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_val_scaled = scaler.transform(X_val_raw)
            joblib.dump(scaler, scaler_path)

            # Agora criamos as sequências separadas, respeitando a linha do tempo
            X_train, y_train = self.create_sequences(X_train_scaled, y_train_raw.values, window_size)
            X_val, y_val = self.create_sequences(X_val_scaled, y_val_raw.values, window_size)

            # Mostra distribuição das classes
            unique, counts = np.unique(y_train, return_counts=True)
            class_distribution = dict(zip(unique, counts))
            logging.info(f"📊 Distribuição das classes no treino (LSTM): {class_distribution}")

            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = {int(i): float(w) for i, w in enumerate(class_weights)}
            logging.info(f"⚖️ Pesos das classes: {class_weight_dict}")

            model, scaler, history = self.train_lstm(X_train, y_train, X_val, y_val, class_weight=class_weight_dict)

            model_path_keras = model_path.replace(".pkl", ".keras")
            model.save(model_path_keras, include_optimizer=True)
            logging.info(f"💾 Modelo LSTM e Scaler salvos com sucesso.")
            return

        # =========================================================================
        # 🌳 FLUXO 2: MODELOS CLÁSSICOS (LightGBM, XGB, RF) COM TIME SERIES SPLIT
        # =========================================================================
        logging.info("======= ⏳ INICIANDO CROSS-VALIDAÇÃO CRONOLÓGICA (TIME SERIES SPLIT) =======")

        # Criamos o validador temporal por blocos com base nas features totais
        tscv = TimeSeriesSplit(n_splits=4)
        fold_accuracies = []

        for fold, (train_idx_cv, val_idx_cv) in enumerate(tscv.split(features)):
            X_tr_fold, X_val_fold = features.iloc[train_idx_cv], features.iloc[val_idx_cv]
            y_tr_fold, y_val_fold = labels.iloc[train_idx_cv], labels.iloc[val_idx_cv]
            w_tr_fold = weights.iloc[train_idx_cv]

            # Normalização temporária para avaliação do bloco
            scaler_fold = StandardScaler()
            X_tr_fold_sc = scaler_fold.fit_transform(X_tr_fold)
            X_val_fold_sc = scaler_fold.transform(X_val_fold)

            model_fold = self._init_model()

            # Treino do bloco injetando os pesos de castigo/recompensa financeiros correspondentes
            model_fold.fit(X_tr_fold_sc, y_tr_fold, sample_weight=w_tr_fold.values)

            preds_fold = model_fold.predict(X_val_fold_sc)
            acc_fold = accuracy_score(y_val_fold, preds_fold)
            fold_accuracies.append(acc_fold)

            logging.info(f"🟩 Bloco Temporal {fold + 1} - Accuracy: {acc_fold:.4f}")

        logging.info(f"📊 Média de Accuracy das Janelas Temporais: {np.mean(fold_accuracies):.4f}")
        logging.info("====================================================================")

        # ------------------------------------------------------------------------
        # 🚀 PRODUÇÃO: TREINO DO MODELO FINAL (O que vai controlar o teu dinheiro)
        # ------------------------------------------------------------------------
        # Usamos o teu corte clássico dos 80/20 originais para alimentar a versão final
        X_train_raw, X_val_raw = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_val = labels.iloc[:split_idx], labels.iloc[split_idx:]
        w_train = weights.iloc[:split_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        joblib.dump(scaler, scaler_path)
        logging.info(f"💾 Scaler de Produção salvo em: {scaler_path}")

        model = self._init_model()
        param_grid = self._get_param_grid()

        if model is not None:
            if self.use_gridsearch and param_grid:
                logging.info(f"🔎 Iniciando GridSearchCV para {self.model_type.value}...")
                # Alterado cv para usar TimeSeriesSplit para garantir que a otimização de parâmetros respeita o tempo
                grid = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=3), n_jobs=-1, verbose=1,
                                    scoring='f1_macro')
                grid.fit(X_train, y_train, sample_weight=w_train.values)
                model = grid.best_estimator_
                logging.info(f"✅ Melhor modelo: {grid.best_params_}")
            else:
                logging.info(f"🚀 Treinando {self.model_type.value} Final com sample_weights...")
                logging.info(f"Shape Treino: {X_train.shape}, Shape Validação: {X_val.shape}")

                # O modelo final de produção aprende a priorizar com base no teu array de pesos financeiro
                model.fit(X_train, y_train, sample_weight=w_train.values)

            try:
                feat_imp = model.feature_importances_
                imp_hash = hashlib.md5(feat_imp.tobytes()).hexdigest()
                logging.info(f"🛡️ Model Signature (Hash): {imp_hash}")
            except Exception:
                imp_hash = "N/A"

            # Avaliação no bloco de validação de produção
            y_val_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_val_pred)
            logging.info(f"Accuracy Final na validação: {acc:.4f}")
            logging.info("Classification Report Final:")
            logging.info(f"\n {classification_report(y_val, y_val_pred, zero_division=0)}")

        # Salvar o modelo validado e treinado
        joblib.dump(model, model_path)

        # --- CÁLCULO E SALVAMENTO DE METADADOS (THRESHOLDS) ---
        metadata_path = model_path.replace(".pkl", "_metadata.json")
        y_val_probs = model.predict_proba(X_val)

        t_sell = self.find_best_threshold(y_val.values, y_val_probs, class_idx=0)
        t_buy = self.find_best_threshold(y_val.values, y_val_probs, class_idx=2)

        metadata = {
            "symbol": symbol,
            "threshold_sell": round(t_sell, 3),
            "threshold_buy": round(t_buy, 3),
            "efficiency_min": 0.35,
            "val_accuracy": round(float(acc), 4),
            "f1_macro": round(f1_score(y_val, y_val_pred, average='macro', zero_division=0), 4),
            "features_count": len(features.columns.tolist()),
            "model_signature": imp_hash,
            "timestamp": datetime.now().timestamp(),
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        logging.info(f"📊 Metadados e Thresholds salvos em: {metadata_path}")
        logging.info(f"🎯 Thresholds Otimizados -> BUY: {t_buy} | SELL: {t_sell}")
        logging.info(f"💾 Modelo final salvo com sucesso em: {model_path}")

    def find_best_threshold(self, y_true, y_probs, class_idx):
        """
        Calcula o threshold ótimo de forma independente para uma classe específica.
        class_idx: 0 para SELL, 2 para BUY
        """
        best_t = 0.42
        best_precision = -1

        # Alargamos a busca (de 0.28 a 0.60) porque o class_weight='balanced'
        # espalha as probabilidades de forma diferente para cada classe
        for t in np.arange(0.28, 0.61, 0.01):

            # Filtramos apenas as previsões da classe atual para este threshold
            preds_classe = np.where(y_probs[:, class_idx] >= t, class_idx, 1)

            # Criamos o vetor real binário (é a classe que queremos ou é HOLD?)
            y_true_bin = np.where(y_true == class_idx, class_idx, 1)

            # Contamos quantos trades este threshold específico abriria
            total_trades = np.sum(preds_classe == class_idx)

            # FILTRO DE SANIDADE INTERGALACO (Mínimo de 20 trades na validação para evitar sorte)
            if total_trades < 20:
                continue

            # Calcular a precisão cirúrgica apenas para esta ação (BUY ou SELL)
            from sklearn.metrics import precision_score
            prec = precision_score(y_true_bin, preds_classe, pos_label=class_idx, zero_division=0)

            # Queremos o threshold que dá a maior precisão possível nas entradas
            if prec > best_precision:
                best_precision = prec
                best_t = t

        return float(best_t)

    def find_best_threshold_(self, y_true, y_probs, class_index):
        best_t = 0.45  # Default mais seguro
        best_score = 0

        # Testamos de 0.35 a 0.65 (evitamos o 0.30 que é muito ruidoso)
        for t in np.arange(0.35, 0.66, 0.01):
            preds = (y_probs[:, class_index] >= t).astype(int)
            actual = (y_true == class_index).astype(int)

            if np.sum(preds) < 15: continue  # Amostra mínima

            prec = precision_score(actual, preds, zero_division=0)
            rec = recall_score(actual, preds, zero_division=0)

            # MÉTRICA SNIPER: Precision vale o triplo do Recall
            # Queremos poucos sinais, mas que sejam certeiros
            success_score = (3 * prec + rec) / 4

            # Filtro de Sanidade: Se a precisão for menor que 22%, ignora o threshold
            if success_score > best_score and prec > 0.22:
                best_score = success_score
                best_t = t

        return float(best_t)

    def find_best_threshold_old(self, y_true, y_probs):
        best_t = 0.42  # Um ponto de partida muito mais realista e irreverente
        best_f1_trades = 0

        # Testamos de 0.35 a 0.55 (acima de 0.55 no BTC atual é utopia)
        for t in np.arange(0.35, 0.56, 0.01):

            # Criamos as previsões do robô: começa tudo em HOLD (1)
            preds_robot = np.ones(len(y_true))

            for i in range(len(y_probs)):
                if y_probs[i, 0] >= t:  # Se ultrapassar o t de SELL
                    preds_robot[i] = 0
                elif y_probs[i, 2] >= t:  # Se ultrapassar o t de BUY
                    preds_robot[i] = 2

            # Contamos quantos trades este threshold abriria no total da validação
            total_trades = np.sum(preds_robot == 0) + np.sum(preds_robot == 2)

            # 🚨 FILTRO DE SANIDADE: Se este threshold abrir menos de 30 trades
            # nas 14 mil velas, ele é BANIDO por ser demasiado "mão de vaca"
            if total_trades < 30:
                continue

                # Calcular o F1-Score real apenas para as classes de ação (0 e 2)
            f1_sell = precision_score((y_true == 0).astype(int), (preds_robot == 0).astype(int), zero_division=0)
            f1_buy = precision_score((y_true == 2).astype(int), (preds_robot == 2).astype(int), zero_division=0)

            # Média da precisão real dos trades abertos
            precision_media = (f1_sell + f1_buy) / 2

            # Queremos o threshold que maximize a precisão, desde que passe no filtro dos 30 trades
            if precision_media > best_f1_trades and precision_media > 0.25:
                best_f1_trades = precision_media
                best_t = t

        return float(best_t)

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

    def plot_feature_importance(self, model, feature_names, symbol):
        # Extrair importâncias (pelo Ganho)
        importances = model.feature_importances_

        # Criar DataFrame para visualização
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Plotar os Top 15
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='viridis')
        plt.title(f'🚀 Top 15 Features Decisivas - Modelo {symbol} (Gain)')
        plt.xlabel('Contribuição Total para a Decisão (Ganho)')
        plt.ylabel('Indicadores')
        plt.show()

        # Log das 5 principais para o terminal
        top_5 = feature_importance_df.head(5)['Feature'].tolist()
        logging.info(f"🏆 As 5 chaves do modelo {symbol}: {', '.join(top_5)}")

    async def run(self):
        pairs = load_pair_configs()

        # Listas para acumular dados para o Treino Combinado final
        all_features = []
        all_labels = []

        for pair_cfg in pairs:
            symbol = pair_cfg.symbol
            print(f"--- Treinando Modelo Especialista para: {symbol} ---")

            # 1. Dados específicos do par
            df_raw = await self.fetch_ohlcv(symbol)
            df_train_raw = df_raw.iloc[:-1000].copy()

            # 2. Prepara os dados
            features_pair, labels_pair, sample_weight, df_enriched = self.prepare_dataset(df_train_raw)

            # 3. Treino e Salvamento Individual
            # Ajusta esta função para aceitar o 'symbol' no nome do ficheiro
            # Ex: model_BTC_USDC.joblib
            self.train_and_save_model(features_pair, labels_pair, sample_weight, symbol=symbol)

            # 4. Treino do Bayes Individual (Se quiseres manter o Bayes por par)
            # self.train_bayesian_supervisor(df_enriched, symbol=symbol)

            # 4. ACUMULAÇÃO para o Combinado
            all_features.append(features_pair)
            all_labels.append(labels_pair)

        if all_features:
            print("\n" + "=" * 40)
            print("🧠 INICIANDO TREINO COMBINADO (MASTER)")
            print("=" * 40)

            # Junta todos os DataFrames de Features e Labels
            X_combined = pd.concat(all_features, axis=0)
            y_combined = pd.concat(all_labels, axis=0)

            print(f"📊 Dataset Master: {len(X_combined)} candles totais.")

            # Salva como: model_MASTER.joblib
            self.train_and_save_model(X_combined, y_combined, sample_weight, symbol="MASTER")

            print("\n✅ Ciclo completo: Modelos Individuais e Modelo MASTER gerados!")
        else:
            print("❌ Nenhum dado coletado para o treino combinado.")

        print("✅ Todos os modelos individuais foram treinados!")
