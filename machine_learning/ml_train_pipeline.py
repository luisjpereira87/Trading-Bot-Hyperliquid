import hashlib
import json
import logging
import os

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
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
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
                min_child_samples=150,  # Cada regra tem de cobrir pelo menos 150 candles
                class_weight='balanced',
                reg_alpha=0.5,  # Regularização L1
                reg_lambda=0.5,  # Regularização L2
                random_state=42
            )
            """

            return LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                class_weight='balanced',
                random_state=42,
                importance_type='gain',
            )

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
        # 1. Definir um limite fixo para todas as janelas
        limit = 1500
        ms_per_candle = 15 * 60 * 1000  # 15 min

        # Em vez de calcular "since" de forma arbitrária,
        # vamos definir pontos de partida claros baseados no AGORA
        now_ms = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)

        windows = [
            (f"period_{i}", now_ms - (i * limit * ms_per_candle))
            for i in range(1, 11)
        ]

        all_dfs = []

        exchange = binance({
            "enableRateLimit": True,
            "testnet": False,
        })

        try:
            for name, start_time in windows:
                # Loop de segurança para garantir que trazemos o LIMIT

                ohlcv_data = await exchange.fetch_ohlcv(symbol, self.TIMEFRAME, since=int(start_time), limit=limit)

                # Aceder aos dados (ajusta conforme a tua estrutura OhlcvWrapper)
                # actual_data = ohlcv_data.ohlcv if hasattr(ohlcv_data, 'ohlcv') else ohlcv_data

                df_temp = pd.DataFrame(ohlcv_data,
                                       columns=["timestamp", "open", "high", "low", "close", "volume"])

                # Validação Crítica
                if len(df_temp) < limit:
                    logging.warning(
                        f"⚠️ {symbol} - Janela '{name}' incompleta: {len(df_temp)}/{limit}. Histórico insuficiente na Exchange?")

                all_dfs.append(df_temp)
                # logging.info(f"✅ {symbol} - Janela '{name}' obtida: {len(df_temp)} candles.")
                real_start = pd.to_datetime(df_temp['timestamp'].iloc[0], unit='ms').strftime('%Y-%m-%d %H:%M')
                real_end = pd.to_datetime(df_temp['timestamp'].iloc[-1], unit='ms').strftime('%Y-%m-%d %H:%M')
                since1 = pd.to_datetime(start_time, unit='ms').strftime('%Y-%m-%d %H:%M')

                logging.info(
                    f"✅ {symbol} [{name}] - Recebido: {len(df_temp)} candles | De {real_start} até {real_end} | since={since1}")
        finally:
            await exchange.close()

        # Concatena logo aqui para garantir um DF único por moeda

        final_df = pd.DataFrame(pd.concat(all_dfs, ignore_index=True)).drop_duplicates(
            subset=['timestamp']).sort_values('timestamp')

        return final_df

    def prepare_dataset(self, df):
        # Calcula indicadores e labels apenas para esta moeda
        df_with_ind, features_raw = MarketBrain.add_indicators(df, is_training=True)

        # GARANTIA ANTI-BATOTA:
        # Forçamos o modelo a ver APENAS as colunas que decidimos
        # features = features_raw[cols_model].copy()

        # Fundamental: Reset do índice para o modelo não decorar a "linha"
        features = features_raw.reset_index(drop=True)
        labels = df_with_ind["label"].reset_index(drop=True)

        return features, labels, df_with_ind

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

    def train_and_save_model(self, features, labels, symbol):
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
        logging.info(f"💾 Modelo salvo em: {model_path}")

        importances = model.feature_importances_
        # feature_names = features
        for name, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
            print(f"{name}: {imp}")

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
            features_pair, labels_pair, df_enriched = self.prepare_dataset(df_train_raw)

            # 3. Treino e Salvamento Individual
            # Ajusta esta função para aceitar o 'symbol' no nome do ficheiro
            # Ex: model_BTC_USDC.joblib
            self.train_and_save_model(features_pair, labels_pair, symbol=symbol)

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
            self.train_and_save_model(X_combined, y_combined, symbol="MASTER")

            print("\n✅ Ciclo completo: Modelos Individuais e Modelo MASTER gerados!")
        else:
            print("❌ Nenhum dado coletado para o treino combinado.")

        print("✅ Todos os modelos individuais foram treinados!")
