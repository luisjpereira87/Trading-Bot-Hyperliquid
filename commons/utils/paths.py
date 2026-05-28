import json
import logging
import os
from pathlib import Path

from pydantic import ValidationError

from commons.models.metadata_dclass import Metadata

# 1. Encontrar a Raiz do Projeto de forma robusta
# Sobe 3 níveis a partir de commons/utils/paths.py para chegar à raiz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Definir o Diretório de Modelos
# No Railway, vamos definir uma variável STORAGE_PATH = /data/ no painel
# Se não existir, ele cria uma pasta 'models_storage' na raiz do projeto
MODEL_STORAGE = os.getenv("STORAGE_PATH", os.path.join(BASE_DIR, "models_storage"))

CONFIG_PATH = os.getenv("CONFIG_PATH", os.path.join(BASE_DIR, "config"))

# 3. Criar a pasta se não existir (evita erros de FileNotFoundError)
if not os.path.exists(MODEL_STORAGE):
    os.makedirs(MODEL_STORAGE, exist_ok=True)


def get_model_path(model_type_value: str, exchange_name: str, symbol: str, extension=".pkl"):
    """Retorna o caminho completo e absoluto para o modelo."""

    if exchange_name is not None:
        filename = f"modelo_{model_type_value.lower()}_{symbol}_{exchange_name}{extension}"
    else:
        filename = f"modelo_{model_type_value.lower()}_{symbol}{extension}"
    return os.path.join(MODEL_STORAGE, filename)


def get_bayesian_path(exchange_name: str, symbol: str):
    """Retorna o caminho completo e absoluto para o modelo."""

    if exchange_name is not None:
        filename = f"modelo_bayesian_{symbol}_{exchange_name}.json"
    else:
        filename = f"modelo_bayesian_{symbol}.json"

    return os.path.join(MODEL_STORAGE, filename)


def get_scaler_path(model_type_value: str, exchange_name: str, symbol: str):
    """Retorna o caminho para o scaler (específico para LSTM)."""

    if exchange_name is not None:
        filename = f"modelo_{model_type_value.lower()}_{symbol}_{exchange_name}_scaler.pkl"
    else:
        filename = f"modelo_{model_type_value.lower()}_{symbol}_scaler.pkl"

    return os.path.join(MODEL_STORAGE, filename)


def get_pairs_path():
    """Retorna o caminho para o pairs"""

    return os.path.join(CONFIG_PATH, "pairs.json")


def get_metadat_json_path(model_type_value: str, exchange_name: str, symbol: str):
    """Retorna o caminho completo e absoluto para o modelo."""

    if exchange_name is not None:
        filename = f"modelo_{model_type_value.lower()}_{symbol}_{exchange_name}_metadata.json"
    else:
        filename = f"modelo_{model_type_value.lower()}_{symbol}_metadata.json"

    file_path = Path(os.path.join(MODEL_STORAGE, filename))

    if not file_path.exists():
        logging.warning(f"⚠️ Arquivo de configuração '{filename}' não encontrado. Usando pares padrão.")
        return None

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON malformado em '{filename}': {e}. Usando pares padrão.")
        return None

    try:
        # validated = [Metadata(**item) for item in data]
        validated = Metadata(**data)
        logging.info(f"✅ {validated} pares carregados de '{filename}' com sucesso.")
        return validated
    except ValidationError as e:
        logging.error(f"❌ Erro de validação nos dados de '{filename}': {e}. Usando pares padrão.")
        return None

    return os.path.join(MODEL_STORAGE, filename)
