import os

# 1. Encontrar a Raiz do Projeto de forma robusta
# Sobe 3 níveis a partir de commons/utils/paths.py para chegar à raiz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Definir o Diretório de Modelos
# No Railway, vamos definir uma variável STORAGE_PATH = /data/ no painel
# Se não existir, ele cria uma pasta 'models_storage' na raiz do projeto
MODEL_STORAGE = os.getenv("STORAGE_PATH", os.path.join(BASE_DIR, "models_storage"))

# 3. Criar a pasta se não existir (evita erros de FileNotFoundError)
if not os.path.exists(MODEL_STORAGE):
    os.makedirs(MODEL_STORAGE, exist_ok=True)

def get_model_path(model_type_value: str, exchange_name: str, extension=".pkl"):
    """Retorna o caminho completo e absoluto para o modelo."""
    filename = f"modelo_{model_type_value.lower()}_{exchange_name}{extension}"
    return os.path.join(MODEL_STORAGE, filename)

def get_scaler_path(model_type_value: str, exchange_name: str):
    """Retorna o caminho para o scaler (específico para LSTM)."""
    return os.path.join(MODEL_STORAGE, f"modelo_{model_type_value.lower()}_{exchange_name}_scaler.pkl")