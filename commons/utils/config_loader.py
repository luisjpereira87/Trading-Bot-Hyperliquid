import json
import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_JSON_PATH = PROJECT_ROOT / "config" / "pairs.json"


class PairConfig(BaseModel):
    symbol: str
    symbol_nado: str
    leverage: int = Field(gt=0)
    capital: float = Field(gt=0, le=1)

    # Novos parâmetros para ExitLogic
    min_profit_pct: float = Field(default=0.005, ge=0)  # 0.5% padrão
    max_hold_candles: int = Field(default=2, ge=1)


def load_pair_configs(file_path: Path = DEFAULT_JSON_PATH) -> List[PairConfig]:
    default_pairs = [
        PairConfig(symbol="BTC/USDC:USDC", symbol_nado="BTC/USDC:USDC", leverage=10, capital=0.3),
        PairConfig(symbol="ETH/USDC:USDC", symbol_nado="ETH/USDC:USDC", leverage=10, capital=0.3),
        PairConfig(symbol="SOL/USDC:USDC", symbol_nado="SOL/USDC:USDC", leverage=10, capital=0.3),
    ]

    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        logging.warning(f"⚠️ Arquivo de configuração '{file_path}' não encontrado. Usando pares padrão.")
        return default_pairs

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON malformado em '{file_path}': {e}. Usando pares padrão.")
        return default_pairs

    if not isinstance(data, list):
        logging.error(f"❌ Formato inválido em '{file_path}': esperado lista. Usando pares padrão.")
        return default_pairs

    try:
        validated = [PairConfig(**item) for item in data]
        logging.info(f"✅ {len(validated)} pares carregados de '{file_path}' com sucesso.")
        return validated
    except ValidationError as e:
        logging.error(f"❌ Erro de validação nos dados de '{file_path}': {e}. Usando pares padrão.")
        return default_pairs


def get_pair_by_symbol(symbol: str) -> Optional[PairConfig]:
    for pair in load_pair_configs():
        if pair.symbol == symbol:
            return pair
    return None  # se não encontrar


def get_pair_by_configs_symbol(pair_configs: List[PairConfig], symbol: str) -> Optional[PairConfig]:
    for pair in pair_configs:
        if pair.symbol == symbol:
            return pair
    return None  # se não encontrar
