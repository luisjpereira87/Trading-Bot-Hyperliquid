import json
import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError


class PairConfig(BaseModel):
    symbol: str
    leverage: int = Field(gt=0)
    capital: float = Field(gt=0, le=1)
    min_profit_abs: float = Field(ge=0)
    
    # Novos parâmetros para ExitLogic
    min_profit_pct: float = Field(default=0.005, ge=0)  # 0.5% padrão
    max_hold_candles: int = Field(default=2, ge=1)


def load_pair_configs(path: str = "config/pairs.json") -> List[PairConfig]:
    default_pairs = [
        PairConfig(symbol="BTC/USDC:USDC", leverage=5, capital=0.1, min_profit_abs=10.0),
        PairConfig(symbol="ETH/USDC:USDC", leverage=10, capital=0.2, min_profit_abs=5.0),
    ]

    file_path = Path(path)
    if not file_path.exists():
        logging.warning(f"⚠️ Arquivo de configuração '{path}' não encontrado. Usando pares padrão.")
        return default_pairs

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON malformado em '{path}': {e}. Usando pares padrão.")
        return default_pairs

    if not isinstance(data, list):
        logging.error(f"❌ Formato inválido em '{path}': esperado lista. Usando pares padrão.")
        return default_pairs

    try:
        validated = [PairConfig(**item) for item in data]
        logging.info(f"✅ {len(validated)} pares carregados de '{path}' com sucesso.")
        return validated
    except ValidationError as e:
        logging.error(f"❌ Erro de validação nos dados de '{path}': {e}. Usando pares padrão.")
        return default_pairs
    
def get_pair_by_symbol(symbol: str) -> Optional[PairConfig]:
    for pair in load_pair_configs():
        if pair.symbol == symbol:
            return pair
    return None  # se não encontrar

