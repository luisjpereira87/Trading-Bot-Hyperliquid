from dataclasses import dataclass
from typing import List, Optional

from commons.enums.exit_type_enum import ExitTypeEnum
from commons.utils.ohlcv_wrapper import OhlcvWrapper


@dataclass
class TradeClosureInfo:
    exit_order: str
    pnl: float
    exit_type: ExitTypeEnum