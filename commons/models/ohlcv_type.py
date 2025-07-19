from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Ohlcv:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float