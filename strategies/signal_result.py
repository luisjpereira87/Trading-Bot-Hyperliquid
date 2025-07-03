from dataclasses import dataclass
from typing import Optional


@dataclass
class SignalResult:
    signal: str
    sl: Optional[float] = None
    tp: Optional[float] = None