from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from commons.enums.signal_enum import Signal


@dataclass
class SignalResult:
    signal: Signal
    sl: Optional[Union[float, None]] = field(default=None)
    tp: Optional[Union[float, None]] = field(default=None)
    confidence: Optional[Union[float, None]] = field(default=None)
    score: float = 0
    previous_signal: Signal = Signal.HOLD

    def __post_init__(self):
        # Converte sl e tp para float nativo, se forem np.generic (ex: np.float64)
        if isinstance(self.sl, np.generic):
            self.sl = self.sl.item()
        if isinstance(self.tp, np.generic):
            self.tp = self.tp.item()
        if isinstance(self.confidence, np.generic):
            self.confidence = self.confidence.item()
        if isinstance(self.score, np.generic):
            self.score = self.score.item()

@dataclass
class SignalScore:
    signal: Signal
    score: Optional[Union[float, None]] = field(default=None)

    def __post_init__(self):
        if isinstance(self.score, np.generic):
            self.score = self.score.item()