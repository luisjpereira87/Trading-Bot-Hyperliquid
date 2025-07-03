from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from enums.signal_enum import Signal


@dataclass
class SignalResult:
    signal: Signal
    sl: Optional[Union[float, None]] = field(default=None)
    tp: Optional[Union[float, None]] = field(default=None)

    def __post_init__(self):
        # Converte sl e tp para float nativo, se forem np.generic (ex: np.float64)
        if isinstance(self.sl, np.generic):
            self.sl = self.sl.item()
        if isinstance(self.tp, np.generic):
            self.tp = self.tp.item()