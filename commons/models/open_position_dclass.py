from dataclasses import dataclass
from typing import List, Optional, Union

from commons.enums.signal_enum import Signal


@dataclass
class OpenPosition:
    side: Optional[Union[str, None]]
    size: float
    entry_price: float
    id: str
    notional: float
    sl: (float | None)
    tp: (float | None)