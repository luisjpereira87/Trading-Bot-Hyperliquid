from dataclasses import dataclass
from typing import List, Optional

from commons.utils.ohlcv_wrapper import OhlcvWrapper


@dataclass
class OhlcvFormat:
    ohlcv: OhlcvWrapper
    ohlcv_higher: Optional[OhlcvWrapper] = None
  