from enum import Enum, auto


class CandleType(Enum):
    BULL = auto()
    BEAR = auto()
    TOP_EXHAUSTION = auto()
    BOTTOM_EXHAUSTION = auto()
    DOJI = auto()
    NEUTRAL = auto()