from enum import Enum, auto


class CandleType(Enum):
    BULL = auto()
    BEAR = auto()
    TOP_EXHAUSTION = auto()
    BOTTOM_EXHAUSTION = auto()
    DOJI = auto()
    NEUTRAL = auto()
    WEAK_BULL = auto()
    STRONG_BULL = auto()
    WEAK_BEAR = auto()
    STRONG_BEAR = auto()