
from enum import Enum


class StrategyEnum(Enum):
    AI_SUPERTREND = 'ai_supertrend'
    ML_RANDOM_FOREST = 'ml_random_forest'
    ML_XGBOOST = 'ml_xgboost'
    ML_MLP = 'ml_mlp'
    ML_LSTM = 'ml_lsmt'
    CROSS_EMA = 'cross_ema'
    LUXALGO_SUPERTREND = 'luxalgo_supertrend'