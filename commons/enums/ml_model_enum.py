
from enum import Enum


class MLModelType(Enum):
    RANDOM_FOREST = 'RandomForest'
    XGBOOST = 'XGBoost'
    MLP = 'MLP'