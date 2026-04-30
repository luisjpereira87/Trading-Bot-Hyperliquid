from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Metadata:
    symbol: str
    threshold_sell: float
    threshold_buy: float
    efficiency_min: float
    val_accuracy: float
    f1_macro: float
    features_count: float
    model_signature: str
    timestamp: float

    @property
    def is_fresh(self) -> bool:
        last_modified = datetime.fromtimestamp(self.timestamp)
        return datetime.now() < last_modified + timedelta(days=7)
