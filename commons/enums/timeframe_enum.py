from enum import Enum


class TimeframeEnum(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    D1 = "1d"
    W1 = "1w"
    M1C = "1M"  # Monthly, renomeado para evitar conflito com M1 de 1 minuto

    def get_higher(self):
        mapping = {
            TimeframeEnum.M1: TimeframeEnum.M5,
            TimeframeEnum.M5: TimeframeEnum.M15,
            TimeframeEnum.M15: TimeframeEnum.H1,
            TimeframeEnum.M30: TimeframeEnum.H2,
            TimeframeEnum.H1: TimeframeEnum.H4,
            TimeframeEnum.H2: TimeframeEnum.H6,
            TimeframeEnum.H4: TimeframeEnum.D1,
            TimeframeEnum.H6: TimeframeEnum.D1,
            TimeframeEnum.D1: TimeframeEnum.W1,
            TimeframeEnum.W1: TimeframeEnum.M1C,
            TimeframeEnum.M1C: TimeframeEnum.M1C,  # já é o maior
        }
        return mapping.get(self, TimeframeEnum.H1)  # fallback para H1