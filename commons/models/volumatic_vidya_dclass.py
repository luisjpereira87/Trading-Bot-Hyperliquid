
from dataclasses import dataclass


@dataclass
class VolumaticVidya:
    vidya: list[float]
    upper_band: list[float]
    lower_band: list[float]
    is_trend_up: list[bool]
    smoothed: list[float]
    pivot_high: list[float]
    pivot_low: list[float]
    up_trend_volume: list[float]
    down_trend_volume: list[float]
    delta_volume_pct: list[float]
    retest: list[int]