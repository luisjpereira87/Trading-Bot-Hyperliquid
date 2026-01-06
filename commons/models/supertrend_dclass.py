
from dataclasses import dataclass


@dataclass
class Supertrend:
    ts: list
    perf_ama: list
    direction: list
    score: list
    delta_vol: list
    retest: list