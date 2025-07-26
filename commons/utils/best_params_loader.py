import json
import os

from commons.enums.mode_enum import ModeEnum
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.load_params import LoadParams


class BestParamsLoader:
    def __init__(self, filename="config/best_results.json"):
        self.filename = filename
        self.data = self._load_all()

    def _load_all(self) -> dict:
        if not os.path.exists(self.filename):
            return {}
        with open(self.filename, "r") as f:
            return json.load(f)

    def get_all_pairs(self) -> list[str]:
        return list(self.data.keys())

    def get_pair_results(self, pair: str) -> list[dict]:
        entries = self.data.get(pair, [])
        results = []

        for entry in entries:
            params = entry["params"]

            if "mode" in params:
                params["mode"] = ModeEnum(params["mode"])

            weights = []
            i = 0
            while f"w_{i}" in params:
                weights.append(params.pop(f"w_{i}"))
                i += 1

            params["weights"] = weights

            results.append({
                "profit": entry.get("profit", 0),
                "params": params
            })

        return results

    def get_best_strategy_params(self, symbol: str) -> StrategyParams | None:
        all_data = self._load_all()
        if symbol not in all_data or not all_data[symbol]:
            return None

        best_result = all_data[symbol][0]
        raw_params = best_result["params"]

        # TRANSFORMA a lista "weights" em "w_0", "w_1", etc
        weights = raw_params.pop("weights", [])
        for i, w in enumerate(weights):
            raw_params[f"w_{i}"] = w

        return LoadParams.dict_to_strategy_params(raw_params)
