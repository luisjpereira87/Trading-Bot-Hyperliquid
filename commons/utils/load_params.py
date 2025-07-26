import json
import os

from commons.enums.mode_enum import ModeEnum
from commons.models.strategy_params_dclass import StrategyParams


class LoadParams:

    @staticmethod
    def dict_to_strategy_params(params_dict: dict) -> StrategyParams:
        """
        weights_keys = ['weights_trend', 'weights_rsi', 'weights_stochastic', 'weights_price_action', 'weights_proximity_to_bands', 'weights_exhaustion',  'weights_penalty_factor', 'weights_macd', 'weights_cci',
                    'weights_confirmation_candle_penalty', 'weights_divergence']
        """
        
        weights_keys = [
            "weights_trend",         # EMA
            "weights_momentum",
            "weights_oscillators",
            "weights_price_action", # candle, setup 123, breakout, bandas
            "weights_price_levels",
        ]
        weights_values = []
        i = 0
        while f"w_{i}" in params_dict:
            weights_values.append(params_dict.pop(f"w_{i}"))
            i += 1

        strategy_params_data = params_dict.copy()

        for name, val in zip(weights_keys, weights_values):
            strategy_params_data[name] = val

        if 'mode' in strategy_params_data:
            mode_str = strategy_params_data['mode']
            strategy_params_data['mode'] = ModeEnum(mode_str)

        return StrategyParams(**strategy_params_data)

    @staticmethod
    def load_params_with_weights(pair: str, filename="config/best_results.json"):
        if not os.path.exists(filename):
            return []

        with open(filename, "r") as f:
            data = json.load(f)

        if pair not in data:
            return []

        results = []
        for entry in data[pair]:
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

    @staticmethod
    def load_best_params_with_weights(pair: str, filename="config/best_results.json") -> (StrategyParams | None):
        if not os.path.exists(filename):
            return None

        with open(filename, "r") as f:
            data = json.load(f)

        if pair not in data or not data[pair]:
            return None

        best_result = data[pair][0]
        params = best_result["params"]

        weights = []
        i = 0
        while f"w_{i}" in params:
            weights.append(params.pop(f"w_{i}"))
            i += 1

        print("Parametros:", params)
        print("Pesos:", weights)

        return LoadParams.dict_to_strategy_params(params)
