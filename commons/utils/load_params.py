import json

from commons.enums.mode_enum import ModeEnum
from commons.models.strategy_params import StrategyParams


class LoadParams:

    @staticmethod
    def dict_to_strategy_params(params_dict: dict) -> StrategyParams:
        # Extrair os pesos w_0, w_1, ... e remover do dict
        weights_keys = ['weights_trend', 'weights_rsi', 'weights_stochastic', 'weights_price_action', 'weights_proximity_to_bands', 'weights_exhaustion']
        weights_values = []
        i = 0
        while f"w_{i}" in params_dict:
            weights_values.append(params_dict.pop(f"w_{i}"))
            i += 1
        
        # Prepara dicionário com os nomes corretos dos campos para StrategyParams
        strategy_params_data = params_dict.copy()
        
        # Mapeia os pesos para os nomes do dataclass
        for name, val in zip(weights_keys, weights_values):
            strategy_params_data[name] = val
        
        # Converte o modo de string para ModeEnum
        if 'mode' in strategy_params_data:
            mode_str = strategy_params_data['mode']
            strategy_params_data['mode'] = ModeEnum(mode_str)  # ou ModeEnum(mode_str), dependendo do enum
        
        # Cria a instância do dataclass
        return StrategyParams(**strategy_params_data)

    @staticmethod
    def load_params_with_weights(filename="best_results.json"):
        with open(filename, "r") as f:
            data = json.load(f)

        results = []
        for entry in data:
            params = entry["params"]

            # Converter o modo para enum, se precisares
            if "mode" in params:
                params["mode"] = ModeEnum(params["mode"])

            # Extrair pesos w_0, w_1, ..., ordenar por índice e criar lista de pesos
            weights = []
            i = 0
            while f"w_{i}" in params:
                weights.append(params.pop(f"w_{i}"))
                i += 1

            # Agora tens a lista de pesos pronta, junta ao dicionário ou passa separadamente
            params["weights"] = weights

            results.append({
                "profit": entry["profit"],
                "params": params
            })

        return results
    
    @staticmethod
    def load_best_params_with_weights(filename="config/best_results.json") -> StrategyParams:
        with open(filename, "r") as f:
            results = json.load(f)

        # Pega só o primeiro resultado
        best_result = results[0]

        params = best_result["params"]

        # Agora podes extrair os pesos
        weights = []
        i = 0
        while f"w_{i}" in params:
            weights.append(params.pop(f"w_{i}"))
            i += 1

        #params["weights"] = weights

        # params agora tem os outros parâmetros, weights tem os pesos numa lista ordenada
        print("Parametros:", params)
        print("Pesos:", weights)

        return LoadParams.dict_to_strategy_params(params)