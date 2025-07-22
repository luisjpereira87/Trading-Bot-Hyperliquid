import json
import os
from enum import Enum


class SaveJson:

    @staticmethod
    def make_params_serializable(params: dict) -> dict:
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in params.items()
        }


    @staticmethod
    def append_best_params_to_json(results: list, pair: str, filename="config/best_results.json", top_n: int = 10):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                # Se for lista, converte para dict vazio
                if isinstance(data, list):
                    data = {}
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            data = {}

        if pair not in data:
            data[pair] = []

        existing_keys = {json.dumps(SaveJson.make_params_serializable(d["params"]), sort_keys=True) for d in data[pair]}

        for result in results:
            serializable_params = SaveJson.make_params_serializable(result["params"])
            key = json.dumps(serializable_params, sort_keys=True)
            if key not in existing_keys:
                data[pair].append({"profit": result["profit"], "params": serializable_params})

        # Ordenar e truncar só os do pair
        data[pair].sort(key=lambda x: x["profit"], reverse=True)
        data[pair] = data[pair][:top_n]

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

