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
    def append_best_params_to_json(results: list, filename="config/best_results.json", top_n: int = 10):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            data = []

        for result in results:
            serializable_params = SaveJson.make_params_serializable(result["params"])

            existing_keys = {json.dumps(SaveJson.make_params_serializable(d["params"]), sort_keys=True) for d in data}
            key = json.dumps(serializable_params, sort_keys=True)

            if key not in existing_keys:
                data.append({"profit": result["profit"], "params": serializable_params})

        data.sort(key=lambda x: x["profit"], reverse=True)
        with open(filename, "w") as f:
            json.dump(data[:top_n], f, indent=4)
