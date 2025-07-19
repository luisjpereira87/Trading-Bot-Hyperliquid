import asyncio
import logging
import os
import sys

import optuna

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from commons.enums.strategy_enum import StrategyEnum
from commons.enums.timeframe_enum import TimeframeEnum
from commons.models.strategy_params import StrategyParams
from commons.utils.config_loader import PairConfig
from commons.utils.load_params import LoadParams
from tests.backtest_runner import BacktestRunner
from tests.grid_search import GridSearch
from tests.save_json import SaveJson


class OptunaSearch:
    async def run_optuna_search(self, runner:BacktestRunner, n_trials=50):

        def suggest_weights(trial, min_weight=0.10):
            while True:
                raw = [trial.suggest_float(f"w_{i}", min_weight, 1.0) for i in range(5)]
                total = sum(raw)
                normalized = [w / total for w in raw]

                # Verifica se todos os normalizados são ≥ min_weight
                if all(w >= min_weight for w in normalized):
                    return normalized

                # Caso contrário, força o trial a ser descartado
                raise optuna.exceptions.TrialPruned()
            
        def convert_weights_keys(params_dict):
            weight_names = [
                "weights_trend",
                "weights_rsi",
                "weights_stochastic",
                "weights_price_action",
                "weights_proximity_to_bands"
            ]
            for i, name in enumerate(weight_names):
                w_key = f"w_{i}"
                if w_key in params_dict:
                    params_dict[name] = params_dict.pop(w_key)
            return params_dict

        async def objective(trial):
            params = {
                "mode": trial.suggest_categorical("mode", ["aggressive", "conservative"]),
                "multiplier": trial.suggest_float("multiplier", 1.0, 2.0),
                "adx_threshold": trial.suggest_int("adx_threshold", 10, 40),
                "rsi_buy_threshold": trial.suggest_int("rsi_buy_threshold", 20, 40),
                "rsi_sell_threshold": trial.suggest_int("rsi_sell_threshold", 60, 80),
                "sl_multiplier_aggressive": trial.suggest_float("sl_multiplier_aggressive", 1.0, 3.0),
                "tp_multiplier_aggressive": trial.suggest_float("tp_multiplier_aggressive", 2.0, 6.0),
                "sl_multiplier_conservative": trial.suggest_float("sl_multiplier_conservative", 1.0, 3.0),
                "tp_multiplier_conservative": trial.suggest_float("tp_multiplier_conservative", 2.0, 5.0),
                "volume_threshold_ratio": trial.suggest_float("volume_threshold_ratio", 0.2, 0.6),
                "atr_threshold_ratio": trial.suggest_float("atr_threshold_ratio", 0.3, 0.7),

                **dict(zip([
                    "weights_trend", "weights_rsi", "weights_stochastic", "weights_price_action",
                    "weights_proximity_to_bands", "weights_exhaustion"
                ], suggest_weights(trial)))
            }

            params_obj = StrategyParams(**params)
            summary = await runner.run(params_obj)
            return -summary['total_pnl']  # minimizar lucro negativo = maximizar lucro positivo

        def sync_objective(trial):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(objective(trial))

        study = optuna.create_study(direction="minimize")
        #study = optuna.create_study(direction="maximize")

        # Enqueue os 3 melhores para garantir que estão na base
        #top_params = [top1_dict, top2_dict, top3_dict]
        #for p in top_params:
        #    study.enqueue_trial(p)

        study.optimize(sync_objective, n_trials=n_trials, n_jobs=4)

        valid_trials = [t for t in study.trials if t.value is not None]
        sorted_trials = sorted(valid_trials, key=lambda t: t.value)

        best_result = []
        print("Top 3 melhores parâmetros:")
        for trial in sorted_trials[:3]:
            converted = convert_weights_keys(trial.params.copy())
            best_result.append({
                "profit": -trial.value,
                "params": converted,
            })
            print(f"Profit: {-trial.value}, Params: {converted}")

        print("\nTop 3 piores parâmetros:")
        for trial in sorted_trials[-3:]:
            converted = convert_weights_keys(trial.params.copy())
            print(f"Profit: {-trial.value}, Params: {converted}")

        return best_result
# Execução principal
async def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    pair = PairConfig(symbol="BTC/USDC:USDC", leverage=10, capital=1, min_profit_abs= 5.0)

    backtestRunner = BacktestRunner(StrategyEnum.AI_SUPERTREND, TimeframeEnum.M15, pair, 250)
    optuna_search = OptunaSearch()

    result = await optuna_search.run_optuna_search(backtestRunner)

    SaveJson.append_best_params_to_json(result)

    print(LoadParams.load_best_params_with_weights())

if __name__ == "__main__":
    asyncio.run(main())