import asyncio
import logging
import os
import sys

import numpy as np
import optuna

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from commons.enums.strategy_enum import StrategyEnum
from commons.enums.timeframe_enum import TimeframeEnum
from commons.models.strategy_params import StrategyParams
from commons.utils.config_loader import PairConfig, load_pair_configs
from commons.utils.load_params import LoadParams
from tests.backtest_runner import BacktestRunner
from tests.save_json import SaveJson


class OptunaSearch:
    WEIGHT_NAMES = [
        "weights_trend",         # EMA
        "weights_momentum",
        "weights_oscillators",
        "weights_price_action", # candle, setup 123, breakout, bandas
        "weights_price_levels",
    ]


    async def run_optuna_search(self, runner:BacktestRunner, n_trials=50):

        trial_logs = []

        def suggest_weights(trial, min_weight=0.10):
            while True:
                raw = [trial.suggest_float(f"w_{i}", min_weight, 1.0) for i in range(len(self.WEIGHT_NAMES))]
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
                "weights_proximity_to_bands",
                "weights_exhaustion",
                "weights_penalty_factor",
                "weights_macd",
                "weights_cci",
                "weights_confirmation_candle_penalty",
                "weights_divergence"
            ]
            for i, name in enumerate(self.WEIGHT_NAMES):
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
                "tp_multiplier_aggressive": trial.suggest_float("tp_multiplier_aggressive", 2.0, 4.0),
                "sl_multiplier_conservative": trial.suggest_float("sl_multiplier_conservative", 1.0, 3.0),
                "tp_multiplier_conservative": trial.suggest_float("tp_multiplier_conservative", 2.0, 4.0),
                "volume_threshold_ratio": trial.suggest_float("volume_threshold_ratio", 0.2, 0.6),
                "atr_threshold_ratio": trial.suggest_float("atr_threshold_ratio", 0.3, 0.7),
                #"block_lateral_market":  trial.suggest_categorical("block_lateral_market", [True, False]),

                "block_lateral_market":  trial.suggest_categorical("block_lateral_market", [True, False]),

                **dict(zip(self.WEIGHT_NAMES, suggest_weights(trial)))
            }

            params_obj = StrategyParams(**params)
            summary = await runner.run(params_obj)

            neg_wins = -summary["total_pnl"]

            #score = -(summary['total_pnl'] * (summary['wins'] / summary['trades']))  
            # Regista o log numa string, não imprime direto
            trial_logs.append(
                f"[Trial {trial.number}] Wins: {summary['wins']}, Trades: {summary['total_trades']}, "
                f"PnL: {summary['total_pnl']:.2f}, Score (neg wins): {neg_wins:.4f}"
            )

            return neg_wins, summary, params
            #return -summary['total_pnl']  # minimizar lucro negativo = maximizar lucro positivo

        def sync_objective(trial):
            loop = asyncio.get_event_loop()
            score, summary, params = loop.run_until_complete(objective(trial))
            # Guardar os dados no trial para depois usar
            trial.set_user_attr("wins", summary['wins'])
            trial.set_user_attr("trades", summary['total_trades'])
            trial.set_user_attr("pnl", summary['total_pnl'])
            trial.set_user_attr("params", params)
            return score

        study = optuna.create_study(direction="minimize")
        study.optimize(sync_objective, n_trials=n_trials, n_jobs=4)

        # Imprime tudo no fim, ordenado pelos trials
        print("\n--- Logs de todos os trials ---")
        for log_line in trial_logs:
            print(log_line)

        study = optuna.create_study(direction="minimize")
        study.optimize(sync_objective, n_trials=n_trials, n_jobs=4)

        # Agora pega todos os trials válidos e cria o ranking multi-atributo
        valid_trials = [t for t in study.trials if t.value is not None and t.user_attrs.get("trades", 0) > 0]

        wins = np.array([t.user_attrs["wins"] for t in valid_trials])
        trades = np.array([t.user_attrs["trades"] for t in valid_trials])
        pnls = np.array([t.user_attrs["pnl"] for t in valid_trials])
        winrate = wins / trades

        # Normalização simples
        def normalize(arr):
                arr = np.array(arr)
                if len(arr) == 0:
                    return np.array([])  # ou np.zeros_like(arr) se quiseres zeros
                return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


        norm_wins = normalize(wins)
        norm_pnl = normalize(pnls)
        norm_winrate = normalize(winrate)

        # Pesos para cada métrica
        w1, w2, w3 = 0.4, 0.4, 0.2
        scores = w1 * norm_wins + w2 * norm_pnl + w3 * norm_winrate

        # Ordena e pega top 3
        top_indices = np.argsort(scores)[::-1][:3]

        print("\n--- Top 3 parâmetros com ranking multi-atributo ---")
        best_result = []
        for rank, idx in enumerate(top_indices, 1):
            trial = valid_trials[idx]
            params_converted = convert_weights_keys(trial.user_attrs["params"].copy())
            best_result.append({
                "profit": pnls[idx],
                "wins": wins[idx],
                "trades": trades[idx],
                "winrate": winrate[idx],
                "params": params_converted
            })
            print(f"Top {rank}: Wins={wins[idx]}, Trades={trades[idx]}, Winrate={winrate[idx]:.2f}, PnL={pnls[idx]:.2f}")
            print(f"Params: {params_converted}\n")

        return best_result
# Execução principal
async def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    pairs = load_pair_configs()

    #pair = PairConfig(symbol="SOL/USDC:USDC", leverage=10, capital=1, min_profit_abs= 5.0)

    for pair in pairs:

        backtestRunner = BacktestRunner(StrategyEnum.AI_SUPERTREND, TimeframeEnum.M15, pair, 250)
        optuna_search = OptunaSearch()

        result = await optuna_search.run_optuna_search(backtestRunner)

        SaveJson.append_best_params_to_json(result, pair.symbol)

    #print(LoadParams.load_best_params_with_weights())

if __name__ == "__main__":
    asyncio.run(main())