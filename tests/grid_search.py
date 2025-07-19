import asyncio
import logging
import os
import sys

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from commons.enums.strategy_enum import StrategyEnum
from commons.enums.timeframe_enum import TimeframeEnum
from commons.models.strategy_params import StrategyParams
from commons.utils.config_loader import PairConfig
from tests.backtest_runner import BacktestRunner
from tests.grid_search_param import GridSearchParams


class GridSearch:
    async def run_grid_search(self, runner:BacktestRunner):

        samples = list(GridSearchParams.get_random_param_samples(n_samples=50, seed=42))
        results = []

        for params in samples:
            params_obj = StrategyParams(**params)
            summary = await runner.run(params_obj)

            results.append({'params': params, 'summary': summary})

        results_sorted = sorted(results, key=lambda x: x['summary']['total_pnl'], reverse=True)

        best_results = results_sorted[:3]
        print("Top 3 melhores parâmetros:")
        for r in results_sorted[:3]:
            print(f"Profit: {r['summary']['total_pnl']}, Params: {r['params']}")
            print(f"Total: {r['summary']['total_trades']}")
            print(f"Wins: {r['summary']['wins']}")
            print(f"Losses: {r['summary']['losses']}")

        print("\nTop 3 piores parâmetros:")
        for r in results_sorted[-3:]:
            print(f"Profit: {r['summary']['total_pnl']}, Params: {r['params']}")
            print(f"Total: {r['summary']['total_trades']}")
            print(f"Wins: {r['summary']['wins']}")
            print(f"Losses: {r['summary']['losses']}")
        
        return best_results

# Execução principal
async def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    pair = PairConfig(symbol="ETH/USDC:USDC", leverage=2, capital=0.5, min_profit_abs= 5.0)

    backtestRunner = BacktestRunner(StrategyEnum.AI_SUPERTREND, TimeframeEnum.M15, pair, 200)
    gridSearch = GridSearch()

    await gridSearch.run_grid_search(backtestRunner)

if __name__ == "__main__":
    asyncio.run(main())
