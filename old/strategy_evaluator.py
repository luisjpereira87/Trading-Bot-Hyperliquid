import logging
from datetime import datetime


class StrategyEvaluator:
    def __init__(self):
        self.trades = {}

    def register_strategy(self, name):
        if name not in self.trades:
            self.trades[name] = []

    def record_trade(self, strategy_name, profit, timestamp=None):
        if timestamp is None:
            timestamp = datetime.utcnow()
        self.trades[strategy_name].append({
            'profit': profit,
            'timestamp': timestamp
        })

    def report(self):
        for strategy, trades in self.trades.items():
            total_profit = sum(t['profit'] for t in trades)
            count = len(trades)
            avg_profit = total_profit / count if count > 0 else 0
            print(f"Estratégia: {strategy}")
            print(f"  Trades executados: {count}")
            print(f"  Lucro total: {total_profit:.4f}")
            print(f"  Lucro médio por trade: {avg_profit:.4f}")
            print()
    
    def report_summary(self, top_n=5):
        # Calcula lucro total para cada estratégia
        strategy_profits = {strategy: sum(t['profit'] for t in trades) for strategy, trades in self.trades.items()}
        # Calcula número de trades para cada estratégia
        strategy_trades = {strategy: len(trades) for strategy, trades in self.trades.items()}

        # Ordena estratégias por lucro total (descendente)
        sorted_strategies = sorted(
            strategy_profits.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print(f"\n=== TOP {top_n} Estratégias ===")
        for name, profit in sorted_strategies[:top_n]:
            trades = strategy_trades.get(name, 0)
            avg_profit = profit / trades if trades > 0 else 0
            print(f"{name} | Trades: {trades} | Lucro total: {profit:.4f} | Lucro médio/trade: {avg_profit:.4f}")

        print(f"\n=== WORST {top_n} Estratégias ===")
        for name, profit in sorted_strategies[-top_n:]:
            trades = strategy_trades.get(name, 0)
            avg_profit = profit / trades if trades > 0 else 0
            print(f"{name} | Trades: {trades} | Lucro total: {profit:.4f} | Lucro médio/trade: {avg_profit:.4f}")
            
