# backtesting/backtest_runner.py

import asyncio
import logging

from ccxt.async_support import hyperliquid

from tests.strategy_manager import StrategyManager


class DummyExchange:
    def __init__(self, real_exchange, symbol, timeframe, limit=100):
        self.symbol = symbol
        self.timeframe = timeframe
        self.ohlcv_data = []
        self.limit = limit
        self.index = 0
        self.loaded = False
        self.real_exchange = real_exchange

    async def load_data(self):
        if not self.loaded:
            self.ohlcv_data = await self.real_exchange.fetch_ohlcv(
                self.symbol, timeframe=self.timeframe, limit=self.limit
            )
            self.loaded = True
            self.index = 50  # para garantir que os indicadores possam ser calculados

    async def fetch_ohlcv(self, symbol, timeframe=None, *args, **kwargs):
        return self.ohlcv_data[:self.index]

    def next(self):
        if self.index < len(self.ohlcv_data):
            self.index += 1


class BacktestRunner:
    def __init__(self, symbol="ETH/USDC:USDC", timeframe="15m", limit=500):
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.exchange = hyperliquid({
            "enableRateLimit": True,
            "testnet": True,
        })
        self.dummy = DummyExchange(self.exchange, self.symbol, self.timeframe, limit=self.limit)
        self.manager = StrategyManager(self.dummy, self.symbol, self.timeframe)

    async def run(self):
        await self.dummy.load_data()
        logging.info("🚀 Iniciando o backtest...")

        while True:
            await self.manager.run_strategies()
            self.dummy.next()

            if self.dummy.index >= len(self.dummy.ohlcv_data):
                logging.info("🏁 Fim dos dados OHLCV, encerrando backtest.")
                break

        self.manager.report()
        await self.exchange.close()
