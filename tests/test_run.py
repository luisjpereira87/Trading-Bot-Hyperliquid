import asyncio
import logging

from ccxt.async_support import hyperliquid

from tests.strategy_manager import StrategyManager

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


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
            self.index = 50  # Início para garantir indicadores calculáveis

    async def fetch_ohlcv(self, symbol, timeframe=None, *args, **kwargs):
        return self.ohlcv_data[:self.index]

    def next(self):
        if self.index < len(self.ohlcv_data):
            self.index += 1

async def main():
    logging.basicConfig(level=logging.INFO)

    symbol = "ETH/USDC:USDC"
    timeframe = "15m"

    exchange = hyperliquid({
        "enableRateLimit": True,
        "testnet": True,
    })

    dummy = DummyExchange(exchange, symbol, timeframe, limit=500)
    await dummy.load_data()

    manager = StrategyManager(dummy, symbol, timeframe)

    while True:
        await manager.run_strategies()
        dummy.next()

        if dummy.index >= len(dummy.ohlcv_data):
            logging.info("Chegou ao fim dos dados OHLCV, finalizando loop.")
            break

    manager.report()

    

    await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())