import asyncio

import ccxt.async_support as ccxt


async def get_symbols():
    exchange = ccxt.hyperliquid({
        "testnet": True,
        'enableRateLimit': True
    }) # type: ignore
    await exchange.load_markets()
    print("Símbolos disponíveis:")
    for symbol in exchange.symbols: # type: ignore
        print(symbol)
    await exchange.close()

asyncio.run(get_symbols())