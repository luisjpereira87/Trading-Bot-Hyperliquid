import asyncio
import os
from datetime import datetime, timedelta

import pandas as pd
from ccxt.async_support import hyperliquid

# ==== CONFIGURAÇÕES ====
SYMBOLS = ["ETH/USDC:USDC", "BTC/USDC:USDC"]
TIMEFRAME = "15m"
START_DATE = "2024-01-01"
END_DATE = "2024-06-01"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
LIMIT = 1000  # máximo por request para Hyperliquid

# ================

exchange = hyperliquid({
    "enableRateLimit": True,
    "testnet": True,
})


def timeframe_to_milliseconds(tf: str) -> int:
    mapping = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    return mapping[tf]


async def fetch_ohlcv_range(symbol: str, timeframe: str, start_date: str, end_date: str):
    since = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_timestamp = int(pd.to_datetime(end_date).timestamp() * 1000)
    tf_ms = timeframe_to_milliseconds(timeframe)
    filename = f"{OUTPUT_DIR}/{symbol.replace('/', '_').replace(':', '_')}_{timeframe}.csv"

    print(f"\n🔄 Baixando {symbol} de {start_date} a {end_date} ({timeframe})")

    all_candles = []

    while since < end_timestamp:
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=LIMIT)
            if not ohlcv:
                print("⚠️ Sem mais dados.")
                break

            last_ts = ohlcv[-1][0]
            all_candles.extend(ohlcv)
            since = last_ts + tf_ms

        except Exception as e:
            print(f"❌ Erro: {e}")
            await asyncio.sleep(5)

    # Remove duplicatas
    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.drop_duplicates(subset="timestamp", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(filename)
    print(f"✅ Salvo: {filename}")


async def main():
    for symbol in SYMBOLS:
        await fetch_ohlcv_range(symbol, TIMEFRAME, START_DATE, END_DATE)
    await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())