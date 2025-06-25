# trading_bot/main.py
import asyncio
import logging
from bot import TradingBot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

async def main():
    bot = TradingBot()
    await bot.start()  # agora o próprio bot controla o intervalo com base nas velas

if __name__ == "__main__":
    asyncio.run(main())

