# trading_bot/main.py
import asyncio
import logging
import os
import sys

import ccxt.async_support as ccxt
from dotenv import load_dotenv

from commons.enums.strategy_enum import StrategyEnum
from commons.enums.timeframe_enum import TimeframeEnum
from commons.helpers.trading_helpers import TradingHelpers
from commons.utils.config_loader import load_pair_configs
from machine_learning.ml_train_pipeline import MLTrainer
from strategies.ml_strategy import MLModelType
from strategies.strategy_manager import StrategyManager
from tests.test_custom import TestCustom
from trading_bot.bot import TradingBot
from trading_bot.exchange_client import ExchangeClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

async def run_bot():
    print("🔁 A correr o bot de trading...")

    wallet_address = os.getenv("WALLET_ADDRESS")
    private_key = os.getenv("PRIVATE_KEY")

    if not wallet_address or not private_key:
        raise ValueError(
            "Variáveis de ambiente WALLET_ADDRESS e PRIVATE_KEY devem estar definidas"
        )

    timeframe = TimeframeEnum.M15
    pairs = load_pair_configs()

    exchange = ccxt.hyperliquid(
        {
            "walletAddress": wallet_address,
            "privateKey":private_key,
            "testnet": True,
            "enableRateLimit": True,
            "options": {"defaultSlippage": 0.01},
        }
    )
    helpers = TradingHelpers()
    exchange_client = ExchangeClient(exchange, wallet_address)
    strategy = StrategyManager(exchange_client, StrategyEnum.ML_RANDOM_FOREST)
    bot = TradingBot(exchange_client, strategy, helpers, pairs, timeframe)

    await bot.start() 

async def run_train():
    print("🤖 A treinar o modelo ML...")
    mlTrainer = MLTrainer(MLModelType.RANDOM_FOREST)
    await mlTrainer.run() 

    #mlTrainer = MLTrainer(MLModelType.XGBOOST)
    #await mlTrainer.run() 

    #mlTrainer = MLTrainer(MLModelType.MLP)
    #await mlTrainer.run() 
    
"""
async def run_backtest():
    print("📊 A executar backtest...")
    backtestRunner = BacktestRunner()
    await backtestRunner.run() 
"""

async def run_custom_test():
    print("📊 A executar custom_test...")
    testCustom = TestCustom()
    await testCustom.run() 

if __name__ == "__main__":
    if len(sys.argv) < 2:
         asyncio.run(run_bot())
    else:
        comando = sys.argv[1].lower()
        if comando == "train":
            asyncio.run(run_train())
        elif comando == "customtest":
            asyncio.run(run_custom_test())
        else:
            print(f"❌ Comando desconhecido: {comando}")
            print("Usa: python main.py [treino | backtest]")

