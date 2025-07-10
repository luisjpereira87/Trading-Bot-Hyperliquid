# trading_bot/main.py
import asyncio
import logging
import os
import sys

import ccxt.async_support as ccxt
from dotenv import load_dotenv

from commons.utils.config_loader import load_pair_configs
from machine_learning.ml_train_pipeline import MLTrainer
from old.order_manager import OrderManager
from strategies.ml_strategy import MLModelType
from tests.test_custom import TestCustom
from tests.test_run import BacktestRunner
from trading_bot.bot import TradingBot
from trading_bot.exchange_client import ExchangeClient
from trading_bot.exit_logic import ExitLogic
from trading_bot.trading_helpers import TradingHelpers

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

    timeframe = "15m"
    atr_period = 14
    pairs = load_pair_configs()

    #last_candle_times: dict[str, datetime | None] = {pair.symbol: None for pair in self.pairs}

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
    #exit_logic = ExitLogic(helpers, order_manager)

    exchange_client = ExchangeClient(exchange, wallet_address)
    bot = TradingBot(exchange, exchange_client, wallet_address,helpers,pairs,timeframe,atr_period)
    await bot.start() 

async def run_train():
    print("🤖 A treinar o modelo ML...")
    mlTrainer = MLTrainer()
    await mlTrainer.run() 

    #mlTrainer = MLTrainer(MLModelType.XGBOOST)
    #await mlTrainer.run() 

    #mlTrainer = MLTrainer(MLModelType.MLP)
    #await mlTrainer.run() 
    

async def run_backtest():
    print("📊 A executar backtest...")
    backtestRunner = BacktestRunner()
    await backtestRunner.run() 

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
        elif comando == "backtest":
            asyncio.run(run_backtest())
        elif comando == "customtest":
            asyncio.run(run_custom_test())
        else:
            print(f"❌ Comando desconhecido: {comando}")
            print("Usa: python main.py [treino | backtest]")

