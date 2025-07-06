# trading_bot/main.py
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from machine_learning.ml_train_pipeline import MLTrainer
from strategies.ml_strategy import MLModelType
from tests.test_custom import TestCustom
from tests.test_run import BacktestRunner
from trading_bot.bot import TradingBot

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
    bot = TradingBot()
    await bot.start() 

async def run_train():
    print("🤖 A treinar o modelo ML...")
    mlTrainer = MLTrainer()
    await mlTrainer.run() 

    mlTrainer = MLTrainer(MLModelType.XGBOOST)
    await mlTrainer.run() 

    mlTrainer = MLTrainer(MLModelType.MLP)
    await mlTrainer.run() 
    

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

