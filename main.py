# trading_bot/main.py
import asyncio
import logging
import os
import sys

import ccxt.async_support as ccxt
from dotenv import load_dotenv

from commons.enums.signal_enum import Signal
from commons.enums.strategy_enum import StrategyEnum
from commons.enums.timeframe_enum import TimeframeEnum
from commons.helpers.trading_helpers import TradingHelpers
from commons.utils.config_loader import load_pair_configs
from machine_learning.ml_train_pipeline import MLTrainer
from strategies.ml_strategy import MLModelType
from strategies.strategy_manager import StrategyManager
from trading_bot.bot import TradingBot
from trading_bot.exchange_client import ExchangeClient
from trading_bot.nado_exchange_client import NadoExchangeClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

async def run_bot():
    logging.info("🔁 A iniciar motores de trading multi-exchange...")

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
        } # type: ignore
    )
    helpers = TradingHelpers()
    hl_client = ExchangeClient(exchange, wallet_address)
    nado_client = NadoExchangeClient(private_key, None, wallet_address, pairs)
    hl_strategy = StrategyManager(hl_client, StrategyEnum.CROSS_EMA_LINEAR_REGRESSION)
    nado_strategy = StrategyManager(nado_client, StrategyEnum.CROSS_EMA_LINEAR_REGRESSION)
    nado_bot = TradingBot(nado_client, nado_strategy, helpers, pairs, timeframe, 'NADO')
    hl_bot = TradingBot(hl_client, hl_strategy, helpers, pairs, timeframe, 'HYPERLIQUID')

    logging.info("🚀 Lançando Bots em paralelo (NADO + HYPERLIQUID)...")

    # 3. Rodar ambos simultaneamente
    async def safe_run(bot_instance):
        try:
            await bot_instance.start()
        except Exception as e:
            logging.critical(f"🚨 Erro fatal no bot {bot_instance.exchange_name}: {e}", exc_info=True)

    # 3. Rodar ambos simultaneamente com proteção
    await asyncio.gather(
        safe_run(nado_bot),
        safe_run(hl_bot)
    )

async def alpaca_test():
    alpaca = ccxt.alpaca({
        "apiKey": "PK34OT2XTSYU3YE6LAPEW5MD7I",
        "secret": "EtihUMrXTfPXRb5XTAo5kCR2YrZfmU52k2EJVo7opgxC"
    }) # type: ignore

    # If we want to use paper api keys, enable sandbox mode
    alpaca.set_sandbox_mode(True)

    markets = await alpaca.fetch_markets()

    logging.info(f"Markets: {markets}")

async def nado_test():
    wallet_address = os.getenv("WALLET_ADDRESS")
    private_key = os.getenv("PRIVATE_KEY")

    if not wallet_address or not private_key:
        raise ValueError(
            "Variáveis de ambiente WALLET_ADDRESS e PRIVATE_KEY devem estar definidas"
        )

    #timeframe = TimeframeEnum.M15
    pairs = load_pair_configs()

    exchange = NadoExchangeClient(private_key, None, wallet_address, pairs)

    
    symbol = "BTC/USDC:USDC" # Ou o par que estás a usar
    entry_size = 0.002       # Tamanho pequeno para teste
    
    try:
        # ---- PASSO 1: LIMPEZA INICIAL ----
        logging.info("🧹 Passo 1: Limpando ordens antigas...")
        await exchange.cancel_all_orders(symbol)
        
        # ---- PASSO 2: ENTRADA (MARKET/FOK) ----
        # Obtemos o preço atual para calcular as proteções
        price_ref = await exchange.get_entry_price(symbol)

        if price_ref is None:
            return

        sl_price = price_ref * 0.98  # Stop Loss 2% abaixo
        tp_price = price_ref * 1.05  # Take Profit 5% acima
        
        logging.info(f"🚀 Passo 2: Abrindo LONG em {price_ref}...")
        entry_order = await exchange.place_entry_order(
            symbol=symbol,
            leverage=1.0,
            entry_amount=entry_size,
            price_ref=price_ref,
            side=Signal.BUY,
            sl_price=sl_price,
            tp_price=tp_price
        )
        logging.info(f"✅ Entrada concluída. Digest: {entry_order.id}")

        # ---- PASSO 3: VERIFICAÇÃO ----
        logging.info("⏳ Aguardando 5 segundos para o Indexer atualizar...")
        await asyncio.sleep(5)
        
        pos = await exchange.get_open_position(symbol)
        if pos:
            logging.info(f"📊 Posição detectada: {pos.size} {symbol} @ {pos.entry_price}")
        else:
            logging.warning("⚠️ Posição não detectada pelo Indexer (normal se houver lag).")

        # ---- PASSO 4: FECHO TOTAL ----
        logging.info("⚖️ Passo 4: Iniciando fecho da posição e cancelamento de triggers...")
        # Aqui simulamos o bot a decidir sair do trade
        await exchange.close_position(
            symbol=symbol, 
            amount=entry_size, 
            side=Signal.BUY
        )
        
        logging.info("🏁 TESTE CONCLUÍDO COM SUCESSO!")

    except Exception as e:
        logging.error(f"❌ O teste falhou no meio do ciclo: {e}", exc_info=True)

async def test_sol_order():
    wallet_address = os.getenv("WALLET_ADDRESS")
    private_key = os.getenv("PRIVATE_KEY")

    if not wallet_address or not private_key:
        raise ValueError(
            "Variáveis de ambiente WALLET_ADDRESS e PRIVATE_KEY devem estar definidas"
        )

    #timeframe = TimeframeEnum.M15
    pairs = load_pair_configs()

    exchange = NadoExchangeClient(private_key, None, wallet_address, pairs)
    symbol = "SOL/USDC:USDC"
    
    print(f"🚀 Iniciando teste real para {symbol}...")

    try:
        # 2. Vamos buscar o preço atual para o teste
        price_ref = await exchange.get_entry_price(symbol)

        if price_ref is None:
            return

        print(f"📊 Preço atual de mercado: {price_ref}")

        # 3. Definimos valores de teste
        # Vamos usar um amount fixo pequeno (ex: 10 USDC) em vez de capital_pct
        test_amount_usdc = 10.0 
        leverage = 1.0
        test_amount_sol = 1.5
        
        # Simulando um sinal de SELL como o do erro anterior
        side = Signal.SELL
        sl_price = price_ref * 1.05 # 5% acima
        tp_price = price_ref * 0.95 # 5% abaixo

        print(f"🛒 Enviando ordem de {side.value} para {symbol}...")
        
        order = await exchange.place_entry_order(
                symbol=symbol,
                leverage=1.0,
                entry_amount=test_amount_sol, # Ja entra em Tokens
                price_ref=price_ref,
                side=Signal.SELL,
                sl_price=price_ref * 1.05,
                tp_price=price_ref * 0.95
            )
        print(f"✅ SUCESSO! Digest da ordem: {order.id}")

    except Exception as e:
        print(f"❌ O teste falhou! Erro: {e}")


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


if __name__ == "__main__":
    if len(sys.argv) < 2:
         asyncio.run(run_bot())
    else:
        comando = sys.argv[1].lower()
        if comando == "train":
            asyncio.run(run_train())
        elif comando == "alpaca":
            asyncio.run(alpaca_test())
        elif comando == "nado":
            asyncio.run(test_sol_order())
        else:
            print(f"❌ Comando desconhecido: {comando}")
            print("Usa: python main.py [treino | backtest]")

