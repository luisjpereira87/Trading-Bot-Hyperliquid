# trading_bot/main.py
try:
    import pkg_resources
except ImportError:
    import sys

    import pip._vendor.pkg_resources as pkg_resources

    sys.modules["pkg_resources"] = pkg_resources
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
            "privateKey": private_key,
            "testnet": True,
            "enableRateLimit": True,
            "options": {"defaultSlippage": 0.01},
        }  # type: ignore
    )
    exchange.fetch_spot_markets = safe_fetch_spot_markets.__get__(exchange, exchange.__class__)
    # IMPORTANTE: Carregar mercados antes de passar para o ExchangeClient
    logging.info("📦 A carregar mercados da Hyperliquid...")
    await exchange.load_markets()

    helpers = TradingHelpers()
    hl_client = ExchangeClient(exchange, wallet_address)
    nado_client = NadoExchangeClient(private_key, None, wallet_address, pairs)
    hl_strategy = StrategyManager(hl_client, StrategyEnum.ML_LIGHTGBM)
    nado_strategy = StrategyManager(nado_client, StrategyEnum.ML_LIGHTGBM)
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
    })  # type: ignore

    # If we want to use paper api keys, enable sandbox mode
    alpaca.set_sandbox_mode(True)

    markets = await alpaca.fetch_markets()

    logging.info(f"Markets: {markets}")


# 1. Definimos a nossa versão segura do método
async def safe_fetch_spot_markets(self, params={}):
    """Versão corrigida para evitar o erro NoneType + str na Testnet"""
    try:
        request = {'type': 'spotMetaAndAssetCtxs'}
        response = await self.publicPostInfo(self.extend(request, params))

        # Se a resposta for inválida, retornamos lista vazia
        if not response or len(response) < 1:
            return []

        universe = self.safe_list(response[0], 'universe', [])
        tokens = self.safe_list(response[0], 'tokens', [])
        markets = []

        for i in range(len(universe)):
            market_data = universe[i]
            name = self.safe_string(market_data, 'name')

            # O FIX: Se o nome for None ou não tiver '/', ignoramos
            if not name or '/' not in name:
                continue

            # Deixamos o CCXT processar o resto se o nome for válido
            # Mas para este teste, podemos simplesmente ignorar Spot
            pass

        return []  # Retornamos vazio para o Spot não atrapalhar os Swaps
    except Exception:
        return []  # Se falhar qualquer coisa, não crasha o bot


if __name__ == "__main__":
    if len(sys.argv) < 2:
        asyncio.run(run_bot())
    else:
        comando = sys.argv[1].lower()
        if comando == "alpaca":
            asyncio.run(alpaca_test())
        else:
            print(f"❌ Comando desconhecido: {comando}")
            print("Usa: python main.py [treino | backtest]")
