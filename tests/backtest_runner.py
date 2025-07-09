import asyncio
import logging
from datetime import datetime
from typing import List

from ccxt.async_support import hyperliquid

from enums.signal_enum import Signal
from trading_bot.bot import TradingBot
from trading_bot.exit_logic import ExitLogic
from trading_bot.trading_helpers import TradingHelpers
from utils.config_loader import PairConfig, load_pair_configs


# Mock simples de Exchange
class MockExchange:
    def __init__(self, candles):
        self.candles = candles
        self.symbol_data = {}
        #self.current_index: int = 0
        #self.positions = []
        self.position = None

        self.positions = {} 
        self.current_index = {symbol: 0 for symbol in candles.keys()}

    async def fetch_ohlcv(self, symbol, timeframe='1m', limit=100, params=None, **kwargs):
        idx = self.current_index[symbol]
        return self.candles[symbol][:idx+1]

    
    async def fetch_order_book(self, symbol, limit=5, params=None):
        order_book_mock = {
            'asks': [[3100.5, 10], [3101.0, 15], [3101.5, 5]],
            'bids': [[3100.0, 12], [3099.5, 20], [3099.0, 7]]
        }
        return order_book_mock


class ExchangeClientMock:
    def __init__(self, candles):
        self.candles = candles
        self.symbol_data = {}
        #self.current_index: int = 0
        #self.positions = []
        self.position = None

        self.positions = {} 
        self.current_index = {symbol: 0 for symbol in candles.keys()}

    def update_candles(self, symbol, sliced):
        self.symbol_data[symbol] = sliced
        self.current_index[symbol] = len(sliced) - 1

    def advance_candle(self, symbol):
        if self.current_index[symbol] + 1 < len(self.candles[symbol]):
            self.current_index[symbol] += 1

    async def fetch_ohlcv(self, symbol, timeframe='1m', limit=100, params=None, **kwargs):
        idx = self.current_index[symbol]
        return self.candles[symbol][:idx+1]

    async def fetch_positions(self, params=None, **kwargs):
        return []

    async def fetch_balance(self, params=None, **kwargs):
        return {"total": {"USDT": 1000}}

    async def fetch_open_orders(self, symbol, params=None, **kwargs):
        return []
    
    async def fetch_order_book(self, symbol, limit=5, params=None):
        order_book_mock = {
            'asks': [[3100.5, 10], [3101.0, 15], [3101.5, 5]],
            'bids': [[3100.0, 12], [3099.5, 20], [3099.0, 7]]
        }
        return order_book_mock

    async def cancel_all_orders(self, symbol, params=None, **kwargs):
        pass

    async def create_market_order(self, symbol, side, amount, params=None, **kwargs):
        return {"status": "filled", "side": side, "amount": amount}

    async def create_limit_order(self, symbol, side, amount, price, params=None, **kwargs):
        return {"status": "open", "side": side, "amount": amount, "price": price}
    
    async def get_reference_price(self, symbol):
        try:
            idx = self.current_index[symbol]
            price = self.candles[symbol][idx][4]
            return price
        except Exception as e:
            logging.error(f"Erro no mock get_reference_price: {e}")
            return None
        
    async def calculate_entry_amount(self, price_ref, capital_amount):
        # Exemplo simples: calcula quantidade a comprar/vender dividindo capital pelo preço
        if price_ref is None or price_ref <= 0:
            raise ValueError("Preço de referência inválido")
        return capital_amount / price_ref
    
    
    async def open_new_position(self, signal, capital_amount, pair, sl, tp):
        print("ENTROU AQUI")
        symbol = pair.symbol
        idx = self.current_index[symbol]
        price = self.candles[symbol][idx][4]
        size = capital_amount / price
        self.positions[symbol] = {"side": signal.value, "size": size, "entry_price": price}
        self.balance -= capital_amount

        await self.calculate_entry_amount(price,capital_amount)
        logging.info(f"OPEN {signal.value} {symbol} size={size:.4f} price={price:.2f}")

    async def close_position(self, pair, size, side, exit_price):
        symbol = pair.symbol
        pos = self.positions.get(symbol)
        if not pos:
            logging.warning(f"Tentou fechar posição que não existe: {symbol}")
            return 0
        pnl = 0
        if pos["side"] == "buy":
            pnl = (exit_price - pos["entry_price"]) * size
        else:
            pnl = (pos["entry_price"] - exit_price) * size
        self.balance += size * exit_price
        del self.positions[symbol]
        logging.info(f"CLOSE {side.value} {symbol} size={size:.4f} price={exit_price:.2f} PnL={pnl:.2f}")
        return pnl
    
    async def get_total_balance(self):
        try:
            return float(1000)
        except Exception as e:
            logging.error(f"Erro ao obter saldo total: {e}")
            return 0


# Runner principal do backtest
class BacktestRunner:
    def __init__(self, strategy_name="ml", timeframe="1m"):
        self.strategy_name = strategy_name
        self.timeframe = timeframe

    async def run(self, pair: PairConfig, candles: List[list]):
        logging.info(f"🔁 Starting backtest for {pair.symbol}")
        exchange = MockExchange({pair.symbol: candles})
        exchange_client = ExchangeClientMock({pair.symbol: candles})
        #order_manager = MockOrderManager()
        helpers = TradingHelpers()
        #order_manager = MockOrderManager()
        #exit_logic = ExitLogic(helpers, order_manager)  
        #bot = TradingBot(exchange, order_manager, strategy_name=self.strategy_name, timeframe=self.timeframe)
        bot = TradingBot(exchange,exchange_client,None,helpers,load_pair_configs(),self.timeframe,14)



        for i in range(100, len(candles)):
            sliced = candles[:i]
            exchange_client.update_candles(pair.symbol, sliced)

            # Simula último preço atual para controle de PnL
            #if order_manager.current_position:
            #    order_manager.current_position["last_price"] = sliced[-1][4]  # close

            pnl = await bot.run_pair(pair)
            print(pnl)

        #order_manager.print_summary(pair.symbol)


# Função para obter candles históricos (ajuste para o seu projeto)
async def get_historical_ohlcv(symbol: str, timeframe: str = '15m', limit: int = 500):
    # Configura sua exchange Hyperliquid
    exchange =  hyperliquid({
            "enableRateLimit": True,
            "testnet": True,
        })

    try:
        # Busca candles OHLCV históricos (timestamp, open, high, low, close, volume)
        candles = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return candles
    finally:
        await exchange.close()


# Execução principal
async def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    pair = PairConfig(symbol="ETH/USDC:USDC", leverage=2, capital=0.5, min_profit_abs= 5.0)
    candles = await get_historical_ohlcv(pair.symbol, "15m", 500)

    runner = BacktestRunner(strategy_name="ml", timeframe="15m")
    await runner.run(pair, candles)

if __name__ == "__main__":
    asyncio.run(main())


