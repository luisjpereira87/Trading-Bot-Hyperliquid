import os
import sys

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)
import asyncio
import logging
from datetime import datetime
from typing import List

from ccxt.async_support import hyperliquid

from commons.utils.config_loader import PairConfig, load_pair_configs
from trading_bot.bot import TradingBot
from trading_bot.trading_helpers import TradingHelpers


# Mock simples de Exchange
class MockExchange:
    def __init__(self, candles):
        self.candles = candles
        self.symbol_data = {}
        self.position = None


class ExchangeClientMock:
    def __init__(self, candles):
        self.candles = candles
        self.symbol_data = {}
        #self.current_index: int = 0
        #self.positions = []
        self.position = None
        self.balance = 1000
        self.positions = {} 
        self.current_index = {symbol: 0 for symbol in candles.keys()}
        self.total_pnl = 0.0  # USDC
        self.num_wins = 0
        self.num_losses = 0

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

    async def print_balance(self):
        pass

    async def print_open_orders(self, symbol):
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
    
    
    async def open_new_position(self, symbol, leverage, signal, capital_amount, pair, sl, tp):
        idx = self.current_index[symbol]
        price = self.candles[symbol][idx][4]
        size = capital_amount / price
        self.positions[symbol] = { "pair": pair, "side": signal.value, "size": size, "entryPrice": price, "sl": sl,
        "tp": tp}
        #self.balance -= capital_amount

        await self.calculate_entry_amount(price,capital_amount)
        logging.info(f"OPEN {signal.value} {symbol} size={size:.4f} price={price:.2f}")

    async def close_position(self, pair, size, side):
        pos = self.positions.get(pair)
        if not pos:
            logging.warning(f"Tentou fechar posição que não existe: {pair}")
            return 0

        # Preço atual de fechamento da posição (preço do candle atual)
        idx = self.current_index[pair]
        close_price = self.candles[pair][idx][4]  # close price do candle atual

        entry_price = pos['entryPrice']
        position_side = pos['side']  # 'buy' ou 'sell'

        # Calcular PnL dependendo da direção da posição
        if position_side == 'buy':
            pnl = (close_price - entry_price) * pos['size']
        elif position_side == 'sell':
            pnl = (entry_price - close_price) * pos['size']
        else:
            pnl = 0

        # Atualizar saldo e acumular PnL total
        #self.balance += (pnl + entry_price * pos['size'])  # devolve o capital + lucro/prejuízo
        self.balance += pnl  # devolve o capital + lucro/prejuízo
        self.total_pnl += pnl

        # Contabilizar vitória ou perda
        if pnl > 0:
            self.num_wins += 1
        elif pnl < 0:
            self.num_losses += 1

        # Remover posição fechada
        del self.positions[pair]

        logging.info(f"CLOSE {side} {pair} size={size:.4f} entry={entry_price:.2f} close={close_price:.2f} PnL={pnl:.2f}")
        return pnl
    
    async def get_total_balance(self):
        return float(1000)
    
    async def get_open_position(self, symbol=None):
        # Simula uma posição aberta se existir para o símbolo
        pos = self.positions.get(symbol)
        print(pos)
        if pos:
            return {
                'side': pos['side'],
                'size': pos['size'],
                'entryPrice': pos['entryPrice'],
                'notional': pos['size'] * pos['entryPrice']
            }
        return None
    
    async def simulate_tp_sl(self, candle, symbol):
        print(F"AQUIUIIIII {self.positions.get(symbol)}")

        position = self.positions.get(symbol)
        
        if position:
            side = position["side"]
            sl = position["sl"]
            tp = position["tp"]
            entry = position["entryPrice"]
            pair = position["pair"].symbol
            size = position["size"]

            print(F"SIZEEEEEEEE {size}")

            low = candle[2]
            high = candle[3]

            if side == "buy":
                if sl is not None and low <= sl:
                    print(F"close buy SL {high} {sl}")
                    await self.close_position(pair, size, side)
                elif tp is not None and high >= tp:
                    print(F"close buy TP {low} {tp}")
                    await self.close_position(pair, size, side)

            elif side == "sell":
                if sl is not None and high >= sl:
                    print(F"close sell SL {high} {sl}")
                    await self.close_position(pair, size, side)
                elif tp is not None and low <= tp:
                    print(F"close sell TP {low} {tp}")
                    await self.close_position(pair, size, side)
    
    def print_summary(self):
        total_trades = self.num_wins + self.num_losses
        win_rate = (self.num_wins / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = (self.total_pnl / total_trades) if total_trades > 0 else 0

        print("\n📊 Backtest Summary")
        print("---------------------------")
        print(f"💰 Final Balance:       ${self.balance:.2f}")
        print(f"📈 Total PnL:           ${self.total_pnl:.2f}")
        print(f"🧾 Number of Trades:    {total_trades}")
        print(f"✅ Wins:                {self.num_wins}")
        print(f"❌ Losses:              {self.num_losses}")
        print(f"🎯 Win Rate:            {win_rate:.2f}%")
        print(f"📊 Avg PnL per Trade:   ${avg_pnl:.2f}")
        print("---------------------------")



# Runner principal do backtest
class BacktestRunner:
    def __init__(self, strategy_name="ml", timeframe="15m"):
        self.strategy_name = strategy_name
        self.timeframe = timeframe

    async def run(self, pair: PairConfig, candles: List[list]):
        logging.info(f"🔁 Starting backtest for {pair.symbol}")
        exchange = MockExchange({pair.symbol: candles})
        exchange_client = ExchangeClientMock({pair.symbol: candles})
        helpers = TradingHelpers()
        bot = TradingBot(exchange,exchange_client,None,helpers,load_pair_configs(),self.timeframe,14)



        for i in range(100, len(candles)):
            sliced = candles[:i]
            current_candle = sliced[-1]
            exchange_client.update_candles(pair.symbol, sliced)
            await exchange_client.simulate_tp_sl(current_candle, pair.symbol)

            pnl = await bot.run_pair(pair)
            print(pnl)
        
        exchange_client.print_summary()

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


