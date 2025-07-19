import os
import random
import sys
from enum import Enum
from itertools import product

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

import asyncio
import logging
from datetime import datetime
from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import nest_asyncio
import optuna
from ccxt.async_support import hyperliquid

from commons.enums.mode_enum import ModeEnum
from commons.enums.signal_enum import Signal
from commons.enums.strategy_enum import StrategyEnum
from commons.enums.timeframe_enum import TimeframeEnum
from commons.models.ohlcv_format import OhlcvFormat
from commons.models.open_position import OpenPosition
from commons.models.strategy_params import StrategyParams
from commons.utils.config_loader import PairConfig, load_pair_configs
from commons.utils.load_params import LoadParams
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from strategies.strategy_manager import StrategyManager
from tests.grid_search_param import GridSearchParams
from tests.plot_trades import PlotTrades
from trading_bot.bot import TradingBot
from trading_bot.exchange_client import ExchangeClient
from trading_bot.trading_helpers import TradingHelpers

nest_asyncio.apply()


# Mock simples de Exchange
class MockExchange:
    def __init__(self, candles):
        self.candles = candles
        #self.symbol_data = {}
        self.position = None


class ExchangeClientMock(ExchangeClient):
    def __init__(self, candles, candles_higher, pair: PairConfig):
        self.candles = candles
        self.candles_higher = candles_higher
        self.position = None
        self.balance = 1000
        self.positions = {} 
        self.current_index = {symbol: 0 for symbol in candles.keys()}
        self.total_pnl = 0.0  # USDC
        self.num_wins = 0
        self.num_losses = 0
        self.current_candle = []
        self.pair = pair
        self.trades = []

    def update_candles(self, symbol, current_candle, index):
        self.current_candle = current_candle
        #self.symbol_data[symbol] = sliced
        self.current_index[symbol] = index

    def __get_window(self, full_candles, current_index, window_size):
        end = current_index
        start = max(0, current_index - window_size)
        return full_candles[start:end]

    async def fetch_ohlcv(self, symbol, timeframe:TimeframeEnum =TimeframeEnum.M15, limit=100, is_higher: bool = False)->OhlcvFormat:
        idx = self.current_index[symbol]

        window = self.__get_window(self.candles[symbol], idx, limit)
        window_higher = self.__get_window(self.candles_higher[symbol], idx, limit)

        self.current_price = OhlcvWrapper(window).get_last_closed_candle().close
        print(f"CURRENT_PRICE {self.current_price}")
        print(f"[DEBUG fetch_ohlcv] idx usado={self.current_index[symbol]} | último close retornado={self.candles[symbol][self.current_index[symbol]][4]}")
        return OhlcvFormat(OhlcvWrapper(window), OhlcvWrapper(window_higher))
    
    async def get_entry_price(self, symbol: str) -> float:
        return self.current_price

    async def fetch_positions(self, params=None, **kwargs):
        return []

    async def fetch_balance(self, params=None, **kwargs):
        return {"total": {"USDT": self.balance}}
    
    async def fetch_order_book(self, symbol, limit=5, params=None):
        order_book_mock = {
            'asks': [[3100.5, 10], [3101.0, 15], [3101.5, 5]],
            'bids': [[3100.0, 12], [3099.5, 20], [3099.0, 7]]
        }
        return order_book_mock
    
    async def get_available_balance(self):
        return self.balance
    
    async def fetch_ticker(self, symbol):
        return {"close": self.current_candle[4]}

    async def cancel_all_orders(self, symbol, params=None, **kwargs):
        pass

    async def print_balance(self):
        pass

    async def print_open_orders(self, symbol):
        pass
        
    async def calculate_entry_amount(self, price_ref, capital_amount):
        # Exemplo simples: calcula quantidade a comprar/vender dividindo capital pelo preço
        if price_ref is None or price_ref <= 0:
            raise ValueError("Preço de referência inválido")
        
        return capital_amount / price_ref
    
    async def open_new_position(self, symbol, leverage, signal, capital_amount, pair, sl, tp):
        idx = self.current_index[symbol]
        price = self.current_price
        size = capital_amount / price

        self.positions[symbol] = { "pair": pair, "side": signal.value, "size": size, "entryPrice": price, "sl": sl,
        "tp": tp}

        self.trades.append({
            "type": "entry",
            "side": signal.value,
            "index": idx,
            "price": price,
            "sl": sl,
            "tp": tp,
            #"timestamp": self.current_time_or_candle_index,
        })

        await self.calculate_entry_amount(price, capital_amount)
        logging.info(f"OPEN {signal.value} {symbol} idx={idx} size={size:.4f} price={price:.2f}, sl={sl} tp={tp}")

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
            pnl *= self.pair.leverage
        elif position_side == 'sell':
            pnl = (entry_price - close_price) * pos['size']
            pnl *= self.pair.leverage
        else:
            pnl = 0

        self.trades.append({
            "type": "exit",
            "side": pos['side'],
            "index": idx,
            "price": pos['entryPrice'],
            "pnl": pnl,
            #"timestamp": self.current_time_or_candle_index,
        })

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
        return float(self.balance)
    
    async def get_open_position(self, symbol=None):
        # Simula uma posição aberta se existir para o símbolo
        pos = self.positions.get(symbol)
        if pos:
            return OpenPosition(pos['side'], pos['size'], pos['entryPrice'], pos['size'] * pos['entryPrice'], pos['sl'], pos['tp'])
        return None
    
    async def simulate_tp_sl(self, candle, symbol):

        position = self.positions.get(symbol)
        
        if position:
            side = position["side"]
            sl = position["sl"]
            tp = position["tp"]
            entry = position["entryPrice"]
            pair = position["pair"].symbol
            size = position["size"]

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

    def get_performance_summary(self):
        total_trades = self.num_wins + self.num_losses
        win_rate = (self.num_wins / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = (self.total_pnl / total_trades) if total_trades > 0 else 0
        return {
            'balance': self.balance,
            'total_pnl': self.total_pnl,
            'total_trades': self.num_wins + self.num_losses,
            'wins': self.num_wins,
            'losses': self.num_losses,
            'win_rate': win_rate,
            'avg_pnl':avg_pnl
            # adiciona o que quiseres aqui
        }
    
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

    
    def generate_detailed_report(self, trades):
        entries = []
        exits = []
        detailed_trades = []

        # Separar entradas e saídas
        for t in trades:
            if t['type'] == 'entry':
                entries.append(t)
            elif t['type'] == 'exit':
                exits.append(t)

        # Assumindo que a ordem das entradas e saídas corresponde (1a entrada com 1a saída etc)
        for i in range(min(len(entries), len(exits))):
            entry = entries[i]
            exit = exits[i]
            trade_pnl = exit.get('pnl', 0)
            trade = {
                'side': entry['side'],
                'entry_index': entry['index'],
                'entry_price': entry['price'],
                'exit_index': exit['index'],
                'exit_price': exit['price'],
                'pnl': trade_pnl,
                'result': 'win' if trade_pnl > 0 else ('loss' if trade_pnl < 0 else 'breakeven')
            }
            detailed_trades.append(trade)

        # Estatísticas gerais
        total_trades = len(detailed_trades)
        wins = sum(1 for t in detailed_trades if t['result'] == 'win')
        losses = sum(1 for t in detailed_trades if t['result'] == 'loss')
        breakeven = sum(1 for t in detailed_trades if t['result'] == 'breakeven')
        total_pnl = sum(t['pnl'] for t in detailed_trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0

        # Imprimir relatório
        print("\n📋 Detailed Trade Report")
        print("----------------------------")
        for i, trade in enumerate(detailed_trades, 1):
            print(f"Trade {i}: {trade['side'].upper()} | Entry idx: {trade['entry_index']} at {trade['entry_price']:.2f} | Exit idx: {trade['exit_index']} at {trade['exit_price']:.2f} | PnL: {trade['pnl']:.2f} | Result: {trade['result']}")

        print("----------------------------")
        print(f"Total trades: {total_trades}")
        print(f"Wins: {wins} | Losses: {losses} | Breakeven: {breakeven}")
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Total PnL: {total_pnl:.2f}")
        print(f"Average PnL per trade: {avg_pnl:.2f}")
        print("----------------------------")

        return detailed_trades



# Runner principal do backtest
class BacktestRunner:
    def __init__(self, strategy_name: StrategyEnum, timeframe: TimeframeEnum, pair: PairConfig, limit: int = 5):
        self.strategy_name = strategy_name
        self.timeframe = timeframe
        self.pair = pair
        self.limit = limit
        # Guardar operações para plot
        self.trades = []  # lista de dicts: {"type": "entry"|"exit", "side": "buy"|"sell", "index": int, "price": float}
        self.ohlcv = []
        # Define o espaço de busca dos parâmetros

    async def run(self, params=None, is_plot = False):
        logging.info(f"🔁 Starting backtest for {self.pair.symbol}")

        self.ohlcv = await self.get_historical_ohlcv(self.timeframe, self.limit)
        self.ohlcv_higher = await self.get_historical_ohlcv(self.timeframe.get_higher(), self.limit)

        #exchange = MockExchange({self.pair.symbol: self.ohlcv})
        exchange_client = ExchangeClientMock({self.pair.symbol: self.ohlcv}, {self.pair.symbol: self.ohlcv_higher}, self.pair)
        helpers = TradingHelpers()
        strategy = StrategyManager(exchange_client, self.strategy_name)

        bot = TradingBot(exchange_client, strategy, helpers, load_pair_configs(), self.timeframe)
        signals = []
        #window_size = 21

        if params is not None:
            strategy.set_params(params)

        for i in range(strategy.MIN_REQUIRED_CANDLES, len(self.ohlcv)):
            #candles_slice = self.ohlcv[:i]  # candles até i-1 fechados
            current_candle = self.ohlcv[i]  # vela em que vais abrir posição no início

            exchange_client.update_candles(self.pair.symbol, current_candle, i-1)

           
            print(f"[VALIDAÇÃO] i={i} | current_index={exchange_client.current_index[self.pair.symbol]} | candle[i][4]={self.ohlcv[i][4]}")
            signal = await bot.run_pair(self.pair)

            signals.append({'signal': signal, 'index': i-1})
    
        if is_plot:
            #indexes = [s['index'] for s in signals]
            #indexes2 = [s['index'] for s in exchange_client.trades]
            #print(len(exchange_client.trades))
            #print(indexes2)
            #exchange_client.print_summary()
            PlotTrades.plot_trades(self.pair.symbol, self.ohlcv, signals, exchange_client.trades)

        exchange_client.generate_detailed_report(exchange_client.trades)
        summary = exchange_client.get_performance_summary()
        return summary
    
    # Função para obter candles históricos (ajuste para o seu projeto)
    async def get_historical_ohlcv(self, timeframe: TimeframeEnum, limit: int = 5):
        if len(self.ohlcv) > 0:
            return self.ohlcv

        # Configura sua exchange Hyperliquid
        exchange =  hyperliquid({
                "enableRateLimit": True,
                "testnet": True,
            })

        try:
            # Busca candles OHLCV históricos (timestamp, open, high, low, close, volume)
            self.ohlcv = await exchange.fetch_ohlcv(self.pair.symbol, timeframe, limit=limit)
            return self.ohlcv
        finally:
            await exchange.close()


# Execução principal
async def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    pair = PairConfig(symbol="BTC/USDC:USDC", leverage=10, capital=1, min_profit_abs= 5.0)
    runner = BacktestRunner(StrategyEnum.AI_SUPERTREND, TimeframeEnum.M15, pair, 250)

    
    await runner.run(LoadParams.load_best_params_with_weights(), True)
    #print(LoadParams.load_best_params_with_weights())

if __name__ == "__main__":
    asyncio.run(main())


