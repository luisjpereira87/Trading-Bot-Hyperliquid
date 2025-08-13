import os
import sys
from typing import List

import pandas as pd

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

import asyncio
import logging

import nest_asyncio
from ccxt.async_support import hyperliquid

from commons.enums.signal_enum import Signal
from commons.enums.strategy_enum import StrategyEnum
from commons.enums.timeframe_enum import TimeframeEnum
from commons.helpers.trading_helpers import TradingHelpers
from commons.models.closed_order_dclass import ClosedOrder
from commons.models.ohlcv_format_dclass import OhlcvFormat
from commons.models.open_position_dclass import OpenPosition
from commons.models.opened_order_dclass import OpenedOrder
from commons.utils.config_loader import (PairConfig, get_pair_by_symbol,
                                         load_pair_configs)
from commons.utils.load_params import LoadParams
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from strategies.strategy_manager import StrategyManager
from tests.plot_trades import PlotTrades
from trading_bot.bot import TradingBot
from trading_bot.exchange_client import ExchangeClient

nest_asyncio.apply()

class ExchangeClientMock(ExchangeClient):
    def __init__(self, candles, candles_higher, pair: PairConfig, balance: float = 1000):
        self.candles = candles
        self.candles_higher = candles_higher
        self.position = None
        self.balance = balance
        self.positions = {} 
        self.current_index = {symbol: 0 for symbol in candles.keys()}
        self.total_pnl = 0.0  # USDC
        self.num_wins = 0
        self.num_losses = 0
        self.current_candle = []
        self.pair = pair
        self.trades = []
        self.id = 0
        self.stop_loss_orders = {}
        self.closed_orders:List[ClosedOrder] = []

    def update_candles(self, symbol, current_candle, index):
        self.current_candle = current_candle
        self.current_index[symbol] = index

    def __get_window(self, full_candles, current_index, window_size):
        end = current_index
        start = max(0, current_index - window_size)
        return full_candles[start:end +1]

    async def fetch_ohlcv(self, symbol, timeframe:TimeframeEnum =TimeframeEnum.M15, limit=100, is_higher: bool = False)->OhlcvFormat:
        idx = self.current_index[symbol]

        window = self.__get_window(self.candles[symbol], idx, limit)
        window_higher = self.__get_window(self.candles_higher[symbol], idx, limit)

        self.current_price = OhlcvWrapper(window).get_last_closed_candle().close

        print(f"[DEBUG fetch_ohlcv] idx usado={self.current_index[symbol]} | último close retornado={self.candles[symbol][self.current_index[symbol]][4]}")
        return OhlcvFormat(OhlcvWrapper(window), OhlcvWrapper(window_higher))
    
    async def get_entry_price(self, symbol: str) -> float:
        idx = self.current_index[symbol]
        return self.candles[symbol][idx + 1][4]
        #return self.current_price

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
        try:
            if price_ref is None or price_ref <= 0 or capital_amount <= 0:
                # Retorna 0 em caso de dados inválidos (sem exceção)
                return 0.0

            quantity = capital_amount / price_ref

            # Ordem mínima de $10
            min_order_value = 10
            if quantity * price_ref < min_order_value:
                return 0.0

            return round(quantity, 6)

        except Exception as e:
            logging.error(f"Erro ao calcular quantidade de entrada (mock): {e}")
            return 0.0
    
    async def open_new_position(self, symbol, leverage, signal, capital_amount, pair, sl, tp) -> (OpenedOrder | None) :
        idx = self.current_index[symbol]
        price = self.candles[symbol][idx + 1][4]
        print(f"CURRENT PRICE OPEN {price}")

        entry_amount = await self.calculate_entry_amount(price, capital_amount)

        min_order_value = 10
        if entry_amount * price < min_order_value:
            logging.warning(f"🚫 Order below $10 minimum: {entry_amount * price:.2f}")
            return

        size = entry_amount #usa o entry_amount para o tamanho

        self.positions[symbol] = {
            "pair": pair,
            "id": idx,
            "side": signal.value,
            "size": size,
            "entryPrice": price,
            "sl": sl,
            "tp": tp,
        }

        # Verifica se existe mais entries que exits, posição não fechada mas sim reversão
        entries_count = sum(1 for t in self.trades if t['type'] == 'entry')
        exits_count = sum(1 for t in self.trades if t['type'] == 'exit')
        if  entries_count > exits_count:
            await self.close_position(pair, size, signal)


        self.trades.append({
            "type": "entry",
            "side": signal.value,
            "index": idx + 1,
            "price": price,
            "current_candle": self.current_candle,
            "sl": sl,
            "tp": tp,
        })

        logging.info(f"OPEN {signal.value} {symbol} entry_amount={entry_amount} idx={idx} size={size:.4f} price={price:.2f}, sl={sl} tp={tp}")
        
        return OpenedOrder(str(idx), None, None, None, symbol, "entry", signal.value, price, size, False, None)

    async def close_position(self, pair, size, side: Signal):
        pos = self.positions.get(pair)
        if not pos:
            logging.warning(f"Tentou fechar posição que não existe: {pair}")
            return 0

        idx = self.current_index[pair]
        close_price = self.candles[pair][idx + 1][4]

        entry_price = pos['entryPrice']
        position_side = pos['side']

        fee_rate = 0.00035  # 0.035% por operação (taker fee)

        if position_side == 'buy':
            gross_pnl = (close_price - entry_price) * pos['size']
        elif position_side == 'sell':
            gross_pnl = (entry_price - close_price) * pos['size']
        else:
            gross_pnl = 0

        # Calcular taxas
        entry_fee = entry_price * pos['size'] * fee_rate
        exit_fee = close_price * pos['size'] * fee_rate
        total_fees = entry_fee + exit_fee

        # Calcular PnL líquido
        pnl = gross_pnl - total_fees

        self.trades.append({
            "type": "exit",
            "side": pos['side'],
            "index": idx + 1,
            "price": close_price,
            "pnl": pnl,
        })

        # Atualiza balance somando PnL (capital investido está considerado dentro do balance)
        self.balance += pnl
        self.total_pnl += pnl

        if pnl > 0:
            self.num_wins += 1
        elif pnl < 0:
            self.num_losses += 1

        self.closed_orders.append(ClosedOrder(pos['id'], None, None, None, pair, None, pos['side'], close_price, pos['size'], False, None))

        del self.positions[pair]

        logging.info(f"CLOSE {side} {pair} size={size:.4f} idx={idx} entry={entry_price:.2f} close={close_price:.2f} "
                 f"PnL={pnl:.2f} (gross: {gross_pnl:.2f}, fees: {total_fees:.4f})")
        return pnl
    
    async def fetch_closed_orders(self, symbol=None, limit=50) -> List[ClosedOrder]:
        return self.closed_orders
    
    async def get_total_balance(self):
        return float(self.balance)
    
    async def get_open_position(self, symbol=None):
        # Simula uma posição aberta se existir para o símbolo
        pos = self.positions.get(symbol)
        if pos:
            return OpenPosition(pos['side'], pos['size'], pos['entryPrice'], '', pos['size'] * pos['entryPrice'], pos['sl'], pos['tp'])
        return None

    async def modify_stop_loss_order(self, symbol: str, entry_id: str, new_stop_price: float):
        """
        Modifica o SL associado a uma posição existente.
        """  
        if self.trades:
            last_trade = self.trades[-1]
            if last_trade["type"] == "entry":
                last_trade["sl"] = new_stop_price

            print(f"SL atualizado em {symbol} com entry_id {entry_id}")
        else:        
            print(f"⚠️ Nenhuma SL encontrada para modificar em {symbol} com entry_id {entry_id}")
    
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
                hit_sl = sl is not None and low <= sl
                hit_tp = tp is not None and high >= tp

                if hit_tp and hit_sl:
                    # Decide prioridade (ex: assume SL primeiro)
                    print(f"Candle ambíguo BUY: SL e TP atingidos. Prioridade ao SL.")
                    await self.close_position(pair, size, Signal.BUY)
                elif hit_sl:
                    print(f"close buy SL {low} {sl}")
                    await self.close_position(pair, size, Signal.BUY)
                elif hit_tp:
                    print(f"close buy TP {high} {tp}")
                    await self.close_position(pair, size, Signal.BUY)

                return candle

            if side == "sell":
                hit_sl = sl is not None and high >= sl
                hit_tp = tp is not None and low <= tp

                if hit_tp and hit_sl:
                    print(f"Candle ambíguo SELL: SL e TP atingidos. Prioridade ao SL.")
                    await self.close_position(pair, size, Signal.SELL)
                elif hit_sl:
                    print(f"close sell SL {high} {sl}")
                    await self.close_position(pair, size, Signal.SELL)
                elif hit_tp:
                    print(f"close sell TP {low} {tp}")
                    await self.close_position(pair, size, Signal.SELL)
                return candle
        

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
                'current_candle': entry['current_candle'],
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
#        print(f"Current candle: {trade['current_candle']}")
        print("----------------------------")

        return detailed_trades



# Runner principal do backtest
class BacktestRunner:
    def __init__(self, strategy_name: StrategyEnum, timeframe: TimeframeEnum, pair: PairConfig, limit: int = 5, balance: float = 1000):
        self.strategy_name = strategy_name
        self.timeframe = timeframe
        self.pair = pair
        self.limit = limit
        self.balance = balance
        # Guardar operações para plot
        self.trades = []  # lista de dicts: {"type": "entry"|"exit", "side": "buy"|"sell", "index": int, "price": float}
        self.ohlcv = []


    async def run(self, params=None, is_plot = False, train = False):
        logging.info(f"🔁 Starting backtest for {self.pair.symbol}")

        self.ohlcv = await self.get_historical_ohlcv(self.timeframe, self.limit, train)
        self.ohlcv_higher = await self.get_historical_ohlcv(self.timeframe.get_higher(), self.limit, train)

        exchange_client = ExchangeClientMock({self.pair.symbol: self.ohlcv}, {self.pair.symbol: self.ohlcv_higher}, self.pair, self.balance)
        helpers = TradingHelpers()
        strategy = StrategyManager(exchange_client, self.strategy_name)

        bot = TradingBot(exchange_client, strategy, helpers, load_pair_configs(), self.timeframe)
        signals = []

        if params is not None:
            strategy.set_params(params)

        for i in range(strategy.MIN_REQUIRED_CANDLES, len(self.ohlcv) - 1):
            #candles_slice = self.ohlcv[:i]  # candles até i-1 fechados
            current_candle = self.ohlcv[i]  # vela em que vais abrir posição no início

            exchange_client.update_candles(self.pair.symbol, current_candle, i)

            await exchange_client.simulate_tp_sl(current_candle, self.pair.symbol)

            signal = await bot.run_pair(self.pair)

            signals.append({'signal': signal, 'index': i})

            #if i == 180:
            #   break

        #print(self.ohlcv)
        if is_plot:
            PlotTrades.plot_trades(self.pair.symbol, self.ohlcv, signals, exchange_client.trades)

        print(f"[TRADE_SNAPSHOT] {bot.get_average_features()}")
        exchange_client.generate_detailed_report(exchange_client.trades)
        summary = exchange_client.get_performance_summary()
        return summary
    
    # Função para obter candles históricos (ajuste para o seu projeto)
    async def get_historical_ohlcv(self, timeframe: TimeframeEnum, limit: int = 5, train: bool = False):
        if len(self.ohlcv) > 0:
            return self.ohlcv

        # Configura sua exchange Hyperliquid
        exchange =  hyperliquid({
                "enableRateLimit": True,
                "testnet": True,
            })

        try:
            # Busca candles OHLCV históricos (timestamp, open, high, low, close, volume)

            if train:

                since_timestamp = int(pd.Timestamp("2025-06-01").timestamp() * 1000)  # em ms
                old_data = await exchange.fetch_ohlcv(self.pair.symbol, timeframe, since=since_timestamp, limit=limit)

                since_timestamp1 = int(pd.Timestamp("2025-07-01").timestamp() * 1000)  # em ms
                old_data1 = await exchange.fetch_ohlcv(self.pair.symbol, timeframe, since=since_timestamp1, limit=limit)

                return old_data + old_data1

            
            self.ohlcv = await exchange.fetch_ohlcv(self.pair.symbol, timeframe, limit=limit)
            return self.ohlcv
        finally:
            await exchange.close()


# Execução principal
async def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    pair = get_pair_by_symbol("BTC/USDC:USDC")

    if pair != None:

        runner = BacktestRunner(StrategyEnum.AI_SUPERTREND, TimeframeEnum.M15, pair, 250, 1000)
        
        await runner.run(LoadParams.load_best_params_with_weights(pair.symbol), True)
    #print(LoadParams.load_best_params_with_weights())

if __name__ == "__main__":
    asyncio.run(main())


