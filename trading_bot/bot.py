import os
import pytz
import ccxt.async_support as ccxt
from .exchange_client import ExchangeClient
from .order_manager import OrderManager
from .strategy import Strategy
import asyncio
from datetime import datetime, timezone, timedelta
import logging
import math
from .strategies.indicators import Indicators  # Para cálculo ATR

class TradingBot:
    def __init__(self):
        self.wallet_address = os.getenv("WALLET_ADDRESS")
        self.private_key = os.getenv("PRIVATE_KEY")
        self.timeframe = '15m'
        self.tp_pct = 0.03  # Percentual mínimo para TP
        self.sl_pct = 0.015  # Percentual mínimo para SL
        self.tp_factor = 2  # Multiplicador TP x SL
        self.atr_period = 14  # Período ATR para SL/TP dinâmicos

        # Configurações múltiplos pares, leverage e % capital (como float já)
        self.pairs = [
            {"symbol": "BTC/USDC:USDC", "leverage": 5, "capital": 0.10},
            {"symbol": "ETH/USDC:USDC", "leverage": 10, "capital": 0.20},
            # Pode adicionar mais pares aqui
        ]

        self.last_candle_times = {pair['symbol']: None for pair in self.pairs}  # controle vela por par

    def calculate_sl_tp(self, entry_price, side, atr_now):
        """
        Calcula SL e TP dinâmicos usando ATR e percentual mínimo.
        """
        sl_min_dist = self.sl_pct * entry_price
        if side == 'buy':
            sl_price = entry_price - max(sl_min_dist, atr_now)
            tp_price = entry_price + self.tp_factor * (entry_price - sl_price)
        else:  # sell
            sl_price = entry_price + max(sl_min_dist, atr_now)
            tp_price = entry_price - self.tp_factor * (sl_price - entry_price)

        return round(sl_price, 2), round(tp_price, 2)

    async def run_pair(self, pair):
        exchange = ccxt.hyperliquid({
            "walletAddress": self.wallet_address,
            "privateKey": self.private_key,
            "testnet": True,
            'enableRateLimit': True,
            'options': {
                'defaultSlippage': 0.01
            }
        })

        symbol = pair['symbol']
        leverage = int(pair['leverage'])
        capital_pct = float(pair['capital'])

        try:
            exchange_client = ExchangeClient(exchange, self.wallet_address, symbol, leverage)
            order_manager = OrderManager(exchange)
            strategy = Strategy(exchange, symbol, self.timeframe)

            balance_total = await exchange_client.get_total_balance()
            capital_amount = balance_total * capital_pct

            signal = await strategy.get_signal()
            if signal not in ['buy', 'sell']:
                logging.info(f"\n⛔ Nenhum sinal válido para {symbol}. Ignorando.")
                await exchange.close()
                return

            await exchange_client.print_balance()
            await exchange_client.print_open_orders(symbol)
            await exchange_client.cancel_all_orders(symbol)

            current_position = await exchange_client.get_open_position()

            if current_position:
                current_side_ccxt = current_position['side']  # 'long' ou 'short'
                current_size = float(current_position['size'])
                current_side = 'buy' if current_side_ccxt == 'long' else 'sell'

                if (signal == 'buy' and current_side == 'sell') or (signal == 'sell' and current_side == 'buy'):
                    logging.info(f"Fechando posição oposta {symbol}: {current_side_ccxt} de tamanho {current_size}")
                    await order_manager.close_position(symbol, current_size, current_side)
                    current_position = None

            if not current_position:
                price_ref = await exchange_client.get_reference_price()
                entry_amount = await exchange_client.calculate_entry_amount(price_ref, capital_amount)

                side = signal
                logging.info(f"{symbol}: Enviando ordem de entrada {side} com quantidade {entry_amount} a preço {price_ref}")
                await exchange_client.place_entry_order(entry_amount, price_ref, side)
                entry_price = await exchange_client.get_entry_price()

            else:
                current_side_ccxt = current_position['side']
                current_size = float(current_position['size'])
                side = 'buy' if current_side_ccxt == 'long' else 'sell'
                entry_price = current_position.get('entryPrice')
                if entry_price is None:
                    entry_price = await exchange_client.get_entry_price()
                entry_amount = current_size

            # Cálculo ATR para SL e TP
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=self.atr_period + 1)
            highs = [c[2] for c in ohlcv]
            lows = [c[3] for c in ohlcv]
            closes = [c[4] for c in ohlcv]

            indicators = Indicators(ohlcv)
            atr_values = indicators.atr()
            atr_now = atr_values[-1]

            sl_price, tp_price = self.calculate_sl_tp(entry_price, side, atr_now)

            logging.info(f"\n{symbol} 🎯 TP dinâmico: {tp_price} | 🛑 SL dinâmico: {sl_price}")

            close_side = 'sell' if side == 'buy' else 'buy'
            await order_manager.create_tp_sl_orders(symbol, entry_amount, tp_price, sl_price, close_side)

        except Exception:
            logging.exception(f"\n❌ Erro no bot para {symbol}")

        finally:
            await exchange.close()

    def _calculate_sleep_time(self):
        now = datetime.now(timezone.utc)

        unit = self.timeframe[-1]  # 'm', 'h', etc
        amount = int(self.timeframe[:-1])

        if unit == 'm':
            minutes = now.minute
            seconds = now.second

            next_minute = (math.floor(minutes / amount) + 1) * amount

            if next_minute >= 60:
                next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                next_candle = next_hour
            else:
                next_candle = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_minute)

            sleep_seconds = (next_candle - now).total_seconds()
            return max(sleep_seconds, 0)

        elif unit == 'h':
            hour = now.hour
            minutes = now.minute
            seconds = now.second

            next_hour = (math.floor(hour / amount) + 1) * amount
            if next_hour >= 24:
                next_day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                next_candle = next_day
            else:
                next_candle = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)

            sleep_seconds = (next_candle - now).total_seconds()
            return max(sleep_seconds, 0)

        else:
            return 60

    async def start(self):
        while True:
            try:
                exchange = ccxt.hyperliquid({
                    "walletAddress": self.wallet_address,
                    "privateKey": self.private_key,
                    "testnet": True,
                    'enableRateLimit': True
                })

                for pair in self.pairs:
                    candle_closed_time = await self.get_last_closed_candle_time(pair['symbol'], exchange)

                    if candle_closed_time != self.last_candle_times[pair['symbol']]:
                        logging.info(f"\n🕒 Nova vela detectada para {pair['symbol']} ({candle_closed_time}). Executando bot...")
                        self.last_candle_times[pair['symbol']] = candle_closed_time
                        await self.run_pair(pair)
                    else:
                        logging.info(f"⌛ Aguardando nova vela para {pair['symbol']}... Última executada: {self.last_candle_times[pair['symbol']]}")

                sleep_time = self._calculate_sleep_time()
                logging.info(f"⏳ Dormindo por {sleep_time:.1f} segundos até próxima vela.")
                await asyncio.sleep(sleep_time)

                await exchange.close()

            except Exception:
                logging.exception("❌ Erro no loop principal")
                await asyncio.sleep(60)

    async def get_last_closed_candle_time(self, symbol, exchange):
        try:
            candles = await exchange.fetch_ohlcv(symbol, timeframe=self.timeframe)
            last_candle = candles[-2]  # vela fechada mais recente
            timestamp = last_candle[0]

            utc_dt = datetime.utcfromtimestamp(timestamp / 1000).replace(tzinfo=pytz.UTC)
            lisbon_tz = pytz.timezone('Europe/Lisbon')
            local_dt = utc_dt.astimezone(lisbon_tz)

            return local_dt

        except Exception:
            logging.exception(f"❌ Erro ao obter última vela fechada para {symbol}")
            return None



