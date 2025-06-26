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

class TradingBot:
    def __init__(self):
        self.wallet_address = os.getenv("WALLET_ADDRESS")
        self.private_key = os.getenv("PRIVATE_KEY")
        self.symbol = 'BTC/USDC:USDC'
        self.leverage = 5
        self.tp_pct = 0.03
        self.sl_pct = 0.015
        self.timeframe = '15m'
        self.last_candle_time = None  # para controle de execução por vela

    async def run(self):
        exchange = ccxt.hyperliquid({
            "walletAddress": self.wallet_address,
            "privateKey": self.private_key,
            "testnet": True,
            'enableRateLimit': True,
            'options': {
                'defaultSlippage': 0.01
            }
        })

        try:
            exchange_client = ExchangeClient(exchange, self.wallet_address, self.symbol, self.leverage)
            order_manager = OrderManager(exchange)
            strategy = Strategy(exchange, self.symbol, self.timeframe)

            signal = await strategy.get_signal()
            if signal not in ['buy', 'sell']:
                logging.info("\n⛔ Nenhum sinal válido encontrado. Bot parado.")
                return

            await exchange_client.print_balance()
            await exchange_client.print_open_orders()

            await exchange_client.cancel_all_orders()

            current_position = await exchange_client.get_open_position()

            if current_position:
                current_side_ccxt = current_position['side']  # 'long' ou 'short'
                current_size = float(current_position['size'])
                current_side = 'buy' if current_side_ccxt == 'long' else 'sell'

                if (signal == 'buy' and current_side == 'sell') or (signal == 'sell' and current_side == 'buy'):
                    logging.info(f"Fechando posição oposta: {current_side_ccxt} de tamanho {current_size}")
                    await order_manager.close_position(self.symbol, current_size, current_side)
                    current_position = None

            if not current_position:
                price_ref = await exchange_client.get_reference_price()
                entry_amount = await exchange_client.calculate_entry_amount(price_ref)

                side = signal
                logging.info(f"Enviando ordem de entrada {side} com quantidade {entry_amount} a preço {price_ref}")
                await exchange_client.place_entry_order(entry_amount, price_ref, side)
                entry_price = await exchange_client.get_entry_price()

                take_profit_price = round(entry_price * (1 + self.tp_pct), 2) if side == 'buy' else round(entry_price * (1 - self.tp_pct), 2)
                stop_loss_price = round(entry_price * (1 - self.sl_pct), 2) if side == 'buy' else round(entry_price * (1 + self.sl_pct), 2)
                logging.info(f"\n🎯 TP: {take_profit_price} | 🛑 SL: {stop_loss_price}")

                close_side = 'sell' if side == 'buy' else 'buy'
                await order_manager.create_tp_sl_orders(
                    self.symbol,
                    entry_amount,
                    take_profit_price,
                    stop_loss_price,
                    close_side
                )
            else:
                current_side_ccxt = current_position['side']
                current_size = float(current_position['size'])
                current_side = 'buy' if current_side_ccxt == 'long' else 'sell'

                close_side = 'sell' if current_side == 'buy' else 'buy'

                entry_price = current_position.get('entryPrice')
                if entry_price is None:
                    entry_price = await exchange_client.get_entry_price()

                take_profit_price = round(entry_price * (1 + self.tp_pct), 2) if current_side == 'buy' else round(entry_price * (1 - self.tp_pct), 2)
                stop_loss_price = round(entry_price * (1 - self.sl_pct), 2) if current_side == 'buy' else round(entry_price * (1 + self.sl_pct), 2)
                logging.info(f"\n🎯 TP: {take_profit_price} | 🛑 SL: {stop_loss_price}")

                await order_manager.create_tp_sl_orders(
                    self.symbol,
                    current_size,
                    take_profit_price,
                    stop_loss_price,
                    close_side
                )

        except Exception as e:
            logging.error(f"\n❌ Erro no bot: {e}")
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
                candle_closed_time = await self.get_last_closed_candle_time()

                if candle_closed_time != self.last_candle_time:
                    logging.info(f"\n🕒 Nova vela detectada ({candle_closed_time}). Executando bot...")
                    self.last_candle_time = candle_closed_time
                    await self.run()
                else:
                    logging.info(f"⌛ Aguardando nova vela... Última executada: {self.last_candle_time}")

                sleep_time = self._calculate_sleep_time()
                logging.info(f"⏳ Dormindo por {sleep_time:.1f} segundos até próxima vela.")
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logging.error(f"❌ Erro no loop principal: {e}")
                await asyncio.sleep(60)

    async def get_last_closed_candle_time(self):
        exchange = ccxt.hyperliquid({
            "walletAddress": self.wallet_address,
            "privateKey": self.private_key,
            "testnet": True,
            'enableRateLimit': True
        })

        try:
            candles = await exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe)
            await exchange.close()
            last_candle = candles[-2]  # vela fechada mais recente
            timestamp = last_candle[0]

            utc_dt = datetime.utcfromtimestamp(timestamp / 1000).replace(tzinfo=pytz.UTC)
            lisbon_tz = pytz.timezone('Europe/Lisbon')
            local_dt = utc_dt.astimezone(lisbon_tz)

            return local_dt

        except Exception as e:
            logging.error("❌ Erro ao obter última vela fechada: %s", e)
            await exchange.close()
            return None



