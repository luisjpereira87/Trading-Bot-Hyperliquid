import os
import pytz
import ccxt.async_support as ccxt
from .exchange_client import ExchangeClient
from .order_manager import OrderManager
from .strategy import Strategy
import asyncio
from datetime import datetime

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
            exchange_client = ExchangeClient(exchange, self.symbol, self.leverage)
            order_manager = OrderManager(exchange)
            strategy = Strategy(exchange, self.symbol, self.timeframe)

            signal = await strategy.get_signal()
            if signal not in ['buy', 'sell']:
                print("\n⛔ Nenhum sinal válido encontrado. Bot parado.")
                return

            await exchange_client.print_balance()
            await exchange_client.print_open_orders()

            await exchange_client.cancel_all_orders()

            current_position = await exchange_client.get_open_position()
            if current_position:
                current_side = current_position['side']
                if (signal == 'buy' and current_side == 'sell') or (signal == 'sell' and current_side == 'buy'):
                    await order_manager.close_position(self.symbol, current_position['size'], current_side)

            price_ref = await exchange_client.get_reference_price()
            entry_amount = await exchange_client.calculate_entry_amount(price_ref)

            side = 'buy' if signal == 'buy' else 'sell'
            await exchange_client.place_entry_order(entry_amount, price_ref, side)
            entry_price = await exchange_client.get_entry_price()

            take_profit_price = round(entry_price * (1 + self.tp_pct), 2) if side == 'buy' else round(entry_price * (1 - self.tp_pct), 2)
            stop_loss_price = round(entry_price * (1 - self.sl_pct), 2) if side == 'buy' else round(entry_price * (1 + self.sl_pct), 2)
            print(f"\n🎯 TP: {take_profit_price} | 🛑 SL: {stop_loss_price}")

            await order_manager.create_tp_sl_orders(
                self.symbol,
                entry_amount,
                take_profit_price,
                stop_loss_price,
                side
            )

        except Exception as e:
            print("\n❌ Erro no bot:", e)
        finally:
            await exchange.close()

    async def start(self):
        while True:
            try:
                candle_closed_time = await self.get_last_closed_candle_time()

                if candle_closed_time != self.last_candle_time:
                    print(f"\n🕒 Nova vela detectada ({candle_closed_time}). Executando bot...")
                    self.last_candle_time = candle_closed_time
                    await self.run()
                else:
                    print(f"⌛ Aguardando nova vela... Última executada: {self.last_candle_time}")

                await asyncio.sleep(60)  # Checa a cada minuto
            except Exception as e:
                print(f"❌ Erro no loop principal: {e}")
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
            
            # Converte timestamp ms para datetime UTC
            utc_dt = datetime.utcfromtimestamp(timestamp / 1000).replace(tzinfo=pytz.UTC)
            # Define timezone Lisboa (com ajuste para horário de verão automático)
            lisbon_tz = pytz.timezone('Europe/Lisbon')
            local_dt = utc_dt.astimezone(lisbon_tz)
            
            return local_dt

        except Exception as e:
            print("❌ Erro ao obter última vela fechada:", e)
            await exchange.close()
            return None
