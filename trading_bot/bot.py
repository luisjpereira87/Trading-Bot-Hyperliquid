import asyncio
import logging
import math
import os
from datetime import datetime, timedelta, timezone

import ccxt.async_support as ccxt  # type: ignore
import pytz  # type: ignore

from enums.signal_enum import Signal
from strategies.indicators import Indicators  # Para cálculo ATR
from strategies.signal_result import SignalResult
from utils.config_loader import PairConfig, load_pair_configs

from .exchange_client import ExchangeClient
from .exit_logic import ExitLogic
from .order_manager import OrderManager
from .strategy_manager import StrategyManager
from .trading_helpers import TradingHelpers


class TradingBot:
    def __init__(self):
        self.wallet_address = os.getenv("WALLET_ADDRESS")
        self.private_key = os.getenv("PRIVATE_KEY")

        if not self.wallet_address or not self.private_key:
            raise ValueError(
                "Variáveis de ambiente WALLET_ADDRESS e PRIVATE_KEY devem estar definidas"
            )

        self.timeframe = "15m"
        #self.tp_pct = 0.03
        #self.sl_pct = 0.01
        #self.tp_factor = 2
        self.atr_period = 14

        self.pairs = load_pair_configs()

        self.last_candle_times: dict[str, datetime | None] = {pair.symbol: None for pair in self.pairs}

        self.exchange = ccxt.hyperliquid(
            {
                "walletAddress": self.wallet_address,
                "privateKey": self.private_key,
                "testnet": True,
                "enableRateLimit": True,
                "options": {"defaultSlippage": 0.01},
            }
        )
        self.order_manager = OrderManager(self.exchange)
        self.helpers = TradingHelpers()
        self.exit_logic = ExitLogic(self.helpers, self.order_manager)

    async def run_pair(self, pair:PairConfig):
        symbol = pair.symbol
        leverage = int(pair.leverage)
        capital_pct = float(pair.capital)
        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=self.atr_period + 1)
        indicators = Indicators(ohlcv)
        atr_values = indicators.atr()
        atr_now = atr_values[-1]

        try:
            logging.info(f"🚀 Starting processing for {symbol}")

            exchange_client = ExchangeClient(self.exchange, self.wallet_address, symbol, leverage)

            balance_total = await exchange_client.get_total_balance()
            capital_amount = balance_total * capital_pct * leverage

            signal = await StrategyManager(self.exchange, symbol, self.timeframe, 'ml').get_signal()
            #signal.signal = Signal.BUY

            await exchange_client.print_balance()
            await exchange_client.print_open_orders(symbol)

            current_position = await exchange_client.get_open_position(symbol)

            if current_position:
                should_exit = await self.exit_logic.should_exit(self.exchange, pair, signal, current_position, atr_now)
                if should_exit:
                    return

            if signal.signal not in [Signal.BUY, Signal.SELL]:
                logging.info(f"\n⛔ No valid signal for {symbol}. Skipping.")
                return

            current_position = await exchange_client.get_open_position(symbol)
            if current_position:
                if self.helpers.is_signal_opposite_position(signal.signal, Signal.from_str(current_position["side"])):
                    await self.order_manager.close_position(
                        symbol, float(current_position["size"]), self.helpers.get_opposite_side(Signal.from_str(current_position["side"]))
                    )
                    current_position = None
                else:
                    logging.info(f"⚠️ Position already in same direction for {symbol}, skipping new entry.")
                    return

            if not current_position:
                await exchange_client.cancel_all_orders(symbol)
                await self.open_new_position(exchange_client, signal.signal, capital_amount, pair, signal.sl, signal.tp)

            """
            position = await exchange_client.get_open_position(symbol)
            if position:
                await self.order_manager.manage_tp_sl(exchange_client, position, signal, symbol, atr_now)
            else:
                logging.info(f"⚠️ No open position for {symbol}. Skipping TP/SL management.")

            """
            logging.info(f"✅ Processing for {symbol} completed successfully")

        except ccxt.NetworkError as e:
            logging.error(f"⚠️ Network error for {symbol}: {str(e)}")
        except ccxt.ExchangeError as e:
            logging.error(f"⚠️ Exchange error for {symbol}: {str(e)}")
        except Exception:
            logging.exception(f"\n❌ Bot error for {symbol}")
    
    async def try_close_position_with_profit(self, current_position, symbol, atr_now):
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            mark_price = ticker.get("last") or ticker.get("close")
            if mark_price is None:
                return False

            entry_price = current_position["entryPrice"]

            if current_position["side"] == "buy":
                profit_pct = (mark_price - entry_price) / entry_price
            else:
                profit_pct = (entry_price - mark_price) / entry_price

            profit_abs = profit_pct * float(current_position["notional"])
            profit_min_pct = 0.5 * (atr_now / entry_price)

            pair = self.helpers.get_pair(symbol, self.pairs)
            profit_min_abs = getattr(pair, "min_profit_abs", 5.0)  # 5.0 é valor padrão se não existir min_profit_abs

            logging.info(
                f"📈 Checking dynamic close: Profit = {profit_abs:.2f} USDC ({profit_pct*100:.2f}%), Min ATR% = {profit_min_pct*100:.2f}%, Min $ = {profit_min_abs:.2f}"
            )

            if profit_pct >= profit_min_pct and profit_abs >= profit_min_abs:
                logging.info(
                    f"💰 Dynamic exit triggered for {symbol} with {profit_abs:.2f} USDC profit ({profit_pct*100:.2f}%)"
                )
                try:
                    await self.order_manager.close_position(symbol, float(current_position["size"]), self.helpers.get_opposite_side(current_position["side"]))
                    logging.info(f"✅ Position dynamically closed with profit for {symbol}")
                    return True
                except Exception as e:
                    logging.error(f"Error closing position dynamically: {e}")
                    return False

            return False
        except Exception:
            logging.exception("❌ Error in dynamic profit-based close check")
            return False

    async def open_new_position(self, exchange_client, signal:Signal, capital_amount, pair, sl, tp):
        price_ref = await exchange_client.get_reference_price()
        if not price_ref or price_ref <= 0:
            raise ValueError("❌ Invalid reference price (None or <= 0)")

        entry_amount = await exchange_client.calculate_entry_amount(price_ref, capital_amount)
        side = signal

        logging.info(
            f"{pair.symbol}: Sending entry order {side} with qty {entry_amount} at price {price_ref}"
        )

        min_order_value = 10
        if entry_amount * price_ref < min_order_value:
            logging.warning(
                f"🚫 Order below $10 minimum: {entry_amount * price_ref:.2f}"
            )
            return

        await exchange_client.place_entry_order(entry_amount, price_ref, side, sl, tp)

    def _calculate_sleep_time(self):
        now = datetime.now(timezone.utc)
        unit = self.timeframe[-1]
        amount = int(self.timeframe[:-1])

        if unit == "m":
            minutes = now.minute
            next_minute = (math.floor(minutes / amount) + 1) * amount
            if next_minute >= 60:
                next_candle = now.replace(
                    minute=0, second=0, microsecond=0
                ) + timedelta(hours=1)
            else:
                next_candle = now.replace(
                    minute=0, second=0, microsecond=0
                ) + timedelta(minutes=next_minute)
            sleep_seconds = (next_candle - now).total_seconds()
            return max(sleep_seconds, 0)

        elif unit == "h":
            hour = now.hour
            next_hour = (math.floor(hour / amount) + 1) * amount
            if next_hour >= 24:
                next_candle = now.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) + timedelta(days=1)
            else:
                next_candle = now.replace(
                    hour=next_hour, minute=0, second=0, microsecond=0
                )
            sleep_seconds = (next_candle - now).total_seconds()
            return max(sleep_seconds, 0)

        else:
            return 60

    async def start(self):
        while True:
            try:
                logging.info("⏳ Loop principal iniciado")

                for pair in self.pairs:
                    candle_closed_time = await self.get_last_closed_candle_time(
                        pair.symbol
                    )
                    if candle_closed_time != self.last_candle_times[pair.symbol]:
                        logging.info(
                            f"\n🕒 Nova vela detectada para {pair.symbol} ({candle_closed_time}). Executando bot..."
                        )
                   
                        self.last_candle_times[pair.symbol] = candle_closed_time
                        await self.run_pair(pair)
                    else:
                        logging.info(
                            f"⌛ Aguardando nova vela para {pair.symbol}... Última executada: {self.last_candle_times[pair.symbol]}"
                        )

                sleep_time = self._calculate_sleep_time()
                logging.info(
                    f"⏳ Dormindo por {sleep_time:.1f} segundos até próxima vela."
                )
                logging.info("✅ Loop principal finalizado. Aguardando próximo ciclo...")
                await asyncio.sleep(sleep_time)

            except ccxt.NetworkError as e:
                logging.error(f"⚠️ Network error no loop principal: {str(e)}")
                await asyncio.sleep(60)
            except ccxt.ExchangeError as e:
                logging.error(f"⚠️ Exchange error no loop principal: {str(e)}")
                await asyncio.sleep(60)
            except Exception:
                logging.exception("❌ Erro no loop principal")
                await asyncio.sleep(60)

    async def get_last_closed_candle_time(self, symbol):
        try:
            candles = await self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe)
            last_candle = candles[-2]
            timestamp = last_candle[0]
            utc_dt = datetime.utcfromtimestamp(timestamp / 1000).replace(
                tzinfo=pytz.UTC
            )
            lisbon_tz = pytz.timezone("Europe/Lisbon")
            local_dt = utc_dt.astimezone(lisbon_tz)
            return local_dt
        except Exception:
            logging.exception(f"❌ Erro ao obter última vela fechada para {symbol}")
            return None
