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
    def __init__(self, exchange, exchange_client, wallet_address, helpers, pairs, timeframe, atr_period):
        self.exchange = exchange
        self.exchange_client = exchange_client
        self.wallet_address = wallet_address
        self.helpers = helpers
        #self.exit_logic = exit_logic
        self.pairs = pairs
        self.timeframe = timeframe
        self.atr_period = atr_period

        self.last_candle_times: dict[str, datetime | None] = {pair.symbol: None for pair in self.pairs}

        

    async def run_pair(self, pair: PairConfig):
        symbol = pair.symbol
        leverage = int(pair.leverage)
        capital_pct = float(pair.capital)

        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=self.atr_period + 1)
        indicators = Indicators(ohlcv)
        atr_values = indicators.atr()
        atr_now = atr_values[-1]

        try:
            logging.info(f"🚀 Starting processing for {symbol}")

            #exchange_client = ExchangeClient(self.exchange, self.wallet_address)
            exit_logic = ExitLogic(self.helpers, self.exchange_client)

            balance_total = await self.exchange_client.get_total_balance()
            capital_amount = balance_total * capital_pct * leverage

            signal = await StrategyManager(self.exchange, symbol, self.timeframe, 'ml').get_signal()

            await self.exchange_client.print_balance()
            await self.exchange_client.print_open_orders(symbol)

            current_position = await self.exchange_client.get_open_position(symbol)

            if current_position:
                side = Signal.from_str(current_position["side"])
                entry_price = float(current_position["entryPrice"])
                position_size = float(current_position["size"])

                should_exit = await exit_logic.should_exit( pair, signal, current_position, atr_now)
                if should_exit:
                    # Obtemos preço de saída estimado do orderbook
                    orderbook = await self.exchange.fetch_order_book(symbol)
                    if side == Signal.BUY:
                        exit_price = orderbook['bids'][0][0] if orderbook['bids'] else None
                    else:
                        exit_price = orderbook['asks'][0][0] if orderbook['asks'] else None

                    if exit_price is None:
                        logging.warning("⚠️ Preço de saída indisponível.")
                        return None

                    # Calcula lucro
                    pnl = (exit_price - entry_price) * position_size if side == Signal.BUY else (entry_price - exit_price) * position_size

                    # Fecha posição
                    await self.exchange_client.close_position(
                        symbol, position_size, self.helpers.get_opposite_side(side)
                    )

                    logging.info(f"💰 PnL realizado para {symbol}: {pnl:.2f}")

                    return {
                        "symbol": symbol,
                        "pnl": pnl,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "side": side.value,
                        "amount": position_size,
                    }

            if signal.signal not in [Signal.BUY, Signal.SELL]:
                logging.info(f"\n⛔ No valid signal for {symbol}. Skipping.")
                return None

            current_position = await self.exchange_client.get_open_position(symbol)
            if current_position:
                if self.helpers.is_signal_opposite_position(signal.signal, Signal.from_str(current_position["side"])):
                    await self.exchange_client.close_position(
                        symbol, float(current_position["size"]),
                        self.helpers.get_opposite_side(Signal.from_str(current_position["side"]))
                    )
                    current_position = None
                else:
                    logging.info(f"⚠️ Position already in same direction for {symbol}, skipping new entry.")
                    return None

            if not current_position:
                await self.exchange_client.cancel_all_orders(symbol)
                await self.exchange_client.open_new_position(symbol, leverage, signal.signal, capital_amount, pair, signal.sl, signal.tp)

            logging.info(f"✅ Processing for {symbol} completed successfully")
            return None

        except ccxt.NetworkError as e:
            logging.error(f"⚠️ Network error for {symbol}: {str(e)}")
        except ccxt.ExchangeError as e:
            logging.error(f"⚠️ Exchange error for {symbol}: {str(e)}")
        except Exception:
            logging.exception(f"\n❌ Bot error for {symbol}")
    

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
