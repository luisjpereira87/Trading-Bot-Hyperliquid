import asyncio
import logging
import math
import os
import warnings
from datetime import datetime, timedelta, timezone

import ccxt.async_support as ccxt  # type: ignore
import pytz  # type: ignore

from commons.enums.signal_enum import Signal
from commons.utils.config_loader import PairConfig
from strategies.indicators import Indicators
from trading_bot.exit_logic import ExitLogic
from trading_bot.strategy_manager import StrategyManager  # Para cálculo ATR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)


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

        ohlcv = await self.exchange_client.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=self.atr_period + 1)
        indicators = Indicators(ohlcv)
        atr_values = indicators.atr()
        atr_now = atr_values[-1]

        try:
            logging.info(f"🚀 Starting processing for {symbol}")

            exit_logic = ExitLogic(self.helpers, self.exchange_client)

            balance_total = await self.exchange_client.get_total_balance()
            available_balance = await self.exchange_client.get_available_balance()
            capital_amount = min(available_balance, balance_total * capital_pct) * leverage
            logging.info(f"[DEBUG] Available balance: {available_balance}")
            logging.info(f"[DEBUG] Capital to deploy (after leverage): {capital_amount}")

            signal = await StrategyManager(self.exchange_client, symbol, self.timeframe, 'ml').get_signal()

            await self.exchange_client.print_balance()
            await self.exchange_client.print_open_orders(symbol)

            current_position = await self.exchange_client.get_open_position(symbol)

            # 1) Verifica saída via ExitLogic, se posição aberta e tamanho > 0
            if current_position:
                side = Signal.from_str(current_position["side"])
                position_size = float(current_position["size"])
                logging.info(f"[DEBUG] Current position size: {position_size}")

                if position_size > 0:
                    should_exit = await exit_logic.should_exit(pair, signal, current_position, atr_now)
                    if should_exit:
                        current_position = None  # atualiza para evitar fechar de novo
                        return

            # 2) Se não há sinal válido, skip
            if signal.signal not in [Signal.BUY, Signal.SELL]:
                logging.info(f"\n⛔ No valid signal for {symbol}. Skipping.")
                return None

            # 3) Se posição ainda aberta, verifica se sinal é oposto para fechar
            if current_position:
                if self.helpers.is_signal_opposite_position(signal.signal, Signal.from_str(current_position["side"])):
                    logging.info(f"[DEBUG] Closing position due to opposite signal for {symbol}")
                    await self.exchange_client.close_position(
                        symbol, float(current_position["size"]),
                        self.helpers.get_opposite_side(Signal.from_str(current_position["side"]))
                    )
                    current_position = None
                else:
                    logging.info(f"⚠️ Position already in same direction for {symbol}, skipping new entry.")
                    return None

            # 4) Se não há posição aberta, abre nova posição
            if not current_position:
                logging.info(f"[DEBUG] Cancelling all orders before opening new position for {symbol}")
                await self.exchange_client.cancel_all_orders(symbol)
                logging.info(f"[DEBUG] Opening new position for {symbol} with size {capital_amount}")
                await self.exchange_client.open_new_position(
                    symbol, leverage, signal.signal, capital_amount, pair, signal.sl, signal.tp
                )

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
