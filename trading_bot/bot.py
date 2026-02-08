import asyncio
import logging
import math
import os
import warnings
from datetime import datetime, timedelta, timezone

import ccxt.async_support as ccxt  # type: ignore
import pytz  # type: ignore

from commons.enums.signal_enum import Signal
from commons.enums.timeframe_enum import TimeframeEnum
from commons.helpers.trading_helpers import TradingHelpers
from commons.models.ohlcv_format_dclass import OhlcvFormat
from commons.models.signal_result_dclass import SignalResult
from commons.utils.best_params_loader import BestParamsLoader
from commons.utils.config_loader import PairConfig
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from strategies.strategy_manager import StrategyManager  # Para cálculo ATR
from trading_bot.exchange_base import ExchangeBase
from trading_bot.exchange_client import ExchangeClient
from trading_bot.exit_logic.exit_logic_ema_based import ExitLogicEmaBased

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)


class TradingBot:
    def __init__(self, exchange_client: ExchangeBase, strategy: StrategyManager, helpers: TradingHelpers, pairs: list[PairConfig], timeframe: TimeframeEnum, exchange_name: str = "EXCHANGE"):
        self.exchange_client = exchange_client
        self.exchange_name = exchange_name
        self.helpers = helpers
        self.pairs = pairs
        self.timeframe: TimeframeEnum = timeframe
        self.last_candle_times: dict[str, datetime | None] = {pair.symbol: None for pair in self.pairs}
        self.signal = None
        self.strategy = strategy
        #self.params_loader = BestParamsLoader()
        self.exit_logic = ExitLogicEmaBased(self.helpers, self.exchange_client)

    async def run_pair(self, pair: PairConfig) -> SignalResult:
        symbol = pair.symbol
        leverage = int(pair.leverage)
        capital_pct = float(pair.capital)
        ohlcv_obj: OhlcvFormat = await self.exchange_client.fetch_ohlcv(symbol, self.timeframe, self.strategy.REQUIRED_CANDLES_200, True)

        ohlcv = ohlcv_obj.ohlcv
        ohlcv_higher = ohlcv_obj.ohlcv_higher
        logging.info(f"\n{'='*20} {self.exchange_name}: {symbol} {'='*20}")
        try:
            logging.info(f"🚀 Starting processing for {symbol}")

            #balance_total = await self.exchange_client.get_total_balance()
            available_balance = await self.exchange_client.get_available_balance()
            capital_amount = (available_balance * capital_pct) * leverage

            logging.info(f"[DEBUG] Available capital_pct: {capital_pct}")
            logging.info(f"[DEBUG] Available balance: {available_balance}")
            logging.info(f"[DEBUG] Capital to deploy (after leverage): {capital_amount}")

            last_closed = ohlcv.get_last_closed_candle()
            ts = datetime.fromtimestamp(last_closed.timestamp / 1000).astimezone(pytz.timezone('Europe/Lisbon'))
            logging.info(f"✅ Último candle fechado usado para {symbol}: {ts.strftime('%Y-%m-%d %H:%M:%S')}")

            price_ref = await self.exchange_client.get_entry_price(symbol)
            logging.info(f"[DEBUG] Current price: {price_ref}")

            self.strategy.required_init(ohlcv, ohlcv_higher, symbol, price_ref)

            signal = await self.strategy.get_signal()

            logging.info(f"[DEBUG] signal: {signal}")

            #await self.exchange_client.print_balance()
            #$await self.exchange_client.print_open_orders(symbol)

            current_position = await self.exchange_client.get_open_position(symbol)

            if current_position is not None:
                logging.info(f"[DEBUG] Available profit: { current_position.unrealizedPnl}")
           
            # 1) Verifica saída via ExitLogic, se posição aberta e tamanho > 0
            if current_position:
                position_size = float(current_position.size)
                logging.info(f"[DEBUG] Current position size: {position_size}")

                if position_size > 0:
                    should_exit = await self.exit_logic.should_exit(ohlcv, pair, signal, current_position)
                    if should_exit:
                        current_position = None  # atualiza para evitar fechar de novo

            # 2) Se não há sinal válido, skip
            if signal.signal not in [Signal.BUY, Signal.SELL]:
                logging.info(f"\n⛔ No valid signal for {symbol}. Skipping.")
                return signal

            # 3) Se posição ainda aberta, verifica se sinal é oposto para fechar
            if current_position:
                if self.helpers.is_signal_opposite_position(signal.signal, Signal.from_str(current_position.side)):
                    logging.info(f"[DEBUG] Closing position due to opposite signal for {symbol}")
                    await self.exchange_client.close_position(
                        symbol, float(current_position.size),
                        self.helpers.get_opposite_side(Signal.from_str(current_position.side))
                    )
                    current_position = None
                else:
                    logging.info(f"⚠️ Position already in same direction for {symbol}, skipping new entry.")
                    return signal

            # 4) Se não há posição aberta, abre nova posição
            if not current_position:
                logging.info(f"[DEBUG] Cancelling all orders before opening new position for {symbol}")
                await self.exchange_client.cancel_all_orders(symbol)
                logging.info(f"[DEBUG] Opening new position for {symbol} with size {capital_amount}")
                order = await self.exchange_client.open_new_position(
                    symbol, leverage, signal.signal, capital_amount, pair, signal.sl, signal.tp
                )

            logging.info(f"✅ Processing for {symbol} completed successfully")
            return signal

        except ccxt.NetworkError as e:
            logging.error(f"⚠️ Network error for {symbol}: {str(e)}")
        except ccxt.ExchangeError as e:
            logging.error(f"⚠️ Exchange error for {symbol}: {str(e)}")
        except Exception:
            logging.exception(f"\n❌ Bot error for {symbol}")
            
        return signal
    

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

                # Assumimos que todas as velas fecharam neste timestamp
                now = datetime.now(pytz.utc)


                # Converter para hora de Lisboa (Europa/Lisbon)
                lisbon_tz = pytz.timezone('Europe/Lisbon')
                now_lisbon = now.astimezone(lisbon_tz)

                rounded_now = self._get_last_closed_candle_time_global(now_lisbon)

                for pair in self.pairs:
                    if self.last_candle_times[pair.symbol] != rounded_now:
                        logging.info(f"\n🕒 Nova vela detectada para {pair.symbol} ({rounded_now}). Executando bot...")
                        self.last_candle_times[pair.symbol] = rounded_now
                        await self.run_pair(pair)
                    else:
                        logging.info(f"⌛ Aguardando nova vela para {pair.symbol}... Última executada: {self.last_candle_times[pair.symbol]}")


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
            candles_obj: OhlcvFormat = await self.exchange_client.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=False)
            candles: OhlcvWrapper = candles_obj.ohlcv
            last_candle = candles.get_last_closed_candle()
            timestamp = last_candle.timestamp

            utc_dt = datetime.fromtimestamp(timestamp / 1000).replace(
                tzinfo=pytz.UTC # type: ignore
            )
            lisbon_tz = pytz.timezone("Europe/Lisbon")
            local_dt = utc_dt.astimezone(lisbon_tz)
            return local_dt
        except Exception:
            logging.exception(f"❌ Erro ao obter última vela fechada para {symbol}")
            return None
        
    def _get_last_closed_candle_time_global(self, now: datetime) -> datetime:
        unit = self.timeframe[-1]
        amount = int(self.timeframe[:-1])

        if unit == "m":
            minute = (now.minute // amount) * amount
            return now.replace(minute=minute, second=0, microsecond=0)
        elif unit == "h":
            hour = (now.hour // amount) * amount
            return now.replace(hour=hour, minute=0, second=0, microsecond=0)
        else:
            raise ValueError("Unsupported timeframe")

            
            
  