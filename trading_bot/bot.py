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
from commons.utils.config_loader import PairConfig
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from strategies.strategy_manager import StrategyManager  # Para cálculo ATR
from trading_bot.exchange_base import ExchangeBase
from trading_bot.exit_logic.exit_logic_trailing_stop import \
    ExitLogicTrailingStop

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)


class TradingBot:
    def __init__(self, exchange_client: ExchangeBase, strategy: StrategyManager, helpers: TradingHelpers,
                 pairs: list[PairConfig], timeframe: TimeframeEnum, exchange_name: str = "EXCHANGE"):
        self.exchange_client = exchange_client
        self.exchange_name = exchange_name
        self.helpers = helpers
        self.pairs = pairs
        self.timeframe: TimeframeEnum = timeframe
        self.last_candle_times: dict[str, datetime | None] = {pair.symbol: None for pair in self.pairs}
        self.signal = None
        self.strategy = strategy
        # self.params_loader = BestParamsLoader()
        self.exit_logic = ExitLogicTrailingStop(self.helpers, self.exchange_client)

    async def run_pair(self, pair: PairConfig) -> SignalResult:
        symbol = pair.symbol
        leverage = int(pair.leverage)
        capital_pct = float(pair.capital)
        ohlcv_obj: OhlcvFormat = await self.exchange_client.fetch_ohlcv(symbol, self.timeframe, None,
                                                                        self.strategy.REQUIRED_CANDLES_500, True)

        ohlcv = ohlcv_obj.ohlcv
        ohlcv_higher = ohlcv_obj.ohlcv_higher
        logging.info(f"\n{'=' * 20} {self.exchange_name}: {symbol} {'=' * 20}")
        try:
            logging.info(f"🚀 Starting processing for {symbol}")

            # balance_total = await self.exchange_client.get_total_balance()
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

            await self.exchange_client.print_balance()
            await self.exchange_client.print_open_orders(symbol)

            current_position = await self.exchange_client.get_open_position(symbol)

            if current_position is not None:
                logging.info(f"[DEBUG] Available profit: {current_position.unrealizedPnl}")

            # 1) Verifica saída via ExitLogic, se posição aberta e tamanho > 0
            if current_position:
                position_size = float(current_position.size)
                logging.info(f"[DEBUG] Current position size: {position_size}")

                if position_size > 0:
                    should_exit = await self.exit_logic.should_exit(ohlcv, pair, signal, current_position, price_ref)
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

        return SignalResult(Signal.HOLD, None, None, None)

    def _calculate_sleep_time(self):
        now = datetime.now(timezone.utc)
        check_interval = 5  # Acordar de 5 em 5 minutos

        minutes = now.minute
        next_minute = (math.floor(minutes / check_interval) + 1) * check_interval

        if next_minute >= 60:
            next_check = now.replace(
                minute=0, second=0, microsecond=0
            ) + timedelta(hours=1)
        else:
            next_check = now.replace(
                minute=next_minute, second=0, microsecond=0
            )

        sleep_seconds = (next_check - now).total_seconds()
        return max(sleep_seconds, 10)  # Garante pelo menos 10s de respiro

    async def start(self):
        while True:
            try:
                logging.info("⏳ Loop de execução de 5 em 5 min (com filtro do candle em fecho) iniciado")
    
                now = datetime.now(pytz.utc)
                lisbon_tz = pytz.timezone('Europe/Lisbon')
                now_lisbon = now.astimezone(lisbon_tz)
    
                # Descobrir o minuto atual
                current_minute = now_lisbon.minute
                
                # O bloco de 15 minutos atual (0, 15, 30, 45)
                block_15m = (current_minute // 15) * 15
                
                # Quanto tempo passou desde o início do bloco de 15 min? (0, 5, ou 10 minutos)
                elapsed_in_block = current_minute - block_15m
    
                # 🚫 IGNORAR O ÚLTIMO CICLO DE 5 MINUTOS:
                # Se elapsed_in_block == 10, significa que estamos no intervalo dos 10 aos 15 minutos (o candle está a fechar). 
                # O bot salta esta execução e vai logo dormir para o próximo bloco de 15m!
                if elapsed_in_block == 10:
                    logging.info(f"⏳ Estamos no minuto {current_minute} (último terço do bloco de 15m). A ignorar o candle em fecho.")
                    sleep_time = self._calculate_sleep_time_with_buffer()
                    await asyncio.sleep(sleep_time)
                    continue
    
                # Caso contrário (minuto 0 ou minuto 5 do bloco), executa normalmente
                minute_block = (current_minute // 5) * 5
                current_5m_slot = now_lisbon.replace(minute=minute_block, second=0, microsecond=0)
    
                for pair in self.pairs:
                    last_exec = self.last_candle_times.get(pair.symbol)
    
                    if last_exec != current_5m_slot:
                        logging.info(
                            f"\n⚡ Executando para {pair.symbol} no slot de 5m ({current_5m_slot.strftime('%H:%M')})...")
                        self.last_candle_times[pair.symbol] = current_5m_slot
                        await self.run_pair(pair)
                    else:
                        logging.info(f"⌛ Já executado para {pair.symbol} no slot {current_5m_slot.strftime('%H:%M')}.")
    
                sleep_time = self._calculate_sleep_time_with_buffer()
                logging.info(f"⏳ Dormindo por {sleep_time:.1f} segundos até ao próximo slot.")
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

    def _calculate_sleep_time_with_buffer(self):
        now = datetime.now(timezone.utc)
        minutes = now.minute
        
        # FORÇAMOS SEMPRE A 5 MINUTOS, pois o loop corre de 5 em 5 min
        amount = 5
        next_minute = (math.floor(minutes / amount) + 1) * amount

        if next_minute >= 60:
            next_check = now.replace(
                minute=0, second=0, microsecond=0
            ) + timedelta(hours=1)
        else:
            next_check = now.replace(
                minute=next_minute, second=0, microsecond=0
            )

        sleep_seconds = (next_check - now).total_seconds()

        # 🛡️ BUFFER DE 15 SEGUNDOS: Acorda 15s DEPOIS da vela de 5m fechar
        BUFFER_SECONDS = 15
        return max(sleep_seconds + BUFFER_SECONDS, BUFFER_SECONDS)

    def _calculate_sleep_time_with_buffer_(self):
        now = datetime.now(timezone.utc)

        # 🛡️ LÊ DINAMICAMENTE O TIMEFRAME (ex: "15m" vira 15 minutos reais)
        unit = self.timeframe[-1]
        amount = int(self.timeframe[:-1])

        if unit == "m":
            minutes = now.minute
            next_minute = (math.floor(minutes / amount) + 1) * amount

            if next_minute >= 60:
                next_check = now.replace(
                    minute=0, second=0, microsecond=0
                ) + timedelta(hours=1)
            else:
                next_check = now.replace(
                    minute=next_minute, second=0, microsecond=0
                )

            sleep_seconds = (next_check - now).total_seconds()

            # 🛡️ BUFFER DE 15 SEGUNDOS: Acorda 15s DEPOIS da vela fechar
            BUFFER_SECONDS = 15
            return max(sleep_seconds + BUFFER_SECONDS, BUFFER_SECONDS)

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

            BUFFER_SECONDS = 15
            return max(sleep_seconds + BUFFER_SECONDS, BUFFER_SECONDS)

        else:
            return 60

    async def start_with_interval_5(self):
        while True:
            try:
                logging.info("⏳ Loop de execução de 5 em 5 min (Candle Atual) iniciado")

                now = datetime.now(pytz.utc)
                lisbon_tz = pytz.timezone('Europe/Lisbon')
                now_lisbon = now.astimezone(lisbon_tz)

                # Define o bloco atual de 5 minutos (ex: 10:00, 10:05, 10:10...)
                minute_block = (now_lisbon.minute // 5) * 5
                current_5m_slot = now_lisbon.replace(minute=minute_block, second=0, microsecond=0)

                for pair in self.pairs:
                    # Se ainda não executou neste slot de 5 minutos, executa com o candle atual
                    last_exec = self.last_candle_times.get(pair.symbol)

                    if last_exec != current_5m_slot:
                        logging.info(
                            f"\n⚡ Executando para {pair.symbol} no slot {current_5m_slot.strftime('%H:%M')} (Candle atual)...")
                        self.last_candle_times[pair.symbol] = current_5m_slot
                        await self.run_pair(pair)
                    else:
                        logging.info(f"⌛ Já executado para {pair.symbol} no slot {current_5m_slot.strftime('%H:%M')}.")

                sleep_time = self._calculate_sleep_time()
                logging.info(f"⏳ Dormindo por {sleep_time:.1f} segundos até ao próximo slot de 5 min.")
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

    def _calculate_sleep_time_old(self):
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

    async def start_old(self):
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
                        logging.info(
                            f"⌛ Aguardando nova vela para {pair.symbol}... Última executada: {self.last_candle_times[pair.symbol]}")

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
            candles_obj: OhlcvFormat = await self.exchange_client.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=5)
            candles: OhlcvWrapper = candles_obj.ohlcv
            last_candle = candles.get_last_closed_candle()
            timestamp = last_candle.timestamp

            utc_dt = datetime.fromtimestamp(timestamp / 1000).replace(
                tzinfo=pytz.UTC  # type: ignore
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
