import asyncio
import logging
import math
import os
from datetime import datetime, timedelta, timezone

import ccxt.async_support as ccxt  # type: ignore
import pytz  # type: ignore

from .exchange_client import ExchangeClient
from .machine_learning.ml_strategy import MLStrategy
from .order_manager import OrderManager
from .strategies.indicators import Indicators  # Para cálculo ATR
from .strategy import Strategy
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

        self.pairs = [
            {"symbol": "BTC/USDC:USDC", "leverage": 5, "capital": 0.10, "min_profit_abs": 10.0},
            {"symbol": "ETH/USDC:USDC", "leverage": 10, "capital": 0.20, "min_profit_abs": 5.0},
        ]

        self.last_candle_times: dict[str, datetime | None] = {pair["symbol"]: None for pair in self.pairs}

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
    
    async def get_combined_signal(self, symbol):
        strategy = Strategy(self.exchange, symbol, self.timeframe)
        ml_strategy = MLStrategy(self.exchange, symbol, timeframe=self.timeframe, train_interval=100)

        ml_signal = await ml_strategy.run()
        other_signal = await strategy.get_signal()

        if ml_signal in ["buy", "sell"]:
            logging.info(f"📊 Sinal recebido para {symbol}: {ml_signal}")
            return {"side": ml_signal, "mode": "normal"}
        logging.info(f"📊 Sinal recebido para {symbol}: {other_signal}")
        return other_signal

    async def run_pair(self, pair):
        symbol = pair["symbol"]
        leverage = int(pair["leverage"])
        capital_pct = float(pair["capital"])
        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=self.atr_period + 1)
        indicators = Indicators(ohlcv)
        atr_values = indicators.atr()
        atr_now = atr_values[-1]

        try:
            logging.info(f"🚀 Iniciando processamento do par {symbol}")

            exchange_client = ExchangeClient(self.exchange, self.wallet_address, symbol, leverage)

            balance_total = await exchange_client.get_total_balance()
            capital_amount = balance_total * capital_pct * leverage

            signal = await self.get_combined_signal(symbol)
           
            if signal.get("side") not in ["buy", "sell"]:
                logging.info(f"\n⛔ Nenhum sinal válido para {symbol}. Ignorando.")
                return

            await exchange_client.print_balance()
            await exchange_client.print_open_orders(symbol)

            current_position = await exchange_client.get_open_position(symbol)

            # 👉 Etapa 1: Fechamento dinâmico com lucro
            if current_position:
                closed_early = await self.try_close_position_dynamically(current_position, symbol, atr_now)
                if closed_early:
                    logging.info(f"✅ Posição encerrada de forma dinâmica com lucro em {symbol}")
                    return  # ⚠️ IMPORTANTE: encerra execução antes de abrir nova posição

            # 👉 Etapa 2: Se ainda tem posição, verificar se é contrária ao novo sinal
            current_position = await exchange_client.get_open_position(symbol)
            if current_position:
                # Verifica se o sinal é igual ao lado da posição aberta — se sim, ignora o sinal
                position_side_signal = "buy" if current_position["side"] == "long" else "sell"
                if signal["side"] == position_side_signal:
                    logging.info(f"⚠️ Já existe posição {position_side_signal} aberta para {symbol}, ignorando sinal {signal['side']}")
                    return  # Sai da execução para esse par, nada a fazer

                if self.helpers.is_signal_opposite_position(signal["side"], current_position["side"]):
                    await self.order_manager.close_position(
                        symbol, float(current_position["size"]), current_position["side"]
                    )
                    current_position = None  # Atualiza estado

            # 👉 Etapa 3: Se não há mais posição, abrir nova
            if not current_position:
                await exchange_client.cancel_all_orders(symbol)
                await self.open_position(exchange_client, signal, capital_amount, pair)

            # 👉 Etapa 4: Gerenciar TP/SL se houver posição
            position = await exchange_client.get_open_position(symbol)
            if position:
                await self.order_manager.manage_tp_sl(exchange_client, position, signal, symbol, atr_now)
            else:
                logging.info(f"⚠️ Sem posição aberta para {symbol}. Pulando gerenciamento de TP/SL.")

            logging.info(f"✅ Processamento do par {symbol} concluído com sucesso")

        except ccxt.NetworkError as e:
            logging.error(f"⚠️ Network error para {symbol}: {str(e)}")
        except ccxt.ExchangeError as e:
            logging.error(f"⚠️ Exchange error para {symbol}: {str(e)}")
        except Exception:
            logging.exception(f"\n❌ Erro no bot para {symbol}")

    async def try_close_position_aggressive(self, current_position, signal, symbol, atr_now):
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            mark_price = ticker.get("last") or ticker.get("close")
            if mark_price is None or signal.get("mode") != "aggressive":
                return False

            entry_price = current_position["entryPrice"]
            current_side = "buy" if current_position["side"] == "long" else "sell"

            if current_side == "buy":
                lucro_pct = (mark_price - entry_price) / entry_price
            else:
                lucro_pct = (entry_price - mark_price) / entry_price

            lucro_minimo_atr = 0.5 * (atr_now / entry_price)

            if lucro_pct >= lucro_minimo_atr:
                logging.info(
                    f"💰 Lucro baseado em ATR ({lucro_pct*100:.2f}%) acima do limite {lucro_minimo_atr*100:.2f}%, fechando posição {current_side} em {symbol}"
                )
                close_side = "sell" if current_side == "buy" else "buy"
                
                await self.order_manager.close_position(symbol, float(current_position["size"]), close_side)
                return True
            return False
        except Exception:
            logging.exception("❌ Falha ao tentar fechar posição agressiva com lucro baseado em ATR")
            return False
        
    async def try_close_position_dynamically(self, current_position, symbol, atr_now):
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            mark_price = ticker.get("last") or ticker.get("close")
            if mark_price is None:
                return False

            entry_price = current_position["entryPrice"]
            current_side = "buy" if current_position["side"] == "long" else "sell"

            if current_side == "buy":
                lucro_pct = (mark_price - entry_price) / entry_price
            else:
                lucro_pct = (entry_price - mark_price) / entry_price

            lucro_absoluto = lucro_pct * float(current_position["notional"])
            lucro_minimo_pct = 0.5 * (atr_now / entry_price)
            
            pair = self.helpers.get_pair(symbol, self.pairs)
            lucro_minimo_abs = pair.get("min_profit_abs", 5.0) if pair else 5.0

            if lucro_pct >= lucro_minimo_pct and lucro_absoluto >= lucro_minimo_abs:
                logging.info(
                    f"💰 Fechamento antecipado: Lucro atual = {lucro_absoluto:.2f} USDC ({lucro_pct*100:.2f}%), ATR min = {lucro_minimo_pct*100:.2f}%, min $ = {lucro_minimo_abs}."
                )
                try:
                    await self.order_manager.close_position(
                        symbol, float(current_position["size"]), current_position["side"]
                    )
                    return True
                except Exception as e:
                    logging.error(f"Erro ao tentar fechar posição antecipadamente: {e}")
                    # Aqui você pode decidir retornar False ou fazer alguma lógica alternativa
                    return False

            return False
        except Exception:
            logging.exception("❌ Erro ao tentar fechar posição de forma dinâmica com lucro")
            return False
    
    async def open_position(self, exchange_client, signal, capital_amount, pair):
        price_ref = await exchange_client.get_reference_price()
        if not price_ref or price_ref <= 0:
            raise ValueError("❌ Preço de referência inválido (None ou <= 0)")

        entry_amount = await exchange_client.calculate_entry_amount(price_ref, capital_amount)
        side = signal["side"]

        logging.info(
            f"{pair['symbol']}: Enviando ordem de entrada {side} com quantidade {entry_amount} a preço {price_ref}"
        )

        min_order_value = 10  # dólares
        if entry_amount * price_ref < min_order_value:
            logging.warning(
                f"🚫 Ordem abaixo do mínimo de $10: {entry_amount * price_ref:.2f}"
            )
            return

        await exchange_client.place_entry_order(entry_amount, price_ref, side)

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
                        pair["symbol"]
                    )
                    if candle_closed_time != self.last_candle_times[pair["symbol"]]:
                        logging.info(
                            f"\n🕒 Nova vela detectada para {pair['symbol']} ({candle_closed_time}). Executando bot..."
                        )
                   
                        self.last_candle_times[pair["symbol"]] = candle_closed_time
                        await self.run_pair(pair)
                    else:
                        logging.info(
                            f"⌛ Aguardando nova vela para {pair['symbol']}... Última executada: {self.last_candle_times[pair['symbol']]}"
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
