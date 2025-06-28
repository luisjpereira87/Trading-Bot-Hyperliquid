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


class TradingBot:
    def __init__(self):
        self.wallet_address = os.getenv("WALLET_ADDRESS")
        self.private_key = os.getenv("PRIVATE_KEY")

        if not self.wallet_address or not self.private_key:
            raise ValueError(
                "Variáveis de ambiente WALLET_ADDRESS e PRIVATE_KEY devem estar definidas"
            )

        self.timeframe = "15m"
        self.tp_pct = 0.03
        self.sl_pct = 0.015
        self.tp_factor = 2
        self.atr_period = 14

        self.pairs = [
            {"symbol": "BTC/USDC:USDC", "leverage": 5, "capital": 0.10},
            {"symbol": "ETH/USDC:USDC", "leverage": 10, "capital": 0.20},
        ]

        self.last_candle_times = {pair["symbol"]: None for pair in self.pairs}

        self.exchange = ccxt.hyperliquid(
            {
                "walletAddress": self.wallet_address,
                "privateKey": self.private_key,
                "testnet": True,
                "enableRateLimit": True,
                "options": {"defaultSlippage": 0.01},
            }
        )

    def calculate_sl_tp(self, entry_price, side, atr_now, mode="normal"):
        """
        Calcula SL e TP dinâmicos usando ATR e percentual mínimo, ajustados pelo modo.
        Inclui logs e validação para evitar valores extremos.

        Parâmetros:
        - entry_price: preço de entrada da posição
        - side: 'buy' ou 'sell'
        - atr_now: valor atual do ATR
        - mode: 'aggressive', 'normal' ou 'conservative' (padrão 'normal')
        """
        # Define parâmetros baseados no modo
        if mode == "aggressive":
            sl_pct = 0.008  # 0.8%
            tp_factor = 1.5
        elif mode == "conservative":
            sl_pct = 0.02  # 2%
            tp_factor = 3
        else:  # normal
            sl_pct = self.sl_pct
            tp_factor = self.tp_factor

        sl_min_dist = sl_pct * entry_price
        sl_distance = max(sl_min_dist, atr_now)

        logging.info(f"🔎 Cálculo TP/SL ({mode}):")
        logging.info(f"📈 Preço de entrada: {entry_price}")
        logging.info(f"📊 ATR atual: {atr_now}")
        logging.info(f"🧮 Distância mínima SL (%): {sl_min_dist}")
        logging.info(f"🧮 Distância usada SL (máx entre % e ATR): {sl_distance}")

        if side == "buy":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_factor * sl_distance
        else:  # sell
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_factor * sl_distance

        max_dist_pct = 0.10  # 10%
        sl_pct_off = abs((sl_price - entry_price) / entry_price)
        tp_pct_off = abs((tp_price - entry_price) / entry_price)

        if sl_pct_off > max_dist_pct or tp_pct_off > (max_dist_pct * tp_factor):
            logging.warning(
                f"🚫 SL ou TP fora de range aceitável. SL: {sl_price}, TP: {tp_price}"
            )
            raise ValueError(
                "SL ou TP calculado está fora do intervalo aceitável. Verifique os parâmetros ou o ATR."
            )

        logging.info(f"✅ SL final: {sl_price} ({sl_pct_off*100:.2f}%)")
        logging.info(f"✅ TP final: {tp_price} ({tp_pct_off*100:.2f}%)")

        return round(sl_price, 2), round(tp_price, 2)

    def is_opposite_side(self, signal, current_side):
        return (signal == "buy" and current_side == "sell") or (
            signal == "sell" and current_side == "buy"
        )

    async def run_pair(self, pair):
        symbol = pair["symbol"]
        leverage = int(pair["leverage"])
        capital_pct = float(pair["capital"])

        try:
            exchange_client = ExchangeClient(
                self.exchange, self.wallet_address, symbol, leverage
            )
            order_manager = OrderManager(self.exchange)
            strategy = Strategy(self.exchange, symbol, self.timeframe)

            balance_total = await exchange_client.get_total_balance()
            capital_amount = balance_total * capital_pct * leverage

            # Instanciar MLStrategy para o par
            ml_strategy = MLStrategy(self.exchange, symbol, timeframe=self.timeframe, train_interval=100)
            
            # Treinar e pegar sinal ML
            ml_signal = await ml_strategy.run()

            # (Opcional) Pegar sinal de outra estratégia, ex: UT Bot ou SuperTrend
            other_signal = await strategy.get_signal()

            # Combinar sinais, exemplo: se ML deu buy/sell, usar, senão fallback
            if ml_signal in ["buy", "sell"]:
                signal = {"side": ml_signal, "mode": "normal"}
            else:
                signal = other_signal
           
            if signal["side"] not in ["buy", "sell"]:
                logging.info(f"\n⛔ Nenhum sinal válido para {symbol}. Ignorando.")
                return

            await exchange_client.print_balance()
            await exchange_client.print_open_orders(symbol)
            await exchange_client.cancel_all_orders(symbol)

            current_position = await exchange_client.get_open_position()

            if current_position:
                closed_early = await self.try_close_position_aggressive(
                    current_position, signal, order_manager, symbol
                )
                if closed_early:
                    return

                if self.is_opposite_side(signal["side"], "buy" if current_position["side"] == "long" else "sell"):
                    await order_manager.close_position(
                        symbol, float(current_position["size"]), current_position["side"]
                    )
                    current_position = None

            if not current_position:
                await self.open_position(exchange_client, signal, capital_amount, pair)

            await self.manage_tp_sl(exchange_client, current_position or await exchange_client.get_open_position(), signal, order_manager, symbol)

        except Exception:
            logging.exception(f"\n❌ Erro no bot para {symbol}")

    async def try_close_position_aggressive(self, current_position, signal, order_manager, symbol):
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=self.atr_period + 1)
            indicators = Indicators(ohlcv)
            atr_values = indicators.atr()
            atr_now = atr_values[-1]

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
                await order_manager.close_position(symbol, float(current_position["size"]), close_side)
                return True
            return False
        except Exception:
            logging.exception("❌ Falha ao tentar fechar posição agressiva com lucro baseado em ATR")
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

    async def manage_tp_sl(self, exchange_client, position, signal, order_manager, symbol):
        entry_price = position.get("entryPrice")
        if entry_price is None:
            entry_price = await exchange_client.get_entry_price()

        entry_amount = float(position["size"])
        side = "buy" if position["side"] == "long" else "sell"

        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=self.atr_period + 1)
        indicators = Indicators(ohlcv)
        atr_values = indicators.atr()
        atr_now = atr_values[-1]

        if atr_now is None:
            logging.warning(f"❌ ATR não pôde ser calculado para {symbol}. Ignorando.")
            return

        sl_price, tp_price = self.calculate_sl_tp(entry_price, side, atr_now, signal["mode"])

        logging.info(f"\n{symbol} 🎯 TP dinâmico: {tp_price} | 🛑 SL dinâmico: {sl_price}")

        close_side = "sell" if side == "buy" else "buy"

        try:
            await order_manager.create_tp_sl_orders(symbol, entry_amount, tp_price, sl_price, close_side)
        except Exception:
            logging.exception("❌ Falha ao criar TP/SL. Fechando posição por segurança.")
            await order_manager.close_position(symbol, entry_amount, close_side)

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
                for pair in self.pairs:
                    candle_closed_time = await self.get_last_closed_candle_time(
                        pair["symbol"]
                    )
                    if candle_closed_time != self.last_candle_times[pair["symbol"]]:
                        logging.info(
                            f"\n🕒 Nova vela detectada para {pair['symbol']} ({candle_closed_time}). Executando bot..."
                        )
                        self.last_candle_times: dict[str, datetime | None] = {
                            pair["symbol"]: None for pair in self.pairs
                        }
                        await self.run_pair(pair)
                    else:
                        logging.info(
                            f"⌛ Aguardando nova vela para {pair['symbol']}... Última executada: {self.last_candle_times[pair['symbol']]}"
                        )

                sleep_time = self._calculate_sleep_time()
                logging.info(
                    f"⏳ Dormindo por {sleep_time:.1f} segundos até próxima vela."
                )
                await asyncio.sleep(sleep_time)

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
