import logging
from datetime import datetime, timedelta, timezone

from commons.enums.signal_enum import Signal
from commons.enums.timeframe_enum import TimeframeEnum
from commons.helpers.trading_helpers import TradingHelpers
from commons.models.ohlcv_format_dclass import OhlcvFormat
from commons.models.open_position_dclass import OpenPosition
from commons.models.opened_order_dclass import OpenedOrder
from commons.utils.config_loader import PairConfig
from commons.utils.ohlcv_wrapper import OhlcvWrapper


class ExchangeClient:
    def __init__(self, exchange, wallet_address):
        self.exchange = exchange
        self.wallet_address = wallet_address
        self.helpers = TradingHelpers()

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: TimeframeEnum = TimeframeEnum.M15,
        limit: int = 14,
        is_higher: bool = False
    ) -> OhlcvFormat:
        try:
            # Fetch timeframe principal
            ohlcv_data = await self.exchange.fetch_ohlcv(symbol, timeframe.value, limit)

            # Timestamp esperado do último candle fechado
            expected_timestamp = self.get_expected_timestamp(timeframe)

            if ohlcv_data:
                last_ts = ohlcv_data[-1][0]

                if last_ts < expected_timestamp:
                    logging.debug(
                        f"[{symbol}] Candle em falta ({timeframe.value}) "
                        f"(expected={expected_timestamp}, last={last_ts}). "
                        f"Adicionando placeholder..."
                    )

                    # Remove o mais antigo
                    ohlcv_data.pop(0)

                    # Preço atual como referência
                    current_price = await self.get_entry_price(symbol)

                    # Adiciona candle "em falta"
                    ohlcv_data.append([
                        expected_timestamp,
                        current_price,  # open
                        current_price,  # high
                        current_price,  # low
                        current_price,  # close
                        0.0             # volume (placeholder)
                    ])

                    logging.debug(
                        f"[{symbol}] Placeholder adicionado "
                        f"(ts={expected_timestamp}, price={current_price})"
                    )

                elif last_ts > expected_timestamp:
                    logging.debug(
                        f"[{symbol}] Exchange devolveu candle futuro "
                        f"(last={last_ts}, expected={expected_timestamp}). Ignorado."
                    )

            # Fetch timeframe maior, se necessário
            ohlcv_higher_data = []
            if is_higher:
                higher_tf = timeframe.get_higher()
                ohlcv_higher_data = await self.exchange.fetch_ohlcv(symbol, higher_tf.value, limit)

            return OhlcvFormat(
                OhlcvWrapper(ohlcv_data),
                OhlcvWrapper(ohlcv_higher_data)
            )

        except Exception as e:
            logging.error(f"[{symbol}] Erro ao buscar os candles ({timeframe.value}): {e}")
            raise

    def get_expected_timestamp(self, timeframe: TimeframeEnum) -> int:
        """
        Retorna o timestamp em ms do ÚLTIMO candle FECHADO para o timeframe fornecido.
        """
        now = datetime.now(timezone.utc)

        # Define duração do candle em minutos
        tf_minutes = {
            TimeframeEnum.M1: 1,
            TimeframeEnum.M5: 5,
            TimeframeEnum.M15: 15,
            TimeframeEnum.M30: 30,
            TimeframeEnum.H1: 60,
            TimeframeEnum.H4: 240,
            TimeframeEnum.D1: 1440
        }.get(timeframe, 15)  # default 15 min

        # Quantos minutos já se passaram desde o início da hora
        minutes_passed = now.minute % tf_minutes
        seconds_passed = now.second
        microseconds_passed = now.microsecond

        # O último fechado é "agora" menos o tempo decorrido no candle atual
        delta = timedelta(
            minutes=minutes_passed,
            seconds=seconds_passed,
            microseconds=microseconds_passed
        )
        last_candle_time = now - delta

        # Retorna timestamp em ms
        return int(last_candle_time.timestamp() * 1000)

    async def get_available_balance(self) -> float:
        try:
            balance = await self.exchange.fetch_balance(params={'user': self.wallet_address})
            return balance['total']['USDC']
        except Exception as e:
            logging.error(f"Erro ao buscar saldo: {e}")
            raise

    
    async def fetch_ticker(self, symbol: str):
        try:
            return await self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logging.error(f"Erro ao buscar preço atual: {e}")

    async def print_balance(self):
        try:
            balance = await self.exchange.fetch_balance(params={'user': self.wallet_address})
            logging.info(f"💰 Saldo total: {balance['total']}")
        except Exception as e:
            logging.error(f"Erro ao buscar saldo: {e}")

    async def print_open_orders(self, symbol: str = ''):
        try:
            params = {'user': self.wallet_address}
            if symbol:
                open_orders = await self.exchange.fetch_open_orders(symbol, params=params)
            else:
                open_orders = await self.exchange.fetch_open_orders(params=params)
            logging.info(f"📘 Ordens abertas para {symbol if symbol else 'todos símbolos'} ({len(open_orders)}):")
            for order in open_orders:
                logging.info(f"  ID: {order.get('id')}, Side: {order.get('side')}, Price: {order.get('price')}, Amount: {order.get('amount')}, Status: {order.get('status')}")
        except Exception as e:
            logging.error(f"Erro ao buscar ordens abertas: {e}")

    async def cancel_all_orders(self, symbol: str = ''):
        try:
            params = {'user': self.wallet_address}
            if symbol:
                open_orders = await self.exchange.fetch_open_orders(symbol, params=params)
            else:
                open_orders = await self.exchange.fetch_open_orders(params=params)
    
            for order in open_orders:
                await self.exchange.cancel_order(order['id'], order['symbol'])
            logging.info(f"🔁 Todas as ordens foram canceladas para {symbol if symbol else 'todos símbolos'}.")
        except Exception as e:
            logging.error(f"Erro ao cancelar ordens: {e}")

    async def get_open_position(self, symbol: str = '') -> (OpenPosition | None):
        try:
            positions = await self.exchange.fetch_positions(params={'user': self.wallet_address})
            for pos in positions:
                if pos["symbol"] == symbol and float(pos.get('contracts', 0)) > 0:

                    size = float(pos['contracts'])
                    entry_price = pos.get('entryPrice') or pos.get('entry_price') or pos.get('averagePrice') or 0.0
                    id = pos.get('id') or pos.get('info', {}).get('order', {}).get('oid')
                    unrealizedPnl = pos.get('unrealizedPnl') or pos.get('unrealizedPnl')
                    
                    return OpenPosition(self.helpers.position_side_to_signal_side(pos['side']), size, entry_price, id, size * entry_price, None, None, unrealizedPnl)

        except Exception as e:
            logging.error(f"Erro ao obter posições abertas: {e}")
        return None

    async def get_reference_price(self, symbol: str):
        try:
            order_book = await self.exchange.fetch_order_book(symbol)
            asks = order_book.get('asks', [])
            bids = order_book.get('bids', [])
            logging.info(f"📈 Top 5 Asks: {asks[:5]}")
            logging.info(f"📉 Top 5 Bids: {bids[:5]}")
            if asks:
                return asks[0][0]
            elif bids:
                return bids[0][0]
        except Exception as e:
            logging.error(f"Erro ao obter order book: {e}")
        return None

    async def calculate_entry_amount(self, price_ref: float, capital_amount: float) -> float:
        """
        Calcula a quantidade a ser usada na entrada com base no capital disponível e no preço de referência.

        Args:
            price_ref (float): preço atual de referência do ativo.
            capital_amount (float): valor do capital disponível para trade (já calculado, ex: 1000 USD).

        Returns:
            float: quantidade de contratos ou tokens para a entrada.
        """
        try:
            if price_ref <= 0 or capital_amount <= 0:
                logging.warning(f"🚫 Preço de referência ({price_ref}) ou capital inválido ({capital_amount}).")
                return 0.0

            quantity = capital_amount / price_ref

            # Impede ordens abaixo de $10
            min_order_value = 10
            if quantity * price_ref < min_order_value:
                logging.warning(f"🚫 Ordem abaixo do mínimo de $10: {quantity * price_ref:.2f}")
                return 0.0

            # Opcional: ajuste para múltiplos mínimos
            # min_qty = 0.001
            # quantity = max(min_qty, math.floor(quantity / min_qty) * min_qty)

            return round(quantity, 6)

        except Exception as e:
            logging.error(f"Erro ao calcular quantidade de entrada: {e}")
            return 0.0

    async def place_entry_order(self, symbol: str, leverage: float, entry_amount: float, price_ref: float, side: Signal, sl_price: (float|None) = None, tp_price: (float|None) = None) -> OpenedOrder:

        logging.info(f"🧾 Params finais para create_order: symbol={symbol}, type=market, side={side}, amount={entry_amount}, price={price_ref}")
        try:
            await self.exchange.set_margin_mode("isolated", symbol, {'leverage': leverage})

            params = {}

            # Se SL e TP forem fornecidos, adicionar ao params no formato correto
            if sl_price is not None and tp_price is not None:
                params = {
                    #'marginMode': 'isolated',
                    'stopLoss': {
                        'triggerPrice': sl_price,
                        'price': sl_price,
                        'type': 'market'
                    },
                    'takeProfit': {
                        'triggerPrice': tp_price,
                        'price': tp_price,
                        'type': 'market'
                    },
                    #'reduceOnly': True
                }

            logging.info(f"🧾 Params finais para create_order: symbol={symbol}, type=market, side={side}, amount={entry_amount}, price={price_ref}, params={params}")
    
            logging.info(f"Enviando ordem market ({side}) com params: {params}")
    
            order = await self.exchange.create_order(
                symbol,
                'market',
                side.value,
                entry_amount,
                price_ref,
                params
            )
    
            logging.info(f"✅ Ordem criada: id={order.get('id')}, side={order.get('side')}, amount={order.get('amount')}, price={order.get('price')}")
            
            return OpenedOrder(order.get('id'), None, None, None, symbol, None, order.get('side'), order.get('price'), order.get('amount'), False, None)
    
        except Exception as e:
            logging.error(f"Erro ao criar ordem de entrada: {e}")
            raise

    async def get_entry_price(self, symbol: str) -> float:
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logging.error(f"Erro ao obter preço de entrada: {e}")
            return 0
        
    async def get_total_balance(self):
        try:
            balance = await self.exchange.fetch_balance(params={'user': self.wallet_address})
            total_usdc = balance['total'].get('USDC', 0)
            return float(total_usdc)
        except Exception as e:
            logging.error(f"Erro ao obter saldo total: {e}")
            return 0
    
    async def open_new_position(self, symbol: str, leverage: float, signal: Signal, capital_amount: float, pair: PairConfig, sl: (float|None), tp: (float|None)) -> (OpenedOrder | None):
        price_ref = await self.get_reference_price(symbol)
        if not price_ref or price_ref <= 0:
            raise ValueError("❌ Invalid reference price (None or <= 0)")

        entry_amount = await self.calculate_entry_amount(price_ref, capital_amount)
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

        await self.place_entry_order(symbol, leverage, entry_amount, price_ref, side, sl, tp)

    async def close_position(self, symbol: str, amount: float, side: Signal):
        """
        Fecha posição com ordem de mercado. Usa 'side' atual para calcular o lado oposto (close_side).
        """

        logging.info(f"[DEBUG] Tentando fechar posição: symbol={symbol}, side={side.value}, amount={amount}")

        try:
            orderbook = await self.exchange.fetch_order_book(symbol)

            if side == Signal.BUY:
                price = orderbook['asks'][0][0] if orderbook['asks'] else None
            else:
                price = orderbook['bids'][0][0] if orderbook['bids'] else None

            logging.info(f"[DEBUG] Preço usado para ordem market: {price}")

            if price is None:
                raise Exception("⚠️ Livro de ofertas vazio para fechamento.")

            # Não enviar preço em ordens market (exchange pode rejeitar)
            order = await self.exchange.create_order(
                symbol,
                'market',
                side.value,
                amount,
                price,
                params={'reduceOnly': True}
            )
            logging.info(f"✅ Ordem de fechamento enviada: {order.get('info')}")

        except Exception as e:
            logging.error(f"❌ Erro ao fechar posição: {e}")
            raise

   