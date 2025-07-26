import logging

from commons.enums.signal_enum import Signal
from commons.enums.timeframe_enum import TimeframeEnum
from commons.models.ohlcv_format import OhlcvFormat
from commons.models.open_position import OpenPosition
from commons.utils.config_loader import PairConfig
from commons.utils.ohlcv_wrapper import OhlcvWrapper

from .trading_helpers import TradingHelpers


class ExchangeClient:
    def __init__(self, exchange, wallet_address):
        self.exchange = exchange
        self.wallet_address = wallet_address
        #self.symbol = symbol
        #self.leverage = leverage
        self.helpers = TradingHelpers()

    async def fetch_ohlcv(self, symbol: str, timeframe: TimeframeEnum = TimeframeEnum.M15, limit: int = 14, is_higher: bool = False) -> OhlcvFormat:
        try:
            # Fetch timeframe principal
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe.value, limit)
            
            # Fetch timeframe maior, se necessário
            ohlcv_higher = []
            if is_higher:
                higher_tf = timeframe.get_higher()
                ohlcv_higher = await self.exchange.fetch_ohlcv(symbol, higher_tf.value, limit)

            
            return OhlcvFormat(OhlcvWrapper(ohlcv), OhlcvWrapper(ohlcv_higher))
        
        except Exception as e:
            logging.error(f"[{symbol}] Erro ao buscar os candles ({timeframe.value}): {e}")
            raise

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
                    
                    return OpenPosition(self.helpers.position_side_to_signal_side(pos['side']), size, entry_price, size * entry_price, None, None)

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

    async def place_entry_order(self, symbol: str, leverage: float, entry_amount: float, price_ref: float, side: Signal, sl_price: (float|None) = None, tp_price: (float|None) = None):

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
            return order
    
        except Exception as e:
            logging.error(f"Erro ao criar ordem de entrada: {e}")
    
        return None

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
    
    async def open_new_position(self, symbol: str, leverage: float, signal: Signal, capital_amount: float, pair: PairConfig, sl: (float|None), tp: (float|None)):
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
        #close_side = 'sell' if side == 'buy' else 'buy'

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
    
    async def fetch_closed_orders(self, symbol=None, limit=50):
        try:
            # O símbolo pode ser None para pegar todas
            params = {}
            if symbol:
                params['symbol'] = symbol

            closed_orders = await self.exchange.fetch_closed_orders(symbol=symbol, limit=limit, params=params)
            return closed_orders
        except Exception as e:
            logging.error(f"❌ Erro ao obter ordens fechadas: {e}")
            raise

    async def find_exit_orders_by_entry_id(self, symbol, entry_order_id):
        """
        Filtra as ordens fechadas que são relacionadas à ordem de entrada pelo ID.

        Args:
            closed_orders: lista de ordens fechadas (dicts)
            entry_order_id: id da ordem de entrada (string)

        Returns:
            lista de ordens fechadas relacionadas à entrada (normalmente reduceOnly=True)
        """

        closed_orders = await self.fetch_closed_orders(symbol, 2)

        #print(f"ORDERSSSSSSSSSS {closed_orders}")

        related = []
        for order in closed_orders:
            # Pode ser que o campo esteja em info['order']['oid'] ou order['id']
            order_info = order.get('info', {}).get('order', {})
            parent_id = order_info.get('parentOid')  # só se houver essa relação na exchange
            oid = order_info.get('oid') or order.get('id')

            #print(f"ORDERSSSSSSSSSS {order_info} {parent_id} {oid} {entry_order_id}")

            # Confirma se a ordem fechada está ligada à ordem original
            if oid == entry_order_id or parent_id == entry_order_id:
                related.append(order)
        return related

