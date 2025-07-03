import logging

from .trading_helpers import TradingHelpers


class ExchangeClient:
    def __init__(self, exchange, wallet_address, symbol, leverage):
        self.exchange = exchange
        self.wallet_address = wallet_address
        self.symbol = symbol
        self.leverage = leverage
        self.helpers = TradingHelpers()

    async def print_balance(self):
        try:
            balance = await self.exchange.fetch_balance(params={'user': self.wallet_address})
            logging.info(f"💰 Saldo total: {balance['total']}")
        except Exception as e:
            logging.error(f"Erro ao buscar saldo: {e}")

    async def print_open_orders(self, symbol=None):
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

    async def cancel_all_orders(self, symbol=None):
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

    async def get_open_position(self, symbol=None):
        try:
            positions = await self.exchange.fetch_positions(params={'user': self.wallet_address})
            for pos in positions:
                if pos["symbol"] == symbol and float(pos.get('contracts', 0)) > 0:
                    size = float(pos['contracts'])
                    entry_price = pos.get('entryPrice') or pos.get('entry_price') or pos.get('averagePrice') or 0.0
                    return {
                        'side': self.helpers.position_side_to_signal_side(pos['side']),
                        'size': size,
                        'entryPrice': entry_price,
                        'notional': size * entry_price
                    }
        except Exception as e:
            logging.error(f"Erro ao obter posições abertas: {e}")
        return None

    async def get_reference_price(self):
        try:
            order_book = await self.exchange.fetch_order_book(self.symbol)
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

    async def place_entry_order(self, entry_amount, price_ref, side, sl_price=None, tp_price=None):
        try:
            await self.exchange.set_margin_mode("isolated", self.symbol, {'leverage': self.leverage})

            #params = {'marginMode': 'isolated'}

            # Se SL e TP forem fornecidos, adicionar ao params no formato correto
            if sl_price is not None and tp_price is not None:
                params = {
                    'marginMode': 'isolated',
                    'stopLoss': {
                        'triggerPrice': sl_price,
                        'price': sl_price
                    },
                    'takeProfit': {
                        'triggerPrice': tp_price,
                        'price': tp_price
                    },
                    'reduceOnly': True
                }

            logging.info(f"Enviando ordem market ({side}) com params: {params}")

            order = await self.exchange.create_order(
                self.symbol,
                'market',  # tipo pode ser 'market' ou 'limit' conforme sua estratégia
                side,
                entry_amount,
                price_ref,
                params
            )

            logging.info(f"✅ Ordem criada: id={order.get('id')}, side={order.get('side')}, amount={order.get('amount')}, price={order.get('price')}")
            return order

        except Exception as e:
            logging.error(f"Erro ao criar ordem de entrada: {e}")

        return None

    async def get_entry_price(self):
        try:
            ticker = await self.exchange.fetch_ticker(self.symbol)
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

