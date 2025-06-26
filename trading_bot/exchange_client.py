import logging

class ExchangeClient:
    def __init__(self, exchange, wallet_address, symbol, leverage):
        self.exchange = exchange
        self.wallet_address = wallet_address
        self.symbol = symbol
        self.leverage = leverage

    async def print_balance(self):
        try:
            balance = await self.exchange.fetch_balance(params={'user': self.wallet_address})
            logging.info(f"💰 Saldo total: {balance['total']}")
        except Exception as e:
            logging.error(f"Erro ao buscar saldo: {e}")

    async def print_open_orders(self):
        try:
            open_orders = await self.exchange.fetch_open_orders(params={'user': self.wallet_address})
            logging.info(f"📘 Ordens abertas ({len(open_orders)}):")
            for order in open_orders:
                logging.info(order)
        except Exception as e:
            logging.error(f"Erro ao buscar ordens abertas: {e}")

    async def cancel_all_orders(self):
        try:
            open_orders = await self.exchange.fetch_open_orders(params={'user': self.wallet_address})
            for order in open_orders:
                await self.exchange.cancel_order(order['id'], self.symbol)
            logging.info("🔁 Todas as ordens foram canceladas.")
        except Exception as e:
            logging.error(f"Erro ao cancelar ordens: {e}")

    async def get_open_position(self):
        try:
            positions = await self.exchange.fetch_positions(params={'user': self.wallet_address})
            for pos in positions:
                if float(pos.get('contracts', 0)) > 0:
                    return {
                        'side': pos['side'],
                        'size': float(pos['contracts'])
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

    async def calculate_entry_amount(self, price_ref):
        try:
            balance = await self.exchange.fetch_balance(params={'user': self.wallet_address})
            balance_usdc = balance['free']['USDC']
            if balance_usdc > 0 and price_ref:
                return balance_usdc / (price_ref * self.leverage)
            else:
                logging.warning("Saldo USDC insuficiente ou preço inválido.")
        except Exception as e:
            logging.error(f"Erro ao calcular quantidade de entrada: {e}")
        return 0

    async def place_entry_order(self, entry_amount, price_ref, side):
        try:
            await self.exchange.set_margin_mode("isolated", self.symbol, {'leverage': self.leverage})
            params = {'marginMode': 'isolated'}
            logging.info(f"Enviando ordem market ({side}) com params: {params}")
            order = await self.exchange.create_order(self.symbol, 'market', side, entry_amount, price_ref, params)
            logging.info(f"✅ Ordem criada: {order}")
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

