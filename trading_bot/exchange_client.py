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
            # Quantidade = capital dividido pelo preço de referência, ajustando para o tamanho do contrato se necessário
            # Se seu contrato for 1:1, esse cálculo serve. Ajuste se seu mercado usar multiplicadores diferentes.
            quantity = capital_amount / price_ref

            # Se quiser ajustar a quantidade para o mínimo aceito ou múltiplos mínimos, faça aqui
            # Exemplo:
            # min_qty = 0.001
            # quantity = max(min_qty, math.floor(quantity / min_qty) * min_qty)

            return quantity

        except Exception as e:
            logging.error(f"Erro ao calcular quantidade de entrada: {e}")
            return 0.0

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
        
    async def get_total_balance(self):
        try:
            balance = await self.exchange.fetch_balance(params={'user': self.wallet_address})
            total_usdc = balance['total'].get('USDC', 0)
            return float(total_usdc)
        except Exception as e:
            logging.error(f"Erro ao obter saldo total: {e}")
            return 0

