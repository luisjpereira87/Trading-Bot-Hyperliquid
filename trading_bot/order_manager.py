import logging

class OrderManager:
    def __init__(self, exchange):
        self.exchange = exchange

    async def create_tp_sl_orders(self, symbol, amount, tp_price, sl_price, side):
        try:
            if amount <= 0:
                logging.warning(f"🚫 Quantidade inválida para TP/SL: {amount}")
                return

            if sl_price is None:
                logging.error("❌ sl_price está como None, não é possível criar SL.")
                return

            # Busca posições abertas para o símbolo
            positions = await self.exchange.fetch_positions()
            position = next((p for p in positions if p['symbol'] == symbol and float(p['contracts']) > 0), None)

            if not position:
                logging.info(f"🚫 Nenhuma posição aberta para {symbol}. TP/SL não será criado.")
                return

            pos_amount = float(position['contracts'])
            pos_side = position['side']  # 'long' ou 'short'
            logging.info(f"📌 Posição atual: lado={pos_side}, quantidade={pos_amount}")
            logging.info(f"📌 Quantidade solicitada para TP/SL: {amount}")
            logging.info(f"📌 Lado da ordem TP/SL (deve ser oposto à posição): {side}")

            # Ajusta a quantidade para no máximo o tamanho da posição
            amount = min(amount, pos_amount)

            # Cancela ordens abertas anteriores
            open_orders = await self.exchange.fetch_open_orders(symbol)
            for order in open_orders:
                try:
                    await self.exchange.cancel_order(order['id'], symbol)
                    logging.info(f"🗑️ Ordem cancelada: {order['id']}")
                except Exception as e:
                    logging.warning(f"⚠️ Falha ao cancelar ordem {order['id']}: {e}")

            # Cria ordem Take Profit (limit)
            tp_order = await self.exchange.create_order(
                symbol,
                'limit',
                side,
                amount,
                tp_price,
                params={'reduceOnly': True}
            )
            logging.info(f"🎯 Ordem Take Profit criada: {tp_order.get('id', 'sem id')}, detalhes: {tp_order.get('info')}")
            logging.info(tp_order)

            # Cria ordem Stop Loss (stop_market)
            sl_order = await self.exchange.create_order(
                symbol,
                'stop_market',
                side,
                amount,
                sl_price,
                params={'stopPrice': sl_price, 'reduceOnly': True}
            )
            logging.info(f"🛑 Ordem Stop Loss criada: {sl_order.get('id', 'sem id')}, detalhes: {sl_order.get('info')}")
            logging.info(sl_order)

        except Exception as e:
            logging.error(f"❌ Erro ao criar ordens TP/SL: {e}")

    async def close_position(self, symbol, amount, side):
        """
        Fecha uma posição aberta com ordem de mercado no lado oposto.
        """
        close_side = 'sell' if side == 'buy' else 'buy'

        try:
            # Busca o livro de ofertas para referência de preço
            orderbook = await self.exchange.fetch_order_book(symbol)
            price = None
            if close_side == 'buy':
                price = orderbook['asks'][0][0] if orderbook['asks'] else None
            else:
                price = orderbook['bids'][0][0] if orderbook['bids'] else None

            if price is None:
                raise Exception("❌ Não foi possível obter preço de referência do livro para fechar posição.")

            params = {
                'reduceOnly': True,
                'price': price,  # Usado internamente para cálculo de slippage
            }

            # Envia ordem de mercado
            order = await self.exchange.create_order(
                symbol,
                'market',
                close_side,
                amount,
                None,
                params
            )
            logging.info(f"✅ Ordem de fechamento de posição enviada: {order}")
        except Exception as e:
            logging.error(f"❌ Erro ao fechar posição: {e}")







