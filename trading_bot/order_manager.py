import logging

class OrderManager:
    def __init__(self, exchange):
        self.exchange = exchange

    async def create_tp_sl_orders(self, symbol, amount, tp_price, sl_price, side):
        try:
            # Busca posições abertas para o símbolo
            positions = await self.exchange.fetch_positions()
            position = next((p for p in positions if p['symbol'] == symbol and float(p['contracts']) > 0), None)

            if not position:
                logging.info(f"🚫 Nenhuma posição aberta para {symbol}. TP/SL não será criado.")
                return

            pos_amount = float(position['contracts'])
            pos_side = position['side']  # 'long' ou 'short'
            logging.info(f"Posição atual: lado={pos_side}, quantidade={pos_amount}")
            logging.info(f"Quantidade recebida para TP/SL: {amount}")
            logging.info(f"Lado da ordem TP/SL (deve ser lado oposto da posição): {side}")

            # Ajustar quantidade para no máximo o tamanho da posição
            amount = min(amount, pos_amount)

            # Cancela ordens abertas anteriores para o símbolo
            open_orders = await self.exchange.fetch_open_orders(symbol)
            for order in open_orders:
                try:
                    await self.exchange.cancel_order(order['id'], symbol)
                    logging.info(f"🗑️ Ordem cancelada: {order['id']}")
                except Exception as e:
                    logging.warning(f"⚠️ Falha ao cancelar ordem {order['id']}: {e}")

            # Cria ordem Take Profit (limit) com reduceOnly=True
            tp_order = await self.exchange.create_order(
                symbol,
                'limit',
                side,         # lado oposto da posição
                amount,
                tp_price,
                params={'reduceOnly': True}
            )
            logging.info(f"📌 Ordem Take Profit criada: {symbol, side, amount,tp_price}, raw info: {tp_order.get('info')}")
            #logging.info(f"📌 Ordem Take Profit criada: {tp_order.get('id', 'sem id')}, raw info: {tp_order.get('info')}")
            logging.info(tp_order)

            # Cria ordem Stop Loss (stop_market) com reduceOnly=True
            logging.info(f"📌 Ordem Stop Loss criada: {symbol, side, amount}, raw info: {tp_order.get('info')}")
            sl_order = await self.exchange.create_order(
                symbol,
                'stop_market',
                side,
                amount,
                sl_price,  # ← aqui estava o erro! Deixa de ser None
                params={'stopPrice': sl_price, 'reduceOnly': True}
            )
            
            logging.info(f"📌 Ordem Stop Loss criada: {tp_order.get('id', 'sem id')}, raw info: {tp_order.get('info')}")
            logging.info(sl_order)

        except Exception as e:
            logging.error(f"❌ Erro ao criar ordens TP/SL: {e}")

    async def close_position(self, symbol, amount, side):
        """
        Fecha a posição aberta enviando ordem de mercado no lado oposto.
        side: lado atual da posição ('buy' ou 'sell'), então precisa da direção oposta para fechar.
        """
        close_side = 'sell' if side == 'buy' else 'buy'

        try:
            # Busca o livro de ofertas para pegar preço de referência
            orderbook = await self.exchange.fetch_order_book(symbol)
            if close_side == 'buy':
                price = orderbook['asks'][0][0] if orderbook['asks'] else None
            else:
                price = orderbook['bids'][0][0] if orderbook['bids'] else None

            if price is None:
                raise Exception("Não foi possível obter preço de referência do livro para fechar posição.")

            params = {
                'reduceOnly': True,
                'price': price,  # preço usado para cálculo interno de slippage
            }

            # Envia ordem market com price None, pois preço vai em params
            order = await self.exchange.create_order(
                symbol,
                'market',
                close_side,
                amount,
                None,
                params
            )
            logging.info(f"Ordem para fechar posição enviada: {order}")
        except Exception as e:
            logging.error(f"Erro ao fechar posição: {e}")





