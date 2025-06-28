import logging

class OrderManager:
    def __init__(self, exchange):
        self.exchange = exchange

    async def create_tp_sl_orders(self, symbol, amount, tp_price, sl_price, side_signal):
        try:
            positions = await self.exchange.fetch_positions()
            position = next((p for p in positions if p['symbol'] == symbol and float(p['contracts']) > 0), None)

            if not position:
                logging.info(f"🚫 Nenhuma posição aberta para {symbol}. TP/SL não será criado.")
                return

            pos_amount = float(position['contracts'])
            pos_side = position['side']  # 'long' ou 'short'

            # Define o lado oposto da posição para TP/SL
            close_side = 'sell' if pos_side == 'long' else 'buy'

            logging.info(f"🎯 Criando TP/SL para {symbol}: lado da posição={pos_side}, lado TP/SL={close_side}")
            logging.info(f"   Quantidade posição={pos_amount} | Quantidade recebida={amount}")

            amount = min(amount, pos_amount)

            # Cancelar ordens abertas anteriores
            open_orders = await self.exchange.fetch_open_orders(symbol)
            for order in open_orders:
                try:
                    await self.exchange.cancel_order(order['id'], symbol)
                    logging.info(f"🗑️ Ordem cancelada: {order['id']}")
                except Exception as e:
                    logging.warning(f"⚠️ Falha ao cancelar ordem {order['id']}: {e}")

            # Criar ordem de Take Profit (limit)
            tp_order = await self.exchange.create_order(
                symbol,
                'limit',
                close_side,
                amount,
                tp_price,
                params={'reduceOnly': True}
            )
            logging.info(f"✅ TP criado: {tp_order.get('info')}")

            # Criar ordem de Stop Loss (stop_market)
            sl_order = await self.exchange.create_order(
                symbol,
                'stop_market',
                close_side,
                amount,
                sl_price,
                params={'stopPrice': sl_price, 'reduceOnly': True}
            )
            logging.info(f"✅ SL criado: {sl_order.get('info')}")

        except Exception as e:
            logging.error(f"❌ Erro ao criar ordens TP/SL: {e}")

    async def close_position(self, symbol, amount, side):
        """
        Fecha posição com ordem de mercado. Usa 'side' atual para calcular o lado oposto (close_side).
        """
        close_side = 'sell' if side == 'buy' else 'buy'

        try:
            orderbook = await self.exchange.fetch_order_book(symbol)

            if close_side == 'buy':
                price = orderbook['asks'][0][0] if orderbook['asks'] else None
            else:
                price = orderbook['bids'][0][0] if orderbook['bids'] else None

            if price is None:
                raise Exception("⚠️ Livro de ofertas vazio para fechamento.")

            order = await self.exchange.create_order(
                symbol,
                'market',
                close_side,
                amount,
                price,  # Agora incluído corretamente!
                params={'reduceOnly': True}
            )
            logging.info(f"✅ Ordem de fechamento enviada: {order.get('info')}")

        except Exception as e:
            logging.error(f"❌ Erro ao fechar posição: {e}")







