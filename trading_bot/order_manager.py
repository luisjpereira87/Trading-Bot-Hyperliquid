import logging


class OrderManager:
    def __init__(self, exchange, min_order_size=0.01):
        self.exchange = exchange
        self.min_order_size = min_order_size

    async def create_tp_sl_orders(self, symbol, amount, tp_price, sl_price, side_signal):
        try:
            positions = await self.exchange.fetch_positions()
            logging.info(f"[DEBUG] Posições abertas: {positions}")

            # Buscar posição aberta para o símbolo e que tenha contratos > 0
            position = next((p for p in positions if p['symbol'] == symbol and abs(float(p['contracts'])) > 0), None)

            if not position:
                logging.info(f"🚫 Nenhuma posição aberta para {symbol}. TP/SL não será criado.")
                return

            pos_amount = abs(float(position['contracts']))
            pos_side = position['side']  # 'long' ou 'short'

            logging.info(f"[DEBUG] Posição encontrada: side={pos_side}, quantidade={pos_amount}")

            # Define o lado oposto da posição para TP/SL
            close_side = 'sell' if pos_side == 'long' else 'buy'

            # Garantir que o amount não ultrapasse o tamanho da posição
            amount_to_use = min(amount, pos_amount)

            # Buscar preço de mercado
            ticker = await self.exchange.fetch_ticker(symbol)
            mark_price = ticker['last']  # Pode usar 'mark' se preferir

            order_value_usdc = amount_to_use * mark_price
            min_usdc_value = 5  # Ajuste conforme necessário para cada par

            logging.info(f"📏 Quantidade final para TP/SL: {amount:.6f}")

            # ⚠️ Verificação do mínimo
            if order_value_usdc < min_usdc_value:
                logging.warning(f"🚫 Valor total da ordem ({order_value_usdc:.2f} USDC) abaixo do mínimo permitido ({min_usdc_value} USDC) para {symbol}. TP/SL não será criado.")
                return

            logging.info(f"🎯 Criando TP/SL para {symbol}: lado da posição={pos_side}, lado TP/SL={close_side}")
            logging.info(f"   Quantidade posição={pos_amount} | Quantidade usada={amount_to_use}")
            logging.info(f"   TP: {tp_price} | SL: {sl_price}")

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
                amount_to_use,
                tp_price,
                params={'reduceOnly': True}
            )
            logging.info(f"✅ TP criado: {tp_order.get('info')}")

            # Criar ordem de Stop Loss (stop_market)
            sl_order = await self.exchange.create_order(
                symbol,
                'stop_market',
                close_side,
                amount_to_use,
                sl_price,  # preço None para stop_market
                params={'stopPrice': sl_price, 'reduceOnly': True}
            )
            logging.info(f"✅ SL criado: {sl_order.get('info')}")

        except Exception as e:
            logging.error(f"❌ Erro ao criar ordens TP/SL: {e}")
            raise

    async def close_position(self, symbol, amount, side):
        """
        Fecha posição com ordem de mercado. Usa 'side' atual para calcular o lado oposto (close_side).
        """
        close_side = 'sell' if side == 'buy' else 'buy'

        logging.info(f"[DEBUG] Tentando fechar posição: symbol={symbol}, side={side}, close_side={close_side}, amount={amount}")

        try:
            orderbook = await self.exchange.fetch_order_book(symbol)

            if close_side == 'buy':
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
                close_side,
                amount,
                price,
                params={'reduceOnly': True}
            )
            logging.info(f"✅ Ordem de fechamento enviada: {order.get('info')}")

        except Exception as e:
            logging.error(f"❌ Erro ao fechar posição: {e}")








