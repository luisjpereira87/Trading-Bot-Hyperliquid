import logging

class OrderManager:
    def __init__(self, exchange):
        self.exchange = exchange

    async def create_tp_sl_orders(self, symbol, entry_amount, tp_price, sl_price, side):
        close_side = 'sell' if side == 'buy' else 'buy'

        try:
            tp_order = await self.exchange.create_order(
                symbol,
                'limit',
                close_side,
                entry_amount,
                tp_price,
                params={'reduceOnly': True}
            )
            logging.info("📌 Ordem Take Profit criada:")
            logging.info(tp_order)
        except Exception as e:
            logging.error(f"❌ Erro ao criar ordem TP: {e}")

        try:
            sl_order = await self.exchange.create_order(
                symbol,
                'stop_market',
                close_side,
                entry_amount,
                sl_price,
                params={'stopPrice': sl_price, 'reduceOnly': True}
            )
            logging.info("📌 Ordem Stop Loss criada:")
            logging.info(sl_order)
        except Exception as e:
            logging.error(f"❌ Erro ao criar ordem SL: {e}")

    async def close_position(self, symbol, size, side):
        close_side = 'buy' if side == 'sell' else 'sell'
        try:
            close_order = await self.exchange.create_order(
                symbol,
                'market',
                close_side,
                size,
                None,
                params={'reduceOnly': True}
            )
            logging.info(f"🔁 Posição {side.upper()} encerrada:")
            logging.info(close_order)
        except Exception as e:
            logging.error(f"❌ Erro ao encerrar posição {side.upper()}: {e}")
