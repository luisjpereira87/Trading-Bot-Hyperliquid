import logging


class OrderManager:
    def __init__(self, exchange, min_order_size=0.01):
        self.exchange = exchange
        self.min_order_size = min_order_size
        self.tp_pct = 0.03
        self.sl_pct = 0.01
        self.tp_factor = 2

    async def manage_tp_sl(self, exchange_client, position, signal, symbol, atr_now):
        if not position:
            logging.info(f"⚠️ Nenhuma posição aberta para {symbol}. Ignorando TP/SL.")
            return

        entry_price = position.get("entryPrice")
        if entry_price is None:
            entry_price = await exchange_client.get_entry_price()

        entry_amount = float(position["size"])
        side = "buy" if position["side"] == "long" else "sell"

        if atr_now is None:
            logging.warning(f"❌ ATR não pôde ser calculado para {symbol}. Ignorando.")
            return

        sl_price, tp_price = self.calculate_sl_tp(entry_price, side, atr_now, signal["mode"])

        logging.info(f"\n{symbol} 🎯 TP dinâmico: {tp_price} | 🛑 SL dinâmico: {sl_price}")

        close_side = "sell" if side == "buy" else "buy"

        # --- NOVA VALIDAÇÃO DE SL/TP ---
        # Define percentuais mínimos e máximos para SL/TP em relação ao preço de entrada
        min_pct_distance = 0.002  # 0.2%
        max_pct_distance = 0.1    # 10%

        def pct_diff(price1, price2):
            return abs(price1 - price2) / price1

        sl_distance = pct_diff(entry_price, sl_price)
        tp_distance = pct_diff(entry_price, tp_price)

        if sl_distance < min_pct_distance:
            logging.warning(f"🚫 SL muito próximo do preço de entrada para {symbol}. Distância: {sl_distance:.4%}. Ignorando.")
            return

        if tp_distance < min_pct_distance:
            logging.warning(f"🚫 TP muito próximo do preço de entrada para {symbol}. Distância: {tp_distance:.4%}. Ignorando.")
            return

        if sl_distance > max_pct_distance:
            logging.warning(f"🚫 SL muito distante do preço de entrada para {symbol}. Distância: {sl_distance:.4%}. Ignorando.")
            return

        if tp_distance > max_pct_distance:
            logging.warning(f"🚫 TP muito distante do preço de entrada para {symbol}. Distância: {tp_distance:.4%}. Ignorando.")
            return
        # --- FIM DA VALIDAÇÃO ADICIONAL ---

        # Validar valores antes de enviar ordens
        min_order_value = 10  # dólares
        if entry_amount * tp_price < min_order_value or entry_amount * sl_price < min_order_value:
            logging.warning(f"🚫 Ordem TP/SL abaixo do mínimo para {symbol}. TP: {entry_amount * tp_price}, SL: {entry_amount * sl_price}")
            return

        try:
            await self.create_tp_sl_orders(symbol, entry_amount, tp_price, sl_price, close_side)
        except Exception:
            logging.exception("❌ Falha ao criar TP/SL. Fechando posição por segurança.")
            await self.close_position(symbol, entry_amount, close_side)

    def calculate_sl_tp(self, entry_price, side, atr_now, mode="normal"):
        """
        Calcula SL e TP dinâmicos com base no ATR, percentual e valor absoluto.
        Ajusta parâmetros dinamicamente para diferentes faixas de preço e modos.
        """

        # Ajusta o sl_pct e tp_factor dinamicamente com base no preço do ativo
        if entry_price < 50:  # tokens baratos
            sl_pct_base = 0.01
            tp_factor_base = 2.0
        elif entry_price < 500:  # tokens médios (ex: LINK, DOGE)
            sl_pct_base = 0.008
            tp_factor_base = 2.2
        elif entry_price < 5000:  # ex: ETH
            sl_pct_base = 0.006
            tp_factor_base = 2.5
        else:  # BTC e similares
            sl_pct_base = 0.004
            tp_factor_base = 3.0

        # Modificadores por modo
        if mode == "aggressive":
            sl_pct = sl_pct_base * 0.8
            tp_factor = tp_factor_base * 1.2
        elif mode == "conservative":
            sl_pct = sl_pct_base * 1.5
            tp_factor = tp_factor_base * 0.9
        else:  # normal
            sl_pct = sl_pct_base
            tp_factor = tp_factor_base

        if sl_pct > 1:
            logging.warning(f"⚠️ sl_pct parece estar em valor absoluto ({sl_pct}). Esperado valor entre 0 e 1.")

        sl_min_dist = sl_pct * entry_price
        atr_cap = entry_price * 0.015  # reduzido para no máximo 1.5%
        atr_now = min(atr_now, atr_cap)

        sl_distance = max(sl_min_dist, atr_now)

        logging.info(f"🔎 Cálculo TP/SL ({mode}):")
        logging.info(f"📈 Preço de entrada: {entry_price}")
        logging.info(f"📊 ATR atual (limitado): {atr_now:.4f}")
        logging.info(f"🧮 Distância mínima SL (%): {sl_min_dist:.4f}")
        logging.info(f"🧮 Distância usada SL: {sl_distance:.4f}")

        if side == "buy":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_factor * sl_distance
        else:  # sell
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_factor * sl_distance

        # Validação para evitar extremos
        sl_pct_off = abs((sl_price - entry_price) / entry_price)
        tp_pct_off = abs((tp_price - entry_price) / entry_price)

        if sl_pct_off > 0.10 or tp_pct_off > 0.25:  # limite mais razoável
            logging.warning(f"🚫 SL ou TP fora de range aceitável. SL: {sl_price}, TP: {tp_price}")
            raise ValueError("SL ou TP calculado está fora do intervalo aceitável.")

        logging.info(f"✅ SL final: {sl_price:.2f} ({sl_pct_off*100:.2f}%)")
        logging.info(f"✅ TP final: {tp_price:.2f} ({tp_pct_off*100:.2f}%)")

        return round(sl_price, 2), round(tp_price, 2)


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

            logging.info(f"📏 Quantidade final para TP/SL: {amount_to_use:.6f}")

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
        #close_side = 'sell' if side == 'buy' else 'buy'

        logging.info(f"[DEBUG] Tentando fechar posição: symbol={symbol}, side={side}, amount={amount}")

        try:
            orderbook = await self.exchange.fetch_order_book(symbol)

            if side == 'buy':
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
                side,
                amount,
                price,
                params={'reduceOnly': True}
            )
            logging.info(f"✅ Ordem de fechamento enviada: {order.get('info')}")

        except Exception as e:
            logging.error(f"❌ Erro ao fechar posição: {e}")
            raise








