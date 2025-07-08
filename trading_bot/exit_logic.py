import logging

from enums.signal_enum import Signal
from strategies.signal_result import SignalResult
from utils.config_loader import PairConfig


class ExitLogic:
    def __init__(self, helpers, order_manager):
        self.helpers = helpers
        self.order_manager = order_manager
        self.hold_counters: dict[str, int] = {}
        self.last_profits: dict[str, float] = {}
        self.trailing_stops: dict[str, float] = {}  # Armazena o preço do trailing stop por símbolo

    async def should_exit(
        self,
        exchange_client,
        pair: PairConfig,
        signal_result:SignalResult,
        position,
        atr_now: float,
    ) -> bool:
        symbol = pair.symbol
        entry_price = position["entryPrice"]
        side = position["side"]
        size = float(position["size"])
        notional = float(position["notional"])

        # Pega o preço atual (mark price)
        ticker = await exchange_client.fetch_ticker(symbol)
        mark_price = ticker.get("last") or ticker.get("close")
        if mark_price is None:
            return False

        # Calcula lucro absoluto e percentual
        if side == "buy":
            profit_pct = (mark_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - mark_price) / entry_price

        profit_abs = profit_pct * notional
        self.last_profits[symbol] = profit_abs

        min_profit_abs = getattr(pair, "min_profit_abs", 5.0)
        min_profit_pct = getattr(pair, "min_profit_pct", 0.005)  # 0.5% default

        logging.info(
            f"📊 {symbol} | Profit = {profit_abs:.2f} USDC ({profit_pct*100:.2f}%), Min $ = {min_profit_abs:.2f}, Min % = {min_profit_pct*100:.2f}%"
        )

        # --- TRAILING STOP DINÂMICO ---

        trailing_stop_multiplier = getattr(pair, "trailing_stop_multiplier", 1.5)
        trailing_stop_distance = atr_now * trailing_stop_multiplier

        prev_trailing_stop = self.trailing_stops.get(symbol)

        if side == "buy":
            # Atualiza trailing stop para preço máximo favorável menos distância do trailing
            new_trailing_stop = max(prev_trailing_stop or entry_price, mark_price - trailing_stop_distance)
            self.trailing_stops[symbol] = new_trailing_stop

            # Se preço atual cair abaixo do trailing stop, fechar posição
            if mark_price <= new_trailing_stop:
                logging.info(f"🚪 Trailing stop hit for {symbol} at {mark_price:.4f} (stop: {new_trailing_stop:.4f})")
                await self.order_manager.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
                self.trailing_stops.pop(symbol, None)
                return True

        else:  # side == "sell"
            # Atualiza trailing stop para preço mínimo favorável mais distância do trailing
            new_trailing_stop = min(prev_trailing_stop or entry_price, mark_price + trailing_stop_distance)
            self.trailing_stops[symbol] = new_trailing_stop

            # Se preço atual subir acima do trailing stop, fechar posição
            if mark_price >= new_trailing_stop:
                logging.info(f"🚪 Trailing stop hit for {symbol} at {mark_price:.4f} (stop: {new_trailing_stop:.4f})")
                await self.order_manager.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
                self.trailing_stops.pop(symbol, None)
                return True

        # --- FECHAR SE LUCRO MÍNIMO ATINGIDO (condição existente) ---
        if profit_abs >= min_profit_abs and profit_pct >= min_profit_pct:
            logging.info(f"💰 Exiting {symbol} with profit")
            await self.order_manager.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
            self.trailing_stops.pop(symbol, None)
            return True

        # --- HOLD persistente ---
        if signal_result.signal == Signal.HOLD and signal_result.confidence is not None and signal_result.confidence >= 0.9:
            self.hold_counters[symbol] = self.hold_counters.get(symbol, 0) + 1
            logging.info(f"⏸️ HOLD detected for {symbol} ({self.hold_counters[symbol]} times)")

            if self.hold_counters[symbol] >= getattr(pair, "max_hold_candles", 2):
                logging.info(f"🔚 Exiting {symbol} due to persistent HOLD signal")
                await self.order_manager.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
                self.hold_counters[symbol] = 0
                self.trailing_stops.pop(symbol, None)
                return True
        else:
            self.hold_counters[symbol] = 0

        # --- Reversão de lucro (ajustar condição para fechar mais rápido) ---
        last_profit = self.last_profits.get(symbol)
        if last_profit and last_profit >= min_profit_abs and profit_abs < last_profit * 0.75:
            logging.info(
                f"⚠️ Profit reversal detected for {symbol} (from {last_profit:.2f} to {profit_abs:.2f}). Closing position."
            )
            await self.order_manager.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
            self.last_profits[symbol] = 0
            self.trailing_stops.pop(symbol, None)
            return True

        return False

