import logging

from commons.enums.signal_enum import Signal
from commons.models.signal_result import SignalResult
from commons.utils.config_loader import PairConfig


class ExitLogic:
    def __init__(self, helpers, exchange_client):
        self.helpers = helpers
        self.exchange_client = exchange_client
        self.hold_counters: dict[str, int] = {}
        self.last_profits: dict[str, float] = {}
        self.trailing_stops: dict[str, float] = {}
        self.partial_taken: dict[str, bool] = {}  # para controle de parcial por símbolo

    async def should_exit(
        self,
        pair: PairConfig,
        signal_result: SignalResult,
        position,
        atr_now: float,
    ) -> bool:
        symbol = pair.symbol
        entry_price = position["entryPrice"]
        side = position["side"]
        size = float(position["size"])
        notional = float(position["notional"])

        ticker = await self.exchange_client.fetch_ticker(symbol)
        mark_price = ticker.get("last") or ticker.get("close")
        if mark_price is None:
            return False

        # Lucro atual
        if side == "buy":
            profit_pct = (mark_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - mark_price) / entry_price

        profit_abs = profit_pct * notional
        self.last_profits[symbol] = profit_abs

        min_profit_abs = getattr(pair, "min_profit_abs", 5.0)
        min_profit_pct = getattr(pair, "min_profit_pct", 0.005)

        logging.info(
            f"📊 {symbol} | Profit = {profit_abs:.2f} USDC ({profit_pct*100:.2f}%), Min $ = {min_profit_abs:.2f}, Min % = {min_profit_pct*100:.2f}%"
        )

        # --- STOP LOSS por ATR ---
        sl_atr_multiplier = getattr(pair, "stop_loss_atr_multiplier", 1.5)
        stop_loss_price = entry_price - atr_now * sl_atr_multiplier if side == "buy" else entry_price + atr_now * sl_atr_multiplier

        if (side == "buy" and mark_price <= stop_loss_price) or (side == "sell" and mark_price >= stop_loss_price):
            logging.warning(f"🛑 Stop loss hit for {symbol} at {mark_price:.4f}")
            await self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
            self._reset_state(symbol)
            return True

        # --- FECHAMENTO PARCIAL EM LUCRO ALTO ---
        partial_take_profit_pct = getattr(pair, "partial_take_profit_pct", 0.03)  # 3%
        partial_close_pct = getattr(pair, "partial_close_pct", 0.5)  # fecha 50%

        if profit_pct > partial_take_profit_pct and not self.partial_taken.get(symbol, False):
            partial_size = size * partial_close_pct
            logging.info(f"✂️ Realizing partial profit on {symbol}: closing {partial_close_pct*100:.0f}%")
            await self.exchange_client.close_position(symbol, partial_size, self.helpers.get_opposite_side(Signal.from_str(side)))
            self.partial_taken[symbol] = True
            return False  # mantém o restante aberto com trailing stop

        # --- TRAILING STOP ---
        trailing_stop_multiplier = getattr(pair, "trailing_stop_multiplier", 1.5)
        if profit_pct > 0.05:
            trailing_stop_multiplier *= 0.5

        trailing_stop_distance = atr_now * trailing_stop_multiplier
        prev_trailing_stop = self.trailing_stops.get(symbol)

        if side == "buy":
            new_trailing_stop = max(prev_trailing_stop or entry_price, mark_price - trailing_stop_distance)
            self.trailing_stops[symbol] = new_trailing_stop
            if mark_price <= new_trailing_stop:
                logging.info(f"🚪 Trailing stop hit for {symbol} at {mark_price:.4f}")
                await self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
                self._reset_state(symbol)
                return True
        else:
            new_trailing_stop = min(prev_trailing_stop or entry_price, mark_price + trailing_stop_distance)
            self.trailing_stops[symbol] = new_trailing_stop
            if mark_price >= new_trailing_stop:
                logging.info(f"🚪 Trailing stop hit for {symbol} at {mark_price:.4f}")
                await self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
                self._reset_state(symbol)
                return True

        # --- LUCRO MÍNIMO ALCANÇADO ---
        if profit_abs >= min_profit_abs and profit_pct >= min_profit_pct:
            logging.info(f"💰 Exiting {symbol} with profit")
            await self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
            self._reset_state(symbol)
            return True

        # --- HOLD persistente + tendência contrária ---
        if signal_result.signal == Signal.HOLD and signal_result.confidence and signal_result.confidence >= 0.9:
            trend_against = self.is_trend_against_position(symbol, side)
            self.hold_counters[symbol] = self.hold_counters.get(symbol, 0) + 1

            if trend_against:
                logging.info(f"📉 Trend against {symbol}. Exiting on HOLD.")
                await self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
                self._reset_state(symbol)
                return True

            if self.hold_counters[symbol] >= getattr(pair, "max_hold_candles", 2):
                logging.info(f"🔚 HOLD max reached for {symbol}")
                await self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
                self._reset_state(symbol)
                return True
        else:
            self.hold_counters[symbol] = 0

        # --- Reversão de lucro ---
        last_profit = self.last_profits.get(symbol)
        if last_profit and last_profit >= min_profit_abs and profit_abs < last_profit * 0.75:
            logging.info(f"⚠️ Profit reversal detected for {symbol}. Closing position.")
            await self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
            self._reset_state(symbol)
            return True

        return False

    def _reset_state(self, symbol: str):
        self.trailing_stops.pop(symbol, None)
        self.hold_counters[symbol] = 0
        self.last_profits[symbol] = 0
        self.partial_taken[symbol] = False

    def is_trend_against_position(self, symbol: str, side: str) -> bool:
        df = self.helpers.get_candles(symbol)
        if df is None or len(df) < 21:
            return False

        ema_fast = df["close"].ewm(span=9).mean().iloc[-1]
        ema_slow = df["close"].ewm(span=21).mean().iloc[-1]

        if side == "buy" and ema_fast < ema_slow:
            return True
        if side == "sell" and ema_fast > ema_slow:
            return True
        return False


