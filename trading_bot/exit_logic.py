import logging

from commons.enums.signal_enum import Signal
from commons.models.open_position import OpenPosition
from commons.models.signal_result import SignalResult
from commons.utils.config_loader import PairConfig
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from strategies.indicators import Indicators
from strategies.strategy_utils import StrategyUtils


class ExitLogic:
    def __init__(self, helpers, exchange_client):
        self.helpers = helpers
        self.exchange_client = exchange_client
        self.hold_counters: dict[str, int] = {}
        self.last_profits: dict[str, float] = {}
        self.trailing_stops: dict[str, float] = {}
        self.partial_taken: dict[str, bool] = {}

    async def should_exit(
        self,
        ohlcv: OhlcvWrapper,
        pair: PairConfig,
        signal_result: SignalResult,
        position: OpenPosition,
    ) -> bool:
        symbol = pair.symbol
        mark_price = await self._get_market_price(symbol)
        if mark_price is None:
            return False

        entry_price = position.entry_price
        side = position.side
        size = float(position.size)
        notional = float(position.notional)

        self.ohlcv = ohlcv
        self.pair = pair

        if side is None:
            return False
        
        indicators = Indicators(self.ohlcv)
        atr = indicators.atr()
        atr_now = atr[-1]

        profit_pct, profit_abs = self._calculate_profit(entry_price, mark_price, notional, side)
        self.last_profits[symbol] = profit_abs

        # Prioridade 1: Stop Loss
        if self._should_stop_loss(entry_price, mark_price, atr_now, side, pair):
            return await self._exit(symbol, size, side, reason="🛑 Stop loss")

        # Prioridade 2: Reversão com candle
        if self._should_exit_by_reversal(symbol, side):
            return await self._exit(symbol, size, side, reason="🔄 Reversal candle")

        # Prioridade 3: Parcial em lucro elevado
        if self._should_take_partial(profit_pct, pair, symbol, size, side):
            return False

        # Prioridade 4: Trailing stop
        if self._should_exit_by_trailing_stop(symbol, mark_price, entry_price, atr_now, side, size):
            return await self._exit(symbol, size, side, reason="🚪 Trailing stop")

        # Prioridade 5: Lucro mínimo atingido
        if self._should_take_profit(profit_abs, profit_pct, pair):
            return await self._exit(symbol, size, side, reason="💰 Profit target")

        # Prioridade 6: HOLD prolongado + tendência contra
        if self._should_exit_by_hold(pair, signal_result, symbol, side, size):
            return True

        # Prioridade 7: Reversão de lucro
        if self._should_exit_by_profit_reversal(symbol, profit_abs, pair):
            return await self._exit(symbol, size, side, reason="⚠️ Profit reversal")

        return False

    async def _get_market_price(self, symbol: str):
        ticker = await self.exchange_client.fetch_ticker(symbol)
        return ticker.get("last") or ticker.get("close")

    def _calculate_profit(self, entry_price, mark_price, notional, side):
        if side == "buy":
            profit_pct = (mark_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - mark_price) / entry_price
        return profit_pct, profit_pct * notional

    def _should_stop_loss(self, entry_price, mark_price, atr, side, pair):
        sl_multiplier = getattr(pair, "stop_loss_atr_multiplier", 1.5)
        stop_loss_price = entry_price - atr * sl_multiplier if side == "buy" else entry_price + atr * sl_multiplier
        return (side == "buy" and mark_price <= stop_loss_price) or (side == "sell" and mark_price >= stop_loss_price)

    def _should_exit_by_reversal(self, symbol: str, side: str) -> bool:

        reversal_top, reversal_bottom = StrategyUtils.detect_reversal_pattern(self.ohlcv)
        print(f"ENTROU AQUIIIIII {reversal_top} {reversal_bottom}")
        return (
            reversal_top if side == "buy"
            else reversal_bottom if side == "sell"
            else False
        )

    def _should_take_partial(self, profit_pct, pair, symbol, size, side) -> bool:
        threshold = getattr(pair, "partial_take_profit_pct", 0.03)
        portion = getattr(pair, "partial_close_pct", 0.5)

        if profit_pct > threshold and not self.partial_taken.get(symbol, False):
            partial_size = size * portion
            logging.info(f"✂️ Partial TP on {symbol} | {portion*100:.0f}% at profit {profit_pct*100:.2f}%")
            self.exchange_client.close_position(symbol, partial_size, self.helpers.get_opposite_side(Signal.from_str(side)))
            self.partial_taken[symbol] = True
            return True
        return False

    def _should_exit_by_trailing_stop(self, symbol, mark_price, entry_price, atr, side, size) -> bool:
        multiplier = getattr(self.pair, "trailing_stop_multiplier", 1.5)
        if self.last_profits.get(symbol, 0) > 0.05:
            multiplier *= 0.5

        distance = atr * multiplier
        prev_stop = self.trailing_stops.get(symbol)

        if side == "buy":
            new_stop = max(prev_stop or entry_price, mark_price - distance)
            self.trailing_stops[symbol] = new_stop
            return mark_price <= new_stop
        else:
            new_stop = min(prev_stop or entry_price, mark_price + distance)
            self.trailing_stops[symbol] = new_stop
            return mark_price >= new_stop

    def _should_take_profit(self, profit_abs, profit_pct, pair):
        return profit_abs >= getattr(pair, "min_profit_abs", 5.0) and profit_pct >= getattr(pair, "min_profit_pct", 0.005)

    def _should_exit_by_hold(self, pair, signal_result, symbol, side, size) -> bool:
        if signal_result.signal == Signal.HOLD and signal_result.confidence and signal_result.confidence >= 0.9:
            self.hold_counters[symbol] = self.hold_counters.get(symbol, 0) + 1
            if self.is_trend_against_position(symbol, side):
                return self._exit_sync(symbol, size, side, "📉 Trend against")
            if self.hold_counters[symbol] >= getattr(pair, "max_hold_candles", 2):
                return self._exit_sync(symbol, size, side, "🔚 HOLD max")
        else:
            self.hold_counters[symbol] = 0
        return False

    def _should_exit_by_profit_reversal(self, symbol, profit_abs, pair):
        last = self.last_profits.get(symbol)
        return last and last >= getattr(pair, "min_profit_abs", 5.0) and profit_abs < last * 0.75

    async def _exit(self, symbol: str, size: float, side: str, reason: str) -> bool:
        logging.info(f"{reason} for {symbol}")
        await self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
        self._reset_state(symbol)
        return True

    def _exit_sync(self, symbol: str, size: float, side: str, reason: str) -> bool:
        logging.info(f"{reason} for {symbol}")
        self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
        self._reset_state(symbol)
        return True

    def _reset_state(self, symbol: str):
        self.trailing_stops.pop(symbol, None)
        self.hold_counters[symbol] = 0
        self.last_profits[symbol] = 0
        self.partial_taken[symbol] = False

    def is_trend_against_position(self, symbol: str, side: str | None) -> bool:
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



