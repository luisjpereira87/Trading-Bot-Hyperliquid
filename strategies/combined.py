from commons.enums.signal_enum import Signal
from strategies.ai_supertrend import AISuperTrend
from strategies.ml_strategy import MLStrategy


class CombinedStrategy:
    def __init__(self, exchange, symbol, timeframe):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.ml_strategy = MLStrategy(exchange)
        self.other_strategy = AISuperTrend(exchange)

    async def get_signal(self):
        ml_signal = await self.ml_strategy.get_signal()
        other_signal = await self.other_strategy.get_signal()

        valid_signals = {Signal.BUY, Signal.SELL, Signal.HOLD}

        def extract_side(signal):
            if isinstance(signal, dict):
                return signal.get("side", Signal.HOLD)
            elif isinstance(signal, str) and signal in valid_signals:
                return signal
            return "hold"

        ml_side = extract_side(ml_signal)
        other_side = extract_side(other_signal)

        if ml_side == other_side:
            return {"side": ml_side, "mode": "combined"}
        if ml_side == Signal.HOLD and other_side in {Signal.BUY, Signal.SELL}:
            return {"side": other_side, "mode": "combined"}
        if other_side == Signal.HOLD and ml_side in {Signal.BUY, Signal.SELL}:
            return {"side": ml_side, "mode": "combined"}
        return {"side":Signal.HOLD, "mode": "combined"}
