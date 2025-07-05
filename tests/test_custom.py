
from enums.signal_enum import Signal
from trading_bot.trading_helpers import TradingHelpers


class TestCustom:
    def __init__(self):
        pass

    async def run(self):
        print(f" TESTE is_opposite_side with params {Signal.SELL} buy  result => {TradingHelpers.is_opposite_side(Signal.SELL, "buy")}")
        print(f" TESTE get_opposite_side with params {Signal.SELL} result => {TradingHelpers.get_opposite_side(Signal.SELL)}")