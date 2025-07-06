
from enums.signal_enum import Signal
from trading_bot.trading_helpers import TradingHelpers


class TestCustom:
    def __init__(self):
        pass

    async def run(self):
        print(f" TESTE is_opposite_side with params {Signal.SELL} buy  result => {TradingHelpers.is_opposite_side(Signal.SELL, "buy")}")
        print(f" TESTE get_opposite_side with params {Signal.SELL} result => {TradingHelpers.get_opposite_side(Signal.SELL)}")
        print(f" TESTE get_opposite_side with params {Signal.BUY} result => {TradingHelpers.get_opposite_side(Signal.BUY)}")
        print(f" TESTE is_signal_opposite_position with params {Signal.SELL} result => {TradingHelpers.is_signal_opposite_position(Signal.SELL, "sell")}")
        print(f" TESTE is_signal_opposite_position with params {Signal.BUY} result => {TradingHelpers.is_signal_opposite_position(Signal.BUY, "sell")}")
    