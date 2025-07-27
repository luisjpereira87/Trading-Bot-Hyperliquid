import logging

from commons.enums.signal_enum import Signal
from commons.helpers.trading_helpers import TradingHelpers
from commons.models.open_position_dclass import OpenPosition
from commons.models.signal_result_dclass import SignalResult
from commons.utils.config_loader import PairConfig
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from trading_bot.exchange_client import ExchangeClient


class ExitLogicSignalBased:
    def __init__(self, helpers: TradingHelpers, exchange_client: ExchangeClient):
        self.helpers = helpers
        self.exchange_client = exchange_client

    async def should_exit(self, ohlcv: OhlcvWrapper, pair: PairConfig, signal: SignalResult, current_position: OpenPosition):
        
        if not current_position or not signal.buy_score or not signal.sell_score or not signal.hold_score or not current_position.side:
            return False

        side = Signal.from_str(current_position.side)

        if side == Signal.BUY and signal.sell_score > signal.buy_score and signal.sell_score > signal.hold_score:
            print("🔁 Reversão: SELL > BUY e HOLD → fechar BUY")
            await self._exit(pair.symbol, current_position.size, current_position.side)
            return True

        if side == Signal.SELL and signal.buy_score > signal.sell_score and signal.buy_score > signal.hold_score:
            print("🔁 Reversão: BUY > SELL e HOLD → fechar SELL")
            await self._exit(pair.symbol, current_position.size, current_position.side)
            return True

        return False
    
    async def _exit(self, symbol: str, size: float, side: str) -> bool:
        await self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
        return True