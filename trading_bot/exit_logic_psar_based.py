import logging

from commons.enums.signal_enum import Signal
from commons.helpers.trading_helpers import TradingHelpers
from commons.models.open_position_dclass import OpenPosition
from commons.models.signal_result_dclass import SignalResult
from commons.utils.config_loader import PairConfig
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from commons.utils.strategies.trend_utils import TrendUtils
from trading_bot.exchange_client import ExchangeClient


class ExitLogicPSARBased:
    def __init__(self, helpers: TradingHelpers, exchange_client: ExchangeClient):
        self.helpers = helpers
        self.exchange_client = exchange_client

    async def should_exit(self, ohlcv: OhlcvWrapper, pair: PairConfig, signal: SignalResult, current_position: OpenPosition):

        logging.info(f"[DEBUG ExitLogicPSARBased] position: {current_position} signal: {signal} pair: {pair}")
        if not current_position or not current_position.side:
            return False

        side = Signal.from_str(current_position.side)
        current_price = await self.exchange_client.get_entry_price(pair.symbol)
        entry_price = float(current_position.entry_price)

        # Calcular PSAR
        indicators = IndicatorsUtils(ohlcv)
        psar_values = indicators.psar()  
        last_psar = psar_values[-1]
        tolerance = 0.001  # 0.1% de tolerância

        # P&L atual
        if side == Signal.BUY:
            pl = current_price - entry_price
        else:
            pl = entry_price - current_price

        trend_signal = TrendUtils.trend_strength_signal(ohlcv)

        if side == Signal.BUY:
            # Sai se o preço fechar abaixo do PSAR
            if (current_price < last_psar * (1 + tolerance) and trend_signal != side):
                logging.info("🔁 Reversão por PSAR: fechar BUY")
                await self._exit(pair.symbol, current_position.size, current_position.side)
                return True

        elif side == Signal.SELL:
            # Sai se o preço fechar acima do PSAR
            if (current_price > last_psar * (1 + tolerance) and trend_signal != side):
                logging.info("🔁 Reversão por PSAR: fechar SELL")
                await self._exit(pair.symbol, current_position.size, current_position.side)
                return True

        return False
    
    async def _exit(self, symbol: str, size: float, side: str) -> bool:
        await self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
        return True