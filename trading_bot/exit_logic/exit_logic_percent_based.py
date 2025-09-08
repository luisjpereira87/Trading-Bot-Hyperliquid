import logging

from commons.enums.signal_enum import Signal
from commons.helpers.trading_helpers import TradingHelpers
from commons.models.open_position_dclass import OpenPosition
from commons.models.signal_result_dclass import SignalResult
from commons.utils.config_loader import PairConfig
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from trading_bot.exchange_client import ExchangeClient


class ExitLogicPercentBased:
    def __init__(self, helpers: TradingHelpers, exchange_client: ExchangeClient):
        self.helpers = helpers
        self.exchange_client = exchange_client

    async def should_exit(self, ohlcv: OhlcvWrapper, pair: PairConfig, signal: SignalResult, current_position: OpenPosition):
        if not current_position or not current_position.side:
            return False

        side = Signal.from_str(current_position.side)
        current_price = await self.exchange_client.get_entry_price(pair.symbol)
        entry_price = float(current_position.entry_price)
        leverage = getattr(pair, "leverage", 1)  # por default 1x se não tiver alavancagem

        # thresholds base (sem alavancagem)
        tp_pct = 0.02   # 2%
        sl_pct = 0.01   # 1%

        # ajustar pela alavancagem
        tp_pct = tp_pct / leverage
        sl_pct = sl_pct / leverage

        # calcular P&L %
        pl_pct = (current_price - entry_price) / entry_price if side == Signal.BUY else (entry_price - current_price) / entry_price

        # validar candles recentes (últimos 3)
        closes = ohlcv.closes
        recent_closes = closes[-3:]
        
        # BUY -> tendência contrária se últimos candles estão a descer
        if side == Signal.BUY and all(recent_closes[i] < recent_closes[i-1] for i in range(1, len(recent_closes))):
            logging.info("📉 Tendência contrária em BUY → fechar posição")
            await self._exit(pair.symbol, current_position.size, current_position.side)
            return True

        # SELL -> tendência contrária se últimos candles estão a subir
        if side == Signal.SELL and all(recent_closes[i] > recent_closes[i-1] for i in range(1, len(recent_closes))):
            logging.info("📈 Tendência contrária em SELL → fechar posição")
            await self._exit(pair.symbol, current_position.size, current_position.side)
            return True

       
        # Take Profit
        if pl_pct >= tp_pct:
            logging.info(f"✅ Take Profit atingido ({pl_pct:.2%}) → fechar posição")
            await self._exit(pair.symbol, current_position.size, current_position.side)
            return True
        """
        # Stop Loss
        if pl_pct <= -sl_pct:
            logging.info(f"⛔ Stop Loss atingido ({pl_pct:.2%}) → fechar posição")
            await self._exit(pair.symbol, current_position.size, current_position.side)
            return True
        """
        return False

    async def _exit(self, symbol: str, size: float, side: str) -> bool:
        await self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
        return True