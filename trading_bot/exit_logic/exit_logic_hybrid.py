import numpy as np

from commons.enums.signal_enum import Signal
from commons.helpers.trading_helpers import TradingHelpers
from commons.models.open_position_dclass import OpenPosition
from commons.models.signal_result_dclass import SignalResult
from commons.utils.config_loader import PairConfig
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from trading_bot.exchange_client import ExchangeClient


class ExitLogicHybrid:
    def __init__(self, helpers: TradingHelpers, exchange_client: ExchangeClient, buffer_pct: float = 0.001, atr_factor: float = 1.0):
        """
        buffer_pct: margem percentual em torno da EMA para tolerância
        atr_factor: quantos ATRs usar para definir perda de força
        """
        self.helpers = helpers
        self.exchange_client = exchange_client
        self.buffer_pct = buffer_pct
        self.atr_factor = atr_factor

    async def should_exit(self, ohlcv: OhlcvWrapper, pair: PairConfig, signal: SignalResult, current_position: OpenPosition):
        if not current_position or not current_position.side:
            return False

        side = Signal.from_str(current_position.side)
        current_price = await self.exchange_client.get_entry_price(pair.symbol)
        entry_price = float(current_position.entry_price)

        # Indicadores
        indicators = IndicatorsUtils(ohlcv)
        ema21_values = indicators.ema(21)
        last_ema = ema21_values[-1]

        atr_values = indicators.atr(period=14)
        last_atr = atr_values[-1]

        # PnL
        if side == Signal.BUY:
            pl = current_price - entry_price
        else:
            pl = entry_price - current_price

        # PnL percentual
        pl_pct = pl / entry_price

        # 1) Verificar cruzamento contra a tendência
        if side == Signal.BUY and current_price < last_ema * (1 - self.buffer_pct):
            await self._exit(pair.symbol, current_position.size, current_position.side)
            return True
        elif side == Signal.SELL and current_price > last_ema * (1 + self.buffer_pct):
            await self._exit(pair.symbol, current_position.size, current_position.side)
            return True

        # 2) Verificar perda de força com ATR
        # Ex: se distância do preço atual ao máximo/mínimo recente é menor que atr_factor * ATR → encerra
        lookback = 5
        recent_high = max(ohlcv.highs[-lookback:])
        recent_low = min(ohlcv.lows[-lookback:])

        if side == Signal.BUY:
            if (recent_high - current_price) < self.atr_factor * last_atr and pl > 0:
                await self._exit(pair.symbol, current_position.size, current_position.side)
                return True
        else:
            if (current_price - recent_low) < self.atr_factor * last_atr and pl > 0:
                await self._exit(pair.symbol, current_position.size, current_position.side)
                return True

        # Se nenhuma condição satisfeita, mantém posição
        return False
    
    async def _exit(self, symbol: str, size: float, side: str):
        await self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
        return True