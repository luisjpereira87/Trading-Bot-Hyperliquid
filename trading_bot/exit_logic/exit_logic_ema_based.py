import logging

import numpy as np

from commons.enums.signal_enum import Signal
from commons.helpers.trading_helpers import TradingHelpers
from commons.models.open_position_dclass import OpenPosition
from commons.models.signal_result_dclass import SignalResult
from commons.utils.ai_supertrend.ai_super_trend_utils import AISuperTrendUtils
from commons.utils.config_loader import PairConfig
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from trading_bot.exchange_client import ExchangeClient


class ExitLogicEmaBased:
    def __init__(self, helpers: TradingHelpers, exchange_client: ExchangeClient):
        """
        buffer_pct: margem percentual em torno da EMA para tolerância
        atr_factor: quantos ATRs usar para definir perda de força
        """
        self.helpers = helpers
        self.exchange_client = exchange_client


    async def should_exit(self, ohlcv: OhlcvWrapper, pair: PairConfig, signal: SignalResult, current_position: OpenPosition):
        if not current_position or not current_position.side:
            return False

        aISuperTrendUtils = AISuperTrendUtils(ohlcv)
        ema_cross_signal = aISuperTrendUtils.get_ema_cross_signal()

        logging.info(f"[DEBUG] Saída lógica baseado em reversão de tendência sinal anterior: {ema_cross_signal[-2]}")
        logging.info(f"[DEBUG] Saída lógica baseado em reversão de tendência sinal atual: {ema_cross_signal[-1]}")

        # 1) Verificar cruzamento contra a tendência
        if ema_cross_signal[-1] == Signal.CLOSE:
            await self._exit(pair.symbol, current_position.size, current_position.side)
            return True
        

        # Se nenhuma condição satisfeita, mantém posição
        return False
    
    async def _exit(self, symbol: str, size: float, side: str):
        await self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
        return True
