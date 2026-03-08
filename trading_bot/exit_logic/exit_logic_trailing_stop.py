import logging

import numpy as np

from commons.enums.signal_enum import Signal
from commons.helpers.trading_helpers import TradingHelpers
from commons.models.open_position_dclass import OpenPosition
from commons.models.signal_result_dclass import SignalResult
from commons.utils.config_loader import PairConfig
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from trading_bot.exchange_base import ExchangeBase
from trading_bot.exchange_client import ExchangeClient


class ExitLogicTrailingStop:
    def __init__(self, helpers: TradingHelpers, exchange_client: ExchangeBase):
        """
        buffer_pct: margem percentual em torno da EMA para tolerância
        atr_factor: quantos ATRs usar para definir perda de força
        """
        self.helpers = helpers
        self.exchange_client = exchange_client


    async def should_exit(self, ohlcv: OhlcvWrapper, pair: PairConfig, signalResult: SignalResult, current_position: OpenPosition, current_price: float):
        if not current_position or not current_position.side:
            return False

        logging.info(f"[DEBUG] Saída lógica baseado em trailing stop")

        await self.exchange_client.apply_trailing_stop(pair.symbol, current_price)

        # A posição continua aberta
        return False
