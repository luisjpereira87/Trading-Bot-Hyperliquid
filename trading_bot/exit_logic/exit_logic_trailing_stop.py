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


    async def should_exit_(self, ohlcv: OhlcvWrapper, pair: PairConfig, signalResult: SignalResult, current_position: OpenPosition, current_price: float):
        if not current_position or not current_position.side:
            return False

        logging.info(f"[DEBUG] Saída lógica baseado em trailing stop")

        await self.exchange_client.apply_trailing_stop(pair.symbol, current_price)

        # A posição continua aberta
        return False

    async def should_exit(self, ohlcv: OhlcvWrapper, pair: PairConfig, signalResult: SignalResult,
                          current_position: OpenPosition, current_price: float):
        if not current_position or not current_position.side:
            return False

        # 1. Calcular o lucro atual em percentagem (%)
        entry_price = current_position.entry_price
        duration_minutes = current_position.duration_minutes
        side = current_position.side

        if side == 'buy':
            profit_pct = (current_price - entry_price) / entry_price
        else:  # 'sell' / short
            profit_pct = (entry_price - current_price) / entry_price

        # Se a tua estratégia usa velas de 15m, por exemplo, 3 velas = 45 minutos
        MAX_CANDLES_STALE = 3
        timeframe_minutes = 15  # Ajusta ao teu timeframe
        max_duration_minutes = MAX_CANDLES_STALE * timeframe_minutes

        # 3. Regra de Saída por Estagnação (Time-Based Exit)
        # Condição: Passou o tempo limite + está positivo + o lucro é modesto (ex: entre 0.2% e 1.0%)
        # Assim evitamos fechar cedo demais se a vela já for um "monstro" de lucro (deixamos o trailing tratar disso)
        MIN_PROFIT_TO_SECURE = 0.002  # 0.2% (acima das taxas da corretora)
        MAX_PROFIT_BEFORE_TRAILING = 0.010  # 1.0% (se passar disto, confiamos no trailing stop)

        if (duration_minutes >= max_duration_minutes) and (
                MIN_PROFIT_TO_SECURE <= profit_pct < MAX_PROFIT_BEFORE_TRAILING):
            logging.info(
                f"⏳ [TIMEOUT] Posição estagnada há {duration_minutes:.0f} mins com lucro de {profit_pct:.2%}. A fechar para garantir o ganho."
            )
            await self.exchange_client.close_position(pair.symbol, current_position.size, self.helpers.get_opposite_side(Signal.from_str(side)))
            return True

        # Se não ativou o timeout, mantém o trailing stop normal na exchange
        logging.info(f"[DEBUG] A manter posição. Aplicando trailing stop...")
        await self.exchange_client.apply_trailing_stop(pair.symbol, current_price)

        # A posição continua aberta pelo fluxo normal
        return False
