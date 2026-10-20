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

    async def should_exit__(self, ohlcv: OhlcvWrapper, pair: PairConfig, signalResult: SignalResult,
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

        MAX_CANDLES_STALE = 3
        timeframe_minutes = 15
        max_duration_minutes = MAX_CANDLES_STALE * timeframe_minutes

        MIN_PROFIT_TO_SECURE = 0.002  # 0.2%
        MAX_PROFIT_BEFORE_TRAILING = 0.010  # 1.0%

        # 💡 2. CALCULAR O HISTÓRICO DE LUCRO DESDE A ENTRADA
        # Precisamos de obter os preços de fecho desde que a posição abriu até agora.
        # Podes encontrar o índice correspondente ao `entry_time` ou usar os últimos N candles da duração.
        # Como a duração máxima são 3 candles (ou mais), vamos buscar os preços correspondentes à vida da posição:
        candles_in_trade = int(duration_minutes / timeframe_minutes) + 1

        if candles_in_trade > 1 and len(ohlcv.closes) >= candles_in_trade:
            # Fatiar os preços desde a entrada até ao momento atual
            historical_closes = ohlcv.closes[-candles_in_trade:]

            # Calcular o lucro em percentagem para cada vela desde a entrada
            if side == 'buy':
                historical_profits = [(c - entry_price) / entry_price for c in historical_closes]
            else:
                historical_profits = [(entry_price - c) / entry_price for c in historical_closes]

            # Lucro médio desde que abriu a posição
            avg_profit_since_entry = sum(historical_profits) / len(historical_profits)
        else:
            avg_profit_since_entry = profit_pct

        # 💡 3. VERIFICAR SE O LUCRO ESTÁ A CRESCER FACE À MÉDIA DA POSIÇÃO
        # Se o lucro atual for superior à média histórica da posição, significa que está em aceleração positiva.
        is_profit_increasing = profit_pct > avg_profit_since_entry

        # Condição de saída:
        # Passou o tempo + lucro modesto + O LUCRO NÃO ESTÁ A CRESCER (estagnou ou desceu face à média)
        if (duration_minutes >= max_duration_minutes) and \
                (MIN_PROFIT_TO_SECURE <= profit_pct < MAX_PROFIT_BEFORE_TRAILING) and \
                not is_profit_increasing:
            logging.info(
                f"⏳ [TIMEOUT + LUCRO ESTAGNADO] A fechar. Lucro atual ({profit_pct:.2%}) está abaixo ou igual à média desde a entrada ({avg_profit_since_entry:.2%})."
            )
            await self.exchange_client.close_position(pair.symbol, current_position.size,
                                                      self.helpers.get_opposite_side(Signal.from_str(side)))
            return True

        # Se o lucro estiver a subir (is_profit_increasing == True), o bot ignora o timeout e deixa correr!
        logging.info(
            f"[DEBUG] Lucro em crescimento ({profit_pct:.2%} vs média {avg_profit_since_entry:.2%}). A manter posição...")
        await self.exchange_client.apply_trailing_stop(pair.symbol, current_price)

        return False
