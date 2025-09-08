import logging

import numpy as np

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
        #tolerance = 0.001  # 0.1% de tolerância

        # ATR (usamos últimos valores para medir força)
        atr_values = indicators.atr(period=14)
        last_atr = atr_values[-1]
        prev_atr = atr_values[-2] if len(atr_values) > 1 else last_atr

        atr_multiplier = getattr(pair, "atr_multiplier", 1.5)
        atr_momentum_drop = getattr(pair, "atr_momentum_drop", 0.95)
        tolerance = getattr(pair, "psar_tolerance", 0.001)

        # P&L atual
        if side == Signal.BUY:
            pl = current_price - entry_price
        else:
            pl = entry_price - current_price

        # === 1) Fecho principal por PSAR ===
        if side == Signal.BUY:
            # Sai se o preço fechar abaixo do PSAR
            if (current_price < last_psar * (1 + tolerance)):
                logging.info("🔁 Reversão por PSAR: fechar BUY")
                await self._exit(pair.symbol, current_position.size, current_position.side)
                return True

        elif side == Signal.SELL:
            # Sai se o preço fechar acima do PSAR
            if (current_price > last_psar * (1 + tolerance)):
                logging.info("🔁 Reversão por PSAR: fechar SELL")
                await self._exit(pair.symbol, current_position.size, current_position.side)
                return True
        """         
        # === 2) Fecho auxiliar por perda de momentum (ATR a cair) ===
        if pl > 0:  # só aplica se já está em lucro
            if last_atr < prev_atr * atr_momentum_drop:  # queda de mais de 5% no ATR
                logging.info("⚠️ ATR caiu (mercado perdeu força) → fechar posição e proteger lucro")
                await self._exit(pair.symbol, current_position.size, current_position.side)
                return True
        
        # === 3) Trailing stop dinâmico baseado no ATR ===
        if side == Signal.BUY:
            trailing_stop = current_price - atr_multiplier * last_atr
            if trailing_stop > entry_price and current_price < trailing_stop:
                logging.info("📉 Stop ATR atingido (BUY) → fechar posição")
                await self._exit(pair.symbol, current_position.size, current_position.side)
                return True

        elif side == Signal.SELL:
            trailing_stop = current_price + atr_multiplier * last_atr
            if trailing_stop < entry_price and current_price > trailing_stop:
                logging.info("📈 Stop ATR atingido (SELL) → fechar posição")
                await self._exit(pair.symbol, current_position.size, current_position.side)
                return True
        """
        """
        # === 4) Candle contrário à tendência com lucro ===
        if pl > 0:  # só aplica se estamos em lucro
            prev_close = ohlcv.closes[-2]
            last_close = ohlcv.closes[-1]

            if side == Signal.BUY:
                # candle vermelho contra a tendência
                if last_close < prev_close:
                    logging.info("🛑 Candle vermelho contra BUY → fechar posição")
                    await self._exit(pair.symbol, current_position.size, current_position.side)
                    return True

            elif side == Signal.SELL:
                # candle verde contra a tendência
                if last_close > prev_close:
                    logging.info("🛑 Candle verde contra SELL → fechar posição")
                    await self._exit(pair.symbol, current_position.size, current_position.side)
                    return True
        """
        """
        last_open = ohlcv.opens[-1]
        last_close = ohlcv.closes[-1]
        last_volume = ohlcv.volumes[-1]
        mean_volume = np.mean(ohlcv.volumes[-20:])

        # Candle contrário
        if side == Signal.BUY:
            candle_contra = last_close < last_open   # candle vermelho
        else:
            candle_contra = last_close > last_open   # candle verde

        # Força do candle (≥ 50% ATR)
        candle_size = abs(last_close - last_open)
        candle_forte = candle_size >= 0.5 * last_atr

        # Volume anormal (≥ 1.5x média)
        volume_alto = last_volume > mean_volume * 1.5

        if pl > 0 and candle_contra and candle_forte and volume_alto:
            logging.info("🚨 Candle contrário forte + volume alto → fechar posição e proteger lucro")
            await self._exit(pair.symbol, current_position.size, current_position.side)
            return True
        """
        return False
    
    async def _exit(self, symbol: str, size: float, side: str) -> bool:
        await self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
        return True