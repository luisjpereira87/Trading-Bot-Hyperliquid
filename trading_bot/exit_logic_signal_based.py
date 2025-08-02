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

    async def should_exit(self, ohlcv: OhlcvWrapper, pair: PairConfig, signal: SignalResult, current_position: OpenPosition, price_ref: float):

        logging.info(f"[DEBUG ExitLogicSignalBased] position: {current_position} signal: {signal} pair: {pair}")
        if not current_position or not current_position.side:
            return False
        
        buy_score = signal.buy_score or 0.00
        sell_score = signal.sell_score or 0.00
        hold_score = signal.hold_score or 0.00
        
        #current_price = await self.exchange_client.get_entry_price(pair.symbol)
        entry_price = float(current_position.entry_price)

        side = Signal.from_str(current_position.side)

        # P&L atual
        if side == Signal.BUY:
            pl = (price_ref - entry_price) * current_position.size
        else:
            pl = (entry_price - price_ref) * current_position.size

        
        if self.should_exit_due_to_failed_tp(ohlcv, price_ref, current_position):
            logging.info(f"[ExitLogic] Quase tocou TP e reverteu — Saída antecipada ({pair})")
            await self._exit(pair.symbol, current_position.size, current_position.side)
            return True
        

        if side == Signal.BUY and sell_score > buy_score:
            logging.info("🔁 Reversão: SELL > BUY → fechar BUY")
            await self._exit(pair.symbol, current_position.size, current_position.side)
            return True

        if side == Signal.SELL and buy_score > sell_score:
            logging.info("🔁 Reversão: BUY > SELL → fechar SELL")
            await self._exit(pair.symbol, current_position.size, current_position.side)
            return True

        return False

    
    def should_exit_due_to_failed_tp(self, ohlcv: OhlcvWrapper, current_price: float,  current_position: OpenPosition, lookback: int = 3, threshold: float = 0.02) -> bool:
        """
        Verifica se o preço esteve quase a atingir o TP e agora está a reverter.
        - `lookback`: nº de candles fechados a analisar
        - `threshold`: distância percentual até ao TP (ex: 0.002 = 0.2%)
        """
        entry_price = float(current_position.entry_price)
        side = Signal.from_str(current_position.side)
        tp = current_position.tp or 0.02
        sl = current_position.sl or 0.01
        
        if tp is None or side is None:
            return False

        recent = ohlcv.get_recent_closed(lookback)

        # 1. Verifica se algum candle esteve quase a tocar o TP
        near_tp = False
        for c in recent:
            price = c.high if side == Signal.BUY else c.low
            dist = abs(price - tp) / tp
            if dist < threshold:
                near_tp = True
                break

        if not near_tp:
            return False

        # 2. Verifica se o preço atual se está a afastar do TP
        last_close = recent[-1].close
        is_reversing = (side == Signal.BUY and current_price < last_close) or \
                    (side == Signal.SELL and current_price > last_close)

        # 3. Opcional: garantir ainda lucro antes de sair
        still_profitable = (side == Signal.BUY and current_price > entry_price) or \
                        (side == Signal.SELL and current_price < entry_price)

        return near_tp and is_reversing and still_profitable
    
    async def _exit(self, symbol: str, size: float, side: str) -> bool:
        await self.exchange_client.close_position(symbol, size, self.helpers.get_opposite_side(Signal.from_str(side)))
        return True
