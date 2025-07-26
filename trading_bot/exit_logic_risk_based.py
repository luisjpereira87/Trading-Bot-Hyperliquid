from commons.enums.signal_enum import Signal
from commons.models.signal_result_dclass import SignalResult
from commons.utils.config_loader import PairConfig
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from trading_bot.exchange_client import ExchangeClient
from trading_bot.trading_helpers import TradingHelpers


class ExitLogicRiskBased:
    def __init__(self, helpers: TradingHelpers, exchange_client: ExchangeClient):
        self.helpers = helpers
        self.exchange_client = exchange_client

    async def should_exit(self, ohlcv: OhlcvWrapper, pair: PairConfig, signal: SignalResult, current_position):
        # Exemplo: fechar se perda maior que 1R ou lucro maior que 3R
        entry_price = float(current_position.entry_price)
        current_price = await self.exchange_client.get_entry_price(pair.symbol)
        position_size = float(current_position.size)
        side = Signal.from_str(current_position.side)

        # Calcular R (risco) em valor absoluto
        if signal.sl is None:
            return False  # Se não tem SL definido, não sai por R

        stop_loss = float(signal.sl)
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0:
            return False  # Evitar divisão por zero

        # P&L atual
        if side == Signal.BUY:
            pl = current_price - entry_price
        else:
            pl = entry_price - current_price

        r_multiple = pl / risk_per_unit

        # Condições simples para saída
        if r_multiple <= -1:  # Perda 1R
            print(f"Saindo da posição pois atingiu -1R")
            return True
        if r_multiple >= 3:  # Lucro 3R
            print(f"Saindo da posição pois atingiu +3R")
            return True

        # Lógica para mover o SL para trailing stop:
        # Se o preço avançou 1.5R, movemos SL para o ponto de entrada (break even)
        if r_multiple >= 1.5 and stop_loss != entry_price:
            new_sl = entry_price
            await self.exchange_client.modify_stop_loss_order(pair.symbol, current_position.id, new_sl)
            print(f"SL movido para break even em {new_sl}")

        # Pode adicionar outras condições aqui (candles reversão, volume, etc)

        return False
