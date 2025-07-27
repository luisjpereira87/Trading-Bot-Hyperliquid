from typing import Optional

from commons.enums.signal_enum import Signal
from commons.utils.config_loader import PairConfig


class TradingHelpers:
    @staticmethod
    def position_side_to_signal_side(position_side: str) -> Optional[str]:
        """
        Converte 'long' para 'buy' e 'short' para 'sell'.
        """
        mapping = {
            "long": "buy",
            "short": "sell"
        }
        return mapping.get(position_side, None)
    
    @staticmethod
    def get_close_side_from_position_side(position_side: Signal) -> Optional[Signal]:
        """
        Dado o lado da posição ('long' ou 'short'), retorna o lado da ordem para fechar a posição:
        'long' -> 'sell'
        'short' -> 'buy'
        """
        mapping = {
            Signal.LONG: Signal.SELL,
            Signal.SHORT: Signal.BUY
        }
        return mapping.get(position_side)

    @staticmethod
    def is_opposite_side(side1: Signal, side2: Signal) -> bool:
        opposites = {
            Signal.BUY: Signal.SELL,
            Signal.SELL: Signal.BUY
        }

        # Evita comparação inválida com HOLD
        if side1 not in opposites or side2 not in opposites:
            return False

        return opposites[side1] == side2
    
    @staticmethod
    def get_opposite_side(side: Signal) -> Signal:
        """
        Dado o lado da posição ('sell' ou 'buy'), retorna o lado oposto 
        'buy' -> 'sell'
        'sell' -> 'buy'
        """
        mapping = {
            Signal.BUY: Signal.SELL,
            Signal.SELL: Signal.BUY
        }

        if side == Signal.BUY:
            return Signal.SELL
        elif side == Signal.SELL:
            return Signal.BUY
        else:
            return Signal.HOLD

        #return mapping.get(side)

    @classmethod
    def is_signal_opposite_position(cls, signal_side: Signal, position_side: Signal) -> bool:
        """
        Verifica se o sinal é contrário à posição aberta.
        """
        #pos_signal = cls.position_side_to_signal_side(position_side)
        if position_side is None:
            return False
        return cls.is_opposite_side(signal_side, position_side)

    
    @staticmethod
    def is_valid_signal(signal: dict) -> bool:
        """
        Valida se o dicionário de sinal tem o campo 'side' correto.
        """
        return signal.get("side") in [Signal.BUY, Signal.SELL]

    @staticmethod
    def format_side(side: Signal) -> Signal:
        """
        Normaliza o side para lowercase (ex: 'Buy' -> 'buy')
        """
        if side:
            return side
        return side
    
    @staticmethod
    def get_pair(symbol: str, pairs: list[PairConfig]) -> Optional[PairConfig]:
        for pair in pairs:
            if pair.symbol == symbol:
                return pair
        return None