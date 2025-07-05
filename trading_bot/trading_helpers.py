from typing import Optional

from enums.signal_enum import Signal
from utils.config_loader import PairConfig


class TradingHelpers:
    @staticmethod
    def position_side_to_signal_side(position_side: Signal) -> Optional[Signal]:
        """
        Converte 'long' para 'buy' e 'short' para 'sell'.
        """
        mapping = {
            Signal.LONG: Signal.BUY,
            Signal.SHORT: Signal.SELL
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
    def is_opposite_side(side1: Signal, side2: str) -> bool:
        """
        Verifica se side1 e side2 são opostos ('buy' x 'sell').
        """
        opposites = {Signal.BUY: Signal.SELL, Signal.SELL: Signal.BUY}
        return opposites.get(side1) == side2
    
    @staticmethod
    def get_opposite_side(side: Signal) -> Optional[Signal]:
        """
        Dado o lado da posição ('sell' ou 'buy'), retorna o lado oposto 
        'buy' -> 'sell'
        'sell' -> 'buy'
        """
        mapping = {
            Signal.BUY: Signal.SELL,
            Signal.SELL: Signal.BUY
        }

        return mapping.get(side)

    @classmethod
    def is_signal_opposite_position(cls, signal_side: Signal, position_side: str) -> bool:
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