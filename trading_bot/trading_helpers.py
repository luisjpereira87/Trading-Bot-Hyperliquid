from typing import Optional


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
        return mapping.get(position_side.lower(), None)
    
    @staticmethod
    def get_close_side_from_position_side(position_side: str) -> Optional[str]:
        """
        Dado o lado da posição ('long' ou 'short'), retorna o lado da ordem para fechar a posição:
        'long' -> 'sell'
        'short' -> 'buy'
        """
        mapping = {
            "long": "sell",
            "short": "buy"
        }
        return mapping.get(position_side.lower())

    @staticmethod
    def is_opposite_side(side1: str, side2: str) -> bool:
        """
        Verifica se side1 e side2 são opostos ('buy' x 'sell').
        """
        opposites = {"buy": "sell", "sell": "buy"}
        return opposites.get(side1) == side2
    
    @staticmethod
    def get_opposite_side(side: str) -> Optional[str]:
        """
        Dado o lado da posição ('sell' ou 'buy'), retorna o lado oposto 
        'buy' -> 'sell'
        'sell' -> 'buy'
        """
        mapping = {
            "buy": "sell",
            "sell": "buy"
        }

        return mapping.get(side.lower())

    @classmethod
    def is_signal_opposite_position(cls, signal_side: str, position_side: str) -> bool:
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
        return signal.get("side") in ["buy", "sell"]

    @staticmethod
    def format_side(side: str) -> str:
        """
        Normaliza o side para lowercase (ex: 'Buy' -> 'buy')
        """
        if side:
            return side.lower()
        return side
    
    @staticmethod
    def get_pair(symbol: str, pairs: list[dict]) -> dict | None:
        for pair in pairs:
            if pair["symbol"] == symbol:
                return pair
        return None