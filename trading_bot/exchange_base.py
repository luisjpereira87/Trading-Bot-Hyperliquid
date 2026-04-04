import asyncio
import logging
from abc import ABC, abstractmethod

from commons.enums.signal_enum import Signal
from commons.enums.timeframe_enum import TimeframeEnum
from commons.models.ohlcv_format_dclass import OhlcvFormat
from commons.models.open_position_dclass import OpenPosition
from commons.models.opened_order_dclass import OpenedOrder
from commons.utils.config_loader import PairConfig
from trading_bot.profit_manager import ProfitManager


class ExchangeBase(ABC):

    def __init__(self):
        self.profit_manager = ProfitManager()

    @abstractmethod
    def get_name(self):
        return "Exchange"
    
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: TimeframeEnum, since: int = None, limit: int = None, is_higher: bool = False, is_training = False) -> OhlcvFormat:
        pass

    @abstractmethod
    async def get_available_balance(self) -> float:
        pass

    @abstractmethod
    async def get_open_position(self, symbol: str) -> (OpenPosition | None):
        pass

    #@abstractmethod
    #async def place_entry_order(self, symbol: str, size: float, side: Signal) -> OpenedOrder:
    #    pass

    @abstractmethod
    async def cancel_all_orders(self, symbol: str):
        pass

    @abstractmethod
    async def close_position(self, symbol: str, amount: float, side: Signal):
        pass

    @abstractmethod
    async def get_entry_price(self, symbol: str) -> float:
        pass
    
    @abstractmethod
    async def place_entry_order(self, symbol: str, leverage: float, entry_amount: float, price_ref: float, side: Signal, sl_price: (float|None) = None, tp_price: (float|None) = None) -> OpenedOrder:
         pass
    
    @abstractmethod
    async def print_open_orders(self, symbol: str):
        pass

    @abstractmethod
    async def print_balance(self):
        pass

    @abstractmethod
    async def _place_protections(self, symbol: str, size: float, side: str, sl: float, tp: float):
        pass

    async def apply_trailing_stop(self, symbol: str, current_price: float):
        """
        Lógica Universal de Gestão de Profit e Trailing.
        Implementada na Base para evitar repetição em Nado/Hyperliquid/Mock.
        """
        pos = await self.get_open_position(symbol)
        if not pos or abs(pos.size) < 1e-8:
            self.profit_manager.clear(symbol)
            return

        # 1. Cálculo de PNL (Independente de exchange)
        entry_price = float(pos.entry_price)
        is_buy = 'buy' in str(pos.side).lower()

        # PNL Simples (Podes multiplicar pela leverage aqui se quiseres ROE)
        pnl_pct = (current_price - entry_price) / entry_price if is_buy else (entry_price - current_price) / entry_price

        # 2. Consultar o Cérebro
        decision = self.profit_manager.update_and_check(symbol, pnl_pct)

        # 3. Execução A: Fecho por Momentum (Recuo de lucro)
        if decision["should_market_close"]:
            logging.info(f"⚠️ [MOMENTUM] {symbol} recuou do pico. Fechando posição!")
            # Note: Usamos o close_position que as subclasses implementam
            await self.close_position(symbol, abs(pos.size), Signal.from_str(pos.side))
            return

        # 4. Execução B: Atualização de Patamar (Trailing Stop)
        if decision["should_update_trailing"]:
            logging.info(decision["log_msg"])

            # Cálculo dos preços de proteção
            adjustment = decision["adjustment"]
            # Preço do SL (Garante o lucro do patamar)
            new_sl = entry_price * (1 + adjustment) if is_buy else entry_price * (1 - adjustment)

            # Take Profit dinâmico (Opcional: podes manter o original ou um alvo fixo)
            # Aqui um exemplo de TP a 5% da entrada
            new_tp = entry_price * 1.05 if is_buy else entry_price * 0.95

            # Orquestração de ordens
            await self.cancel_all_orders(symbol)
            await asyncio.sleep(1.2)  # Delay para segurança das APIs

            # Chama o método de proteção da subclasse
            # Nota: Adicionei os argumentos necessários para as ordens
            await self._place_protections(symbol, abs(pos.size), pos.side, new_sl, new_tp)