from abc import ABC, abstractmethod
from typing import Any


class StrategyBase(ABC):
    @abstractmethod
    async def get_signal(self) -> Any:
        pass