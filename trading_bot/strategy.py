from trading_bot.strategies.ai_supertrend import AISuperTrend
from trading_bot.strategies.supertrend import SuperTrend
from trading_bot.strategies.ut_bot_alerts import UTBotAlerts

class Strategy:
    def __init__(self, exchange, symbol, timeframe, name='ai_supertrend'):
        self.name = name.lower()
        if self.name == 'ai_supertrend':
            self.strategy = AISuperTrend(exchange, symbol, timeframe)
        elif self.name == 'supertrend':
            self.strategy = SuperTrend(exchange, symbol, timeframe)
        elif self.name == 'ut_bot_alerts':
            self.strategy = UTBotAlerts(exchange, symbol, timeframe)
        else:
            raise ValueError(f"Estratégia '{name}' não reconhecida.")

    async def get_signal(self):
        return await self.strategy.get_signal()

