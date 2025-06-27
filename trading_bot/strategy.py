from trading_bot.strategies.ai_supertrend import AISuperTrend
from trading_bot.strategies.supertrend import SuperTrend
from trading_bot.strategies.ut_bot_alerts import UTBotAlerts

class Strategy:
    def __init__(self, exchange, symbol, timeframe, name='ai_supertrend'):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.name = name.lower()
        self.strategy = None

    async def get_signal(self):
        if self.name == 'ai_supertrend':
            mode = await self._detect_mode()
            self.strategy = AISuperTrend(self.exchange, self.symbol, self.timeframe, mode=mode)
        elif self.name == 'supertrend':
            self.strategy = SuperTrend(self.exchange, self.symbol, self.timeframe)
        elif self.name == 'ut_bot_alerts':
            self.strategy = UTBotAlerts(self.exchange, self.symbol, self.timeframe)
        else:
            raise ValueError(f"Estratégia '{self.name}' não reconhecida.")

        return await self.strategy.get_signal()

    async def _detect_mode(self, period=30, volume_multiplier=1.3):
        try:
            """
            Detecta dinamicamente se o modo deve ser 'aggressive' ou 'conservative'
            com base no volume atual vs média de volume.
            """
            ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=period + 1)
            if len(ohlcv) <= period:
                return 'conservative'

            volumes = [candle[5] for candle in ohlcv[-(period + 1):-1]]  # ignora o último (incompleto)
            avg_volume = sum(volumes) / len(volumes)
            current_volume = ohlcv[-1][5]

            if current_volume > avg_volume * volume_multiplier:
                return 'aggressive'
            return 'conservative'
        except Exception as e:
            print(f"[VolumeAnalyzer] Erro ao analisar volume: {e}")
            return 'conservative'

