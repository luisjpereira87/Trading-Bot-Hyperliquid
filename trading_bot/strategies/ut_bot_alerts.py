class UTBotAlerts:
    def __init__(self, exchange, symbol, timeframe):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe

    async def get_signal(self):
        ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe)
        closes = [c[4] for c in ohlcv]

        if len(closes) < 2:
            return 'hold'

        # Lógica simplificada baseada em momentum
        delta = closes[-1] - closes[-2]

        if delta > 0.5:
            print("\n📊 Sinal: BUY (UT Bot Alerts simulado)")
            return 'buy'
        elif delta < -0.5:
            print("\n📊 Sinal: SELL (UT Bot Alerts simulado)")
            return 'sell'

        print("\n📊 Sinal: HOLD (UT Bot Alerts simulado)")
        return 'hold'
