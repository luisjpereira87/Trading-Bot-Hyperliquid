from .indicators import Indicators

class UTBotAlerts:
    def __init__(self, exchange, symbol, timeframe):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe

    async def get_signal(self):
        ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe)
        if len(ohlcv) < 14:
            return 'hold'

        indicators = Indicators(ohlcv)
        rsi = indicators.rsi()
        atr = indicators.atr()
        closes = indicators.closes

        stoch_k, stoch_d = indicators.stochastic()
        k_now = stoch_k[-1]
        d_now = stoch_d[-1]
        k_prev = stoch_k[-2]
        d_prev = stoch_d[-2]

        price = closes[-1]
        rsi_now = rsi[-1]
        atr_now = atr[-1]

        # UTBot Alerts Lógica simples (pode ajustar conforme preferir)
        buy_condition = price > closes[-2] and rsi_now > 50
        sell_condition = price < closes[-2] and rsi_now < 50

        # Confirmação Stochastic
        stoch_buy = (k_prev < d_prev) and (k_now > d_now) and k_now < 20
        stoch_sell = (k_prev > d_prev) and (k_now < d_now) and k_now > 80

        if buy_condition and stoch_buy:
            print("\n📊 Sinal: BUY (UTBot + Stochastic)")
            return 'buy'

        if sell_condition and stoch_sell:
            print("\n📊 Sinal: SELL (UTBot + Stochastic)")
            return 'sell'

        print("\n📊 Sinal: HOLD (UTBot)")
        return 'hold'


