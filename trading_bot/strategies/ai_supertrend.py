from .indicators import Indicators

class AISuperTrend:
    def __init__(self, exchange, symbol, timeframe):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe

    async def get_signal(self):
        ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe)
        if len(ohlcv) < 21:
            return 'hold'

        indicators = Indicators(ohlcv)

        atr = indicators.atr()
        ema21 = indicators.ema()
        rsi = indicators.rsi()
        stoch_k, stoch_d = indicators.stochastic()

        multiplier = 1.5
        closes = indicators.closes
        upper_band = [closes[i] + multiplier * atr[i] for i in range(len(atr))]
        lower_band = [closes[i] - multiplier * atr[i] for i in range(len(atr))]

        price = closes[-1]
        ema_now = ema21[-1]
        rsi_now = rsi[-1]
        k_now = stoch_k[-1]
        d_now = stoch_d[-1]
        k_prev = stoch_k[-2]
        d_prev = stoch_d[-2]

        # Condições anteriores
        buy_condition = closes[-2] < lower_band[-2] and closes[-1] > lower_band[-1] and price > ema_now and 40 <= rsi_now <= 70
        sell_condition = closes[-2] > upper_band[-2] and closes[-1] < upper_band[-1] and price < ema_now and 30 <= rsi_now <= 60

        # Condições Stochastic
        stoch_buy = (k_prev < d_prev) and (k_now > d_now) and k_now < 20  # Cruzamento de baixo para cima e sobrevenda
        stoch_sell = (k_prev > d_prev) and (k_now < d_now) and k_now > 80  # Cruzamento de cima para baixo e sobrecompra

        if buy_condition and stoch_buy:
            return 'buy'
        if sell_condition and stoch_sell:
            return 'sell'
        return 'hold'



