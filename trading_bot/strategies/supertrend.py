from .indicators import Indicators

class SuperTrend:
    def __init__(self, exchange, symbol, timeframe, period=10, multiplier=3):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.period = period
        self.multiplier = multiplier

    async def get_signal(self):
        ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe)
        closes = [c[4] for c in ohlcv]

        if len(closes) < self.period + 2:  # Garantir candles suficientes para cálculo
            return 'hold'

        indicators = Indicators(ohlcv)
        atr = indicators.atr(self.period)
        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]
        hl2 = [(h + l) / 2 for h, l in zip(highs, lows)]

        upperband = [hl2[i] + self.multiplier * atr[i] for i in range(len(atr))]
        lowerband = [hl2[i] - self.multiplier * atr[i] for i in range(len(atr))]

        # Stochastic
        stoch_k, stoch_d = indicators.stochastic()
        k_now = stoch_k[-1]
        d_now = stoch_d[-1]
        k_prev = stoch_k[-2]
        d_prev = stoch_d[-2]

        # Condições SuperTrend
        buy_condition = closes[-2] < upperband[-2] and closes[-1] > upperband[-1]
        sell_condition = closes[-2] > lowerband[-2] and closes[-1] < lowerband[-1]

        # Condições Stochastic
        stoch_buy = (k_prev < d_prev) and (k_now > d_now) and k_now < 20
        stoch_sell = (k_prev > d_prev) and (k_now < d_now) and k_now > 80

        if buy_condition and stoch_buy:
            print("\n📊 Sinal: BUY (SuperTrend + Stochastic)")
            return 'buy'

        if sell_condition and stoch_sell:
            print("\n📊 Sinal: SELL (SuperTrend + Stochastic)")
            return 'sell'

        print("\n📊 Sinal: HOLD (SuperTrend)")
        return 'hold'


