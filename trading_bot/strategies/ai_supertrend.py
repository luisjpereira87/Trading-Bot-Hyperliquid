import logging
from .indicators import Indicators

class AISuperTrend:
    def __init__(self, exchange, symbol, timeframe):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe

    async def get_signal(self):
        ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe)
        if len(ohlcv) < 21:
            logging.info(f"{self.symbol} - Dados insuficientes para cálculo dos indicadores.")
            return 'hold'

        indicators = Indicators(ohlcv)

        atr = indicators.atr()
        ema21 = indicators.ema()
        rsi = indicators.rsi()
        stoch_k, stoch_d = indicators.stochastic()

        multiplier = 1.5
        closes = indicators.closes
        highs = indicators.highs
        lows = indicators.lows

        upper_band = [closes[i] + multiplier * atr[i] for i in range(len(atr))]
        lower_band = [closes[i] - multiplier * atr[i] for i in range(len(atr))]

        price = closes[-1]
        ema_now = ema21[-1]
        rsi_now = rsi[-1]
        k_now, d_now = stoch_k[-1], stoch_d[-1]
        k_prev, d_prev = stoch_k[-2], stoch_d[-2]

        near_lower_band = price < lower_band[-1] * 1.01
        near_upper_band = price > upper_band[-1] * 0.99

        buy_condition = (
            price > ema_now and
            rsi_now > 45 and
            k_prev < d_prev and k_now > d_now and k_now < 50
        )

        sell_condition = (
            price < ema_now and
            rsi_now < 55 and
            k_prev > d_prev and k_now < d_now and k_now > 50
        )

        logging.info(
            f"{self.symbol} - Indicadores:"
            f"\n🟢 Price: {price}"
            f"\n📈 EMA21: {ema_now}"
            f"\n📊 RSI: {rsi_now}"
            f"\n📉 Stoch K: {k_now} | D: {d_now} (prev K: {k_prev}, D: {d_prev})"
            f"\n🟩 Lower Band: {lower_band[-1]} | Upper Band: {upper_band[-1]}"
            f"\n✅ Near Lower Band: {near_lower_band}, Near Upper Band: {near_upper_band}"
            f"\n💡 Buy Cond: {buy_condition}, Sell Cond: {sell_condition}"
        )

        if buy_condition and near_lower_band:
            logging.info(f"{self.symbol} - 🎯 Sinal final: BUY")
            return 'buy'

        if sell_condition and near_upper_band:
            logging.info(f"{self.symbol} - 🎯 Sinal final: SELL")
            return 'sell'

        logging.info(f"{self.symbol} - 🚫 Sinal final: HOLD")
        return 'hold'
