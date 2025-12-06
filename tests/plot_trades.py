import asyncio
import logging
import os
import sys
from datetime import datetime

import matplotlib.dates as mdates
import numpy as np
from ccxt.async_support import hyperliquid
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from commons.enums.signal_enum import Signal
from commons.enums.timeframe_enum import TimeframeEnum
from commons.utils.config_loader import PairConfig, get_pair_by_symbol
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from strategies.cross_ema_strategy import CrossEmaStrategy


class PlotTrades:

    def __init__(self):
        pass
    
    @staticmethod
    def build_trade(ax, dates, trades):
        # Plotar entradas e saídas
        
        for trade in trades:

            index = trade.get("index")
            if index is None or index >= len(dates):
                print(f"[WARNING] Trade com índice inválido: {index}, max index = {len(dates)-1}")
                continue

            dt = dates[trade["index"]]
            price = trade["price"]

            if trade["type"] == "entry":
                ax.scatter(dt, trade["sl"], color='red', marker='v', label='SL' if trade["index"] == 0 else "", alpha=0.6)
                ax.scatter(dt, trade["tp"], color='green', marker='^', label='TP' if trade["index"] == 0 else "", alpha=0.6)

                """
                if trade["side"] == "buy":
                    ax.scatter(dt, price, marker='^', color='blue', s=100, label='Entry Buy')
                else:
                    ax.scatter(dt, price, marker='v', color='orange', s=100, label='Entry Sell')
            else:
                if trade["side"] == "buy":
                    ax.scatter(dt, price, marker='o', color='darkblue', s=100, label='Exit Buy')
                else:
                    ax.scatter(dt, price, marker='o', color='darkorange', s=100, label='Exit Sell')

            """

    @staticmethod
    def build_signals(ax, dates, signals, highs, lows):
        current_signal = None
        for i in range(len(signals)):
            index = signals[i]['index']
            signal_result = signals[i]['signal']
            dt = dates[index]
            x = mdates.date2num(dt)
            score = getattr(signal_result, 'score', None)
            signal_type = getattr(signal_result, 'signal', None)

            if signal_type != Signal.HOLD and score is not None:
                if signal_type == Signal.BUY:
                    color = 'green'
                    circle_y = lows[index] * 0.995
                    ax.scatter(x, circle_y, marker='o', color=color, s=100, label='Buy')
                    ax.vlines(x=x, ymin=lows[index], ymax=circle_y, linestyles='dashed', colors=color, linewidth=1.2, zorder=5)
                    # Texto um pouco abaixo do círculo
                    ax.text(
                        x,
                        circle_y,    # ligeiramente abaixo do círculo
                        f"Buy ({score:.2f}), Index {index}",
                        fontsize=9,
                        ha='center',
                        va='top',           # alinhado pelo topo do texto
                        color=color,
                        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                        zorder=6
                    )

                elif signal_type == Signal.SELL:
                    color = 'red'
                    circle_y = highs[index] * 1.005
                    ax.scatter(x, circle_y, marker='o', color=color, s=100, label='Sell')
                    ax.vlines(x=x, ymin=highs[index], ymax=circle_y, linestyles='dashed', colors=color, linewidth=1.2, zorder=5)
                    # Texto um pouco acima do círculo
                    ax.text(
                        x,
                        circle_y ,   # ligeiramente acima do círculo
                        f"Sell ({score:.2f}), Index {index}",
                        fontsize=9,
                        ha='center',
                        va='bottom',        # alinhado pela base do texto
                        color=color,
                        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                        zorder=6
                    )

                current_signal = signal_type


    


    @staticmethod
    def plot_trades(symbol, candles, signals, trades):
        # Preparar dados para plot
        dates = [datetime.fromtimestamp(c[0]/1000) for c in candles]
        opens = [c[1] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        closes = [c[4] for c in candles]

        fig, ax = plt.subplots(figsize=(15, 7))
        ax.set_title(f"Backtest Trades - {symbol}")

        # Plotar candles com linhas de alta/baixa
        for i in range(len(candles)):
            color = 'green' if closes[i] >= opens[i] else 'red'
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color=color)  # linha high-low
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=5)  # corpo candle

        # Plotar entradas e saídas
        PlotTrades.build_trade(ax, dates, trades)

        # Plotar pontuação (score) sobre os candles
        PlotTrades.build_signals(ax, dates, signals, highs, lows)

        # Evitar legendas repetidas
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_supertrend(ohlcv: OhlcvWrapper, symbol):
        dates = ohlcv.dates
        opens = ohlcv.opens
        highs = ohlcv.highs
        lows = ohlcv.lows
        closes = ohlcv.closes

        # Obter todos os arrays do SuperTrend
        indicatorsUtils = IndicatorsUtils(ohlcv)
        supertrend, trend, upperband, lowerband, supertrend_smooth = indicatorsUtils.supertrend()
        ema_cross_signal = CrossEmaStrategy.build_signal(indicatorsUtils, ohlcv)

        fig, ax = plt.subplots(figsize=(15, 7))
        ax.set_title(f"Backtest Trades - {symbol}")

        # Plotar candles coloridos pela tendência
        for i in range(len(closes)):
            color = 'green' if trend[i] == 1 else 'red'
            # Linha high-low
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
            # Corpo do candle
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=5)

        # SuperTrend Up / Down
        closes_arr = np.array(closes, dtype=float)
        st_up = np.where(trend == 1, supertrend, np.nan)
        st_down = np.where(trend == -1, supertrend, np.nan)
        ax.plot(dates, st_up, color='blue', label='SuperTrend Up', linewidth=2)
        ax.plot(dates, st_down, color='orange', label='SuperTrend Down', linewidth=2)

        # UpperBand / LowerBand
        ax.plot(dates, upperband, color='purple', linestyle='--', label='Upper Band', alpha=0.7)
        ax.plot(dates, lowerband, color='brown', linestyle='--', label='Lower Band', alpha=0.7)

        # SuperTrend Smooth
        ax.plot(dates, supertrend_smooth, color='cyan', linewidth=2, label='SuperTrend Smooth')

        # Evitar legendas repetidas
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()



    @staticmethod
    def plot_supertrend_with_signals(ohlcv: OhlcvWrapper, symbol, sl_pct=0.01, tp_pct=0.02):
        dates = ohlcv.dates
        timestamps = ohlcv.timestamps
        opens = ohlcv.opens
        highs = ohlcv.highs
        lows = ohlcv.lows
        closes = ohlcv.closes

        # Obter todos os arrays do SuperTrend
        indicatorsUtils = IndicatorsUtils(ohlcv)
        stop_line, trend_stop_atr  = indicatorsUtils.stop_atr_tradingview()
        supertrend, trend, upperband, lowerband, supertrend_smooth = indicatorsUtils.supertrend()
        ema_cross_signal = CrossEmaStrategy.build_signal(indicatorsUtils, ohlcv)

        
        
        psar = indicatorsUtils.psar()
        #upper, mid, lower = AISuperTrendUtils(ohlcv).indicators.bollinger_bands(period=20, std_dev=2)

        

        fig, ax = plt.subplots(figsize=(18, 7))
        ax.set_title(f"Backtest Trades - {symbol}")

        for i in range(len(closes)):
            # Cor da vela pela tendência
            color = 'green' if trend[i] == 1 else 'red'
            # Linha high-low
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
            # Corpo candle
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=5)

            # SL e TP baseados no candle
            sl = closes[i] * (1 - sl_pct) if trend[i] == 1 else closes[i] * (1 + sl_pct)
            tp = closes[i] * (1 + tp_pct) if trend[i] == 1 else closes[i] * (1 - tp_pct)
            dates_num = mdates.date2num(dates[i])
            #ax.hlines(sl, dates_num-0.2, dates_num+0.2, colors='red', linestyles='dashed', linewidth=1)
            #ax.hlines(tp, dates_num-0.2, dates_num+0.2, colors='green', linestyles='dashed', linewidth=1)

            if ema_cross_signal[i] == Signal.BUY:
                ax.scatter(dates_num, (0.985*lows[i]), color='green', s=100, zorder=5)
                ax.vlines(dates_num, ymin=lows[i], ymax=(0.985*lows[i]), linestyles='dashed', color='green', linewidth=1.2, zorder=5)
               
                ax.text(
                        dates_num,
                        (0.985*lows[i]),    # ligeiramente abaixo do círculo
                        f"Index {i}",
                        fontsize=9,
                        ha='center',
                        va='top',           # alinhado pelo topo do texto
                        color='green',
                        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                        zorder=6
                    )
            elif ema_cross_signal[i] == Signal.SELL:
                ax.scatter(dates_num, (1.015*highs[i]), color='red', s=100, zorder=5)
                ax.vlines(dates_num, ymin=highs[i], ymax=(1.015*highs[i]), linestyles='dashed', color='red', linewidth=1.2, zorder=5)

                ax.text(
                        dates_num,
                        (1.015*highs[i]),    # ligeiramente abaixo do círculo
                        f"Index {i}",
                        fontsize=9,
                        ha='center',
                        va='top',           # alinhado pelo topo do texto
                        color='red',
                        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                        zorder=6
                    )
            elif ema_cross_signal[i] == Signal.CLOSE:
                ax.scatter(dates_num, (1.015*highs[i]), color='brown', s=100, zorder=5)
                ax.vlines(dates_num, ymin=highs[i], ymax=(1.015*highs[i]), linestyles='dashed', color='brown', linewidth=1.2, zorder=5)

                ax.text(
                        dates_num,
                        (1.015*highs[i]),    # ligeiramente abaixo do círculo
                        f"Index {i}",
                        fontsize=9,
                        ha='center',
                        va='top',           # alinhado pelo topo do texto
                        color='brown',
                        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                        zorder=6
                    )
            
            # PSAR → desenha como pontos
            ax.scatter(dates_num, psar[i], color="green", s=10, label="PSAR", alpha=0.7)
        # SuperTrend Up / Down
        st_up = np.where(trend == 1, supertrend, np.nan)
        st_down = np.where(trend == -1, supertrend, np.nan)

        ax.plot(dates, st_up, color='blue', label='SuperTrend Up', linewidth=2)
        ax.plot(dates, st_down, color='orange', label='SuperTrend Down', linewidth=2)

        # UpperBand / LowerBand
        ax.plot(dates, upperband, color='purple', linestyle='--', label='Upper Band', alpha=0.7)
        ax.plot(dates, lowerband, color='brown', linestyle='--', label='Lower Band', alpha=0.7)

        # EMA 9
        ema9 = indicatorsUtils.ema(9)
        ax.plot(dates, ema9, color='orange', linestyle='-', linewidth=1.5, label='EMA 9')

        # EMA 21
        ema21 = indicatorsUtils.ema(21)
        ax.plot(dates, ema21, color='black', linestyle='-', linewidth=1.5, label='EMA 21')

        # EMA 50
        ema50 = indicatorsUtils.ema(50)
        ax.plot(dates, ema50, color='blue', linestyle='-', linewidth=1.5, label='EMA 50')

         # EMA 50
        ema200 = indicatorsUtils.ema(200)
        ax.plot(dates, ema200, color='grey', linestyle='-', linewidth=1.5, label='EMA 200')

        # Plot das Bandas de Bollinger
        #ax.plot(dates, upper, color="purple", linestyle="--", linewidth=1, label="Bollinger Upper")
        #ax.plot(dates, mid, color="gray", linestyle="--", linewidth=1, label="Bollinger Mid")
        #ax.plot(dates, lower, color="purple", linestyle="--", linewidth=1, label="Bollinger Lower")

        # Plot stop atr
        """
        ax.plot(dates, stop_line, color='orange', linestyle='--', linewidth=1.5, label='Stop ATR')
        """
        up_trend_mask = trend_stop_atr == 1
        down_trend_mask = trend_stop_atr == -1
        #ax.plot(dates, stop_line, color='green', linestyle='-', linewidth=1.5, label='Stop ATR (Uptrend)')
        """
        ax.plot(
            [dates[i] for i in range(len(stop_line)) if trend_stop_atr[i] == -1],
            [stop_line[i] for i in range(len(stop_line)) if trend_stop_atr[i] == -1],
            color='red',
            linestyle='-',
            linewidth=1.5,
            label='Stop ATR (Downtrend)'
        )
        """
        
        # Evitar legendas repetidas
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_luxalgo_supertrend_(ohlcv: OhlcvWrapper, symbol: str):
        import matplotlib.pyplot as plt
        import numpy as np

        dates = ohlcv.dates
        opens = ohlcv.opens
        highs = ohlcv.highs
        lows = ohlcv.lows
        closes = ohlcv.closes

        # Obter todos os arrays do SuperTrend
        indicatorsUtils = IndicatorsUtils(ohlcv)
        res = indicatorsUtils.luxalgo_supertrend_ai()
        ts = res["ts"]
        direction = res["direction"]
        perf_score = res["perf_score"]

        n = len(closes)

        # Inicializar TS long/short separados
        ts_long = np.zeros(n, dtype=float)
        ts_short = np.zeros(n, dtype=float)
        buy_signals = []
        sell_signals = []
        trend_change_scores = []

        for i in range(n):
            if i == 0:
                ts_long[i] = ts[i] if direction[i] == 1 else np.nan
                ts_short[i] = ts[i] if direction[i] == 0 else np.nan
            else:
                # TS contínuo para plot
                ts_long[i] = ts[i] if direction[i] == 1 else ts_long[i-1]
                ts_short[i] = ts[i] if direction[i] == 0 else ts_short[i-1]

                # Detecta mudança de tendência
                if direction[i-1] == 0 and direction[i] == 1 and closes[i] > ts[i-1]:
                    buy_signals.append((dates[i], closes[i]))
                    trend_change_scores.append((dates[i], perf_score[i]))
                elif direction[i-1] == 1 and direction[i] == 0 and closes[i] < ts[i-1]:
                    sell_signals.append((dates[i], closes[i]))
                    trend_change_scores.append((dates[i], perf_score[i]))

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(18, 7))
        ax.set_title(f"LuxAlgo SuperTrend - {symbol}")

        # Plot candles simples
        for i in range(n):
            color = 'green' if direction[i] == 1 else 'red'
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=5)

        
        # Plot TS long/short
        ax.plot(dates, ts_long, color='green', linewidth=2, label='TS Long')
        ax.plot(dates, ts_short, color='red', linewidth=2, label='TS Short')
        """
        # Plot sinais
        if buy_signals:
            bx, by = zip(*buy_signals)
            ax.scatter(bx, by, marker='^', color='lime', s=100, label='BUY')
        if sell_signals:
            sx, sy = zip(*sell_signals)
            ax.scatter(sx, sy, marker='v', color='orange', s=100, label='SELL')

        # Plot pontuação na mudança de tendência
        for dt, score in trend_change_scores:
            ax.text(dt, score, str(score), color='blue', fontsize=10, ha='center', va='bottom')
        """
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_luxalgo_supertrend(ohlcv: OhlcvWrapper, symbol: str):
        dates = ohlcv.dates
        opens = ohlcv.opens
        highs = ohlcv.highs
        lows = ohlcv.lows
        closes = ohlcv.closes

        # --- Obter arrays do SuperTrend AI ---
        indicatorsUtils = IndicatorsUtils(ohlcv)
        res = indicatorsUtils.luxalgo_supertrend_ai()
        ts = res["ts"]
        direction = res["direction"]  # 1 bullish, 0 bearish
        perf_score = res["perf_score"]
        #print("AQUIII", res)
        n = len(closes)

        # --- Inicializar arrays para TS colorido ---
        ts_long = np.full(n, np.nan)
        ts_short = np.full(n, np.nan)

        # --- Identificar sinais de BUY/SELL ---
        buy_signals = []
        sell_signals = []
        trend_change_scores = []  # para armazenar (data, score)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(18, 7))
        ax.set_title(f"SuperTrend AI - {symbol}")

        for i in range(1, n):
            dates_num = mdates.date2num(dates[i])

            # BUY: direção muda de 0 -> 1 e fechamento acima do TS anterior
            if direction[i-1] == 0 and direction[i] == 1 and closes[i] > ts[i-1]:

                ax.scatter(dates_num, (0.985*lows[i]), color='green', s=100, zorder=5)
                ax.vlines(dates_num, ymin=lows[i], ymax=(0.985*lows[i]), linestyles='dashed', color='green', linewidth=1.2, zorder=5)
               
                ax.text(
                        dates_num,
                        (0.985*lows[i]),    # ligeiramente abaixo do círculo
                        f"Index {i}, {perf_score[i]}",
                        fontsize=9,
                        ha='center',
                        va='top',           # alinhado pelo topo do texto
                        color='green',
                        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                        zorder=6
                    )

                #buy_signals.append((dates[i], closes[i]))
                trend_change_scores.append((dates[i], perf_score[i]))
            # SELL: direção muda de 1 -> 0 e fechamento abaixo do TS anterior
            elif direction[i-1] == 1 and direction[i] == 0 and closes[i] < ts[i-1]:
                #sell_signals.append((dates[i], closes[i]))
                trend_change_scores.append((dates[i], perf_score[i]))

                ax.scatter(dates_num, (1.015*highs[i]), color='red', s=100, zorder=5)
                ax.vlines(dates_num, ymin=highs[i], ymax=(1.015*highs[i]), linestyles='dashed', color='red', linewidth=1.2, zorder=5)

                ax.text(
                        dates_num,
                        (1.015*highs[i]),    # ligeiramente abaixo do círculo
                        f"Index {i}, {perf_score[i]}",
                        fontsize=9,
                        ha='center',
                        va='top',           # alinhado pelo topo do texto
                        color='red',
                        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                        zorder=6
                    )
            
            # Trailing stop para plot contínuo
            ts_long[i] = ts[i] if direction[i] == 1 else ts_long[i-1]
            ts_short[i] = ts[i] if direction[i] == 0 else ts_short[i-1]

        # Candles
        for i in range(n):
            color = 'green' if direction[i] == 1 else 'red'
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)  # high-low
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=5)   # body

        #for dt, score in trend_change_scores:
        #    ax.text(dt, score, str(score), color='blue', fontsize=10, ha='center', va='bottom')

        # SuperTrend contínuo colorido

        
        ax.plot(dates, ts_long, color='blue', linewidth=1, label='TS Long')
        ax.plot(dates, ts_short, color='orange', linewidth=1, label='TS Short')

        # Pontos de sinais
        """
        if buy_signals:
            buy_x, buy_y = zip(*buy_signals)
            ax.scatter(buy_x, buy_y, marker='^', color='green', s=100, label='BUY')
        if sell_signals:
            sell_x, sell_y = zip(*sell_signals)
            ax.scatter(sell_x, sell_y, marker='v', color='red', s=100, label='SELL')
        """
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.xticks(rotation=45)
        ax.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_band_segments(ax, dates, band, trend, color_up, color_down, linestyle='--'):
        segments = []
        colors = []

        for i in range(1, len(dates)):
            x = [mdates.date2num(dates[i-1]), mdates.date2num(dates[i])]
            y = [band[i-1], band[i]]
            segments.append(list(zip(x, y)))
            colors.append(color_up if trend[i-1] == 1 else color_down)

        lc = LineCollection(segments, colors=colors, linestyles=linestyle, alpha=0.7, linewidths=2)
        ax.add_collection(lc)
    

    @staticmethod
    async def get_historical_ohlcv(pair: PairConfig, timeframe: TimeframeEnum, limit: int = 50) -> OhlcvWrapper:

        # Configura sua exchange Hyperliquid
        exchange =  hyperliquid({
                "enableRateLimit": True,
                "testnet": False,
            }) # type: ignore

        try:
            # Busca candles OHLCV históricos (timestamp, open, high, low, close, volume)
            
            ohlcv =  await exchange.fetch_ohlcv(pair.symbol, timeframe, limit=limit)
            return OhlcvWrapper(ohlcv)
        finally:
            await exchange.close()

# Execução principal
async def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    pair = get_pair_by_symbol("BTC/USDC:USDC")

    if pair:

        ohlcv = await PlotTrades.get_historical_ohlcv(pair, TimeframeEnum.M15, 750)

        PlotTrades.plot_supertrend_with_signals(ohlcv, pair.symbol)

if __name__ == "__main__":
    asyncio.run(main())