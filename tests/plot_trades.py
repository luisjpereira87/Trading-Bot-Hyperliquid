import asyncio
import logging
import os
import sys
from datetime import datetime

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from ccxt.async_support import hyperliquid
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from commons.enums.signal_enum import Signal
from commons.enums.timeframe_enum import TimeframeEnum
from commons.utils.config_loader import PairConfig, get_pair_by_symbol
from commons.utils.indicators.base_indicators_utils import BaseIndicatorsUtils
from commons.utils.indicators.custom_indicators_utils import \
    CustomIndicatorsUtils
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.indicators.tv_indicators_utils import TvIndicatorsUtils
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
        supertrend, trend, upperband, lowerband, supertrend_smooth, direction,_ = indicatorsUtils.supertrend()
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
        supertrend, trend, upperband, lowerband, supertrend_smooth, direction, perf_score = indicatorsUtils.supertrend()
        ema_cross_signal = CrossEmaStrategy.build_signal(indicatorsUtils, ohlcv)

        psar = indicatorsUtils.psar()

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
    def plot_luxalgo_supertrend(ohlcv: OhlcvWrapper, symbol: str):
        dates = ohlcv.dates
        opens = ohlcv.opens
        highs = ohlcv.highs
        lows = ohlcv.lows
        closes = ohlcv.closes

        # --- Obter arrays do SuperTrend AI ---
        indicatorsUtils = TvIndicatorsUtils(ohlcv)
        supertrend = indicatorsUtils.supertrend_ai()
        direction, rsi = indicatorsUtils.market_structure_rsi()

        print("AQUIII", direction)
        ts = supertrend.ts 
        perf_ama = supertrend.perf_ama
        direction = supertrend.direction
        perf_score = supertrend.score
        #ts = res["ts"]
        #direction = res["direction"]  # 1 bullish, 0 bearish
        #perf_score = res["perf_score"]
        n = len(closes)

        #print("AQUIII", direction)

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
            if direction[i-1] == -1 and direction[i] == 1 and closes[i] > ts[i-1]:

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
            elif direction[i-1] == 1 and direction[i] == -1 and closes[i] < ts[i-1]:
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
            ts_short[i] = ts[i] if direction[i] == -1 else ts_short[i-1]

        # Candles
        for i in range(n):
            color = 'green' if direction[i] == 1 else 'red'
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)  # high-low
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=5)   # body

        #for dt, score in trend_change_scores:
        #    ax.text(dt, score, str(score), color='blue', fontsize=10, ha='center', va='bottom')

        # SuperTrend contínuo colorido

        
        #ax.plot(dates, ts_long, color='blue', linewidth=1, label='TS Long')
        ax.plot(dates, ts, color='orange', linewidth=1, label='TS Short')

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
    def plot_custom_supertrend(ohlcv: OhlcvWrapper, symbol: str):
        dates = ohlcv.dates
        opens = ohlcv.opens
        highs = ohlcv.highs
        lows = ohlcv.lows
        closes = ohlcv.closes

        # --- Obter arrays do SuperTrend AI ---
        indicatorsUtils = IndicatorsUtils(ohlcv)
        supertrend, trend, final_upperband, final_lowerband, supertrend_smooth, direction, perf_score = indicatorsUtils.supertrend()
        #ts = res["ts"]
        #direction = res["direction"]  # 1 bullish, 0 bearish
        #perf_score = res["perf_score"]
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
            if direction[i-1] == -1 and direction[i] == 1:

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
            elif direction[i-1] == 1 and direction[i] == -1:
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
            #ts_long[i] = ts[i] if direction[i] == 1 else ts_long[i-1]
            #ts_short[i] = ts[i] if direction[i] == 0 else ts_short[i-1]

        # Candles
        for i in range(n):
            color = 'green' if direction[i] == 1 else 'red'
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)  # high-low
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=5)   # body

        #for dt, score in trend_change_scores:
        #    ax.text(dt, score, str(score), color='blue', fontsize=10, ha='center', va='bottom')

        # SuperTrend contínuo colorido

        
        ax.plot(dates, final_upperband, color='blue', linewidth=1, label='TS Long')
        ax.plot(dates, final_lowerband, color='orange', linewidth=1, label='TS Short')

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
    def plot_volumatic_vidya(ohlcv: OhlcvWrapper, symbol: str):

        dates = ohlcv.dates
        opens = ohlcv.opens
        highs = ohlcv.highs
        lows = ohlcv.lows
        closes = ohlcv.closes

        indicatorsUtils = TvIndicatorsUtils(ohlcv)
        res = indicatorsUtils.volumatic_vidya(34)

        vidya = res.vidya
        upper = res.upper_band
        lower = res.lower_band
        smoothed = res.smoothed
        pivot_high = np.array(res.pivot_high, dtype=float)
        pivot_low = np.array(res.pivot_low, dtype=float)
        delta_vol = res.delta_volume_pct
        is_trend_up = res.is_trend_up
        dates_array = np.array(dates)
        n = len(closes)

        # ------------- FIGURE -------------
        fig = plt.subplots(figsize=(18, 9))
        fig = plt.figure(figsize=(18, 9))

        # --------- PRICE + INDICATOR AX ---------
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_title(f"Volumatic VIDYA + ATR Bands — {symbol}")

        # ---- CANDLE PLOT (mesmo estilo do teu supertrend) ----
        for i in range(n):
            dates_num = mdates.date2num(dates[i])
            color = "green" if closes[i] >= opens[i] else "red"

            # hilo
            ax1.plot([dates[i], dates[i]], [lows[i], highs[i]],
                    color="black", linewidth=1)

            # corpo
            ax1.plot([dates[i], dates[i]], [opens[i], closes[i]],
                    color=color, linewidth=4)

        # ----------- VIDYA + BANDS -----------
        ax1.plot(dates, vidya, label="VIDYA", color="blue", linewidth=1.2)
        ax1.plot(dates, upper, label="Upper Band", color="orange", linewidth=1)
        ax1.plot(dates, lower, label="Lower Band", color="purple", linewidth=1)

        # ----------- SMOOTHED SWITCH LINE -----------
        ax1.plot(dates, smoothed, label="Smoothed Trend Line",
                color="black", linewidth=1.3, linestyle="--")

        # ----------- PIVOTS -----------
        ph_y = np.where(pivot_high == 1, highs, np.nan)
        pl_y = np.where(pivot_low == 1, lows, np.nan)

        ax1.scatter(dates_array, ph_y, color="red", s=60, label="Pivot High", zorder=5)
        ax1.scatter(dates_array, pl_y, color="green", s=60, label="Pivot Low", zorder=5)

        # ----------- TREND BACKGROUND COLOR -----------
        for i in range(1, n):
            if is_trend_up[i]:
                ax1.axvspan(dates_array[i-1], dates_array[i], color="green", alpha=0.03)
            else:
                ax1.axvspan(dates_array[i-1], dates_array[i], color="red", alpha=0.03)

        ax1.grid(True)
        ax1.legend()
        ax1.set_ylabel("Price")
        plt.xticks(rotation=45)

        # ----------- DELTA VOLUME SUBPLOT -----------
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        ax2.set_title("Delta Volume % (buyers - sellers)")
        ax2.plot(dates, delta_vol, linewidth=1.2)

        ax2.axhline(0, linestyle="--", linewidth=1)
        ax2.set_ylabel("%")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def plot_smart_money_flow_cloud(ohlcv: OhlcvWrapper, symbol: str):
        dates = ohlcv.dates
        opens = ohlcv.opens
        highs = ohlcv.highs
        lows = ohlcv.lows
        closes = ohlcv.closes
        volumes = ohlcv.volumes
        volumes_log = np.log1p(pd.Series(volumes))

        # --- Obter arrays do SuperTrend AI ---
        indicatorsUtils = TvIndicatorsUtils(ohlcv)
        #supertrend = indicatorsUtils.supertrend_ai()
        res = indicatorsUtils.smart_money_flow_cloud()
        signal = res['signal']
        strength = res['strength']
        upper =  res['upper']
        lower =  res['lower']
        bull_retest = res['bull_retest']
        bear_retest = res['bear_retest']
        basis = res['basis']

        dii = indicatorsUtils.directional_imbalance_index()
        dii_upper = dii['upper']
        dii_lower = dii['lower']
        dii_up_count = dii['up_count']
        dii_down_count = dii['down_count']
        dii_bulls_perc = dii['bulls_perc']
        dii_bears_perc = dii['bears_perc']

        n = len(closes)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(18, 7))
        ax.set_title(f"SuperTrend AI - {symbol}")

        for i in range(1, n):
            dates_num = mdates.date2num(dates[i])

            if ((signal[i-1] == -1 and signal[i] == 1) or bull_retest[i]):

                ax.scatter(dates_num, (0.985*lows[i]), color='green', s=100, zorder=5)
                ax.vlines(dates_num, ymin=lows[i], ymax=(0.985*lows[i]), linestyles='dashed', color='green', linewidth=1.2, zorder=5)
               
                ax.text(
                        dates_num,
                        (0.985*lows[i]),    # ligeiramente abaixo do círculo
                        f"Index {i}, {round(strength[i],2)}",
                        fontsize=9,
                        ha='center',
                        va='top',           # alinhado pelo topo do texto
                        color='green',
                        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                        zorder=6
                    )
            # SELL: direção muda de 1 -> 0 e fechamento abaixo do TS anterior
            elif ((signal[i-1] == 1 and signal[i] == -1) or bear_retest[i]):

                ax.scatter(dates_num, (1.015*highs[i]), color='red', s=100, zorder=5)
                ax.vlines(dates_num, ymin=highs[i], ymax=(1.015*highs[i]), linestyles='dashed', color='red', linewidth=1.2, zorder=5)

                ax.text(
                        dates_num,
                        (1.015*highs[i]),    # ligeiramente abaixo do círculo
                        f"Index {i}, {round(strength[i],2)}",
                        fontsize=9,
                        ha='center',
                        va='top',           # alinhado pelo topo do texto
                        color='red',
                        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                        zorder=6
                    )

        # Candles
        for i in range(n):
            color = 'green' if signal[i] == 1 else 'red'
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)  # high-low
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=5)   # body

                # ----------- VIDYA + BANDS -----------
        ax.plot(dates, upper, label="Upper Band", color="orange", linewidth=1)
        ax.plot(dates, lower, label="Lower Band", color="purple", linewidth=1)
        ax.plot(dates, basis, label="Middle Band", color="red", linewidth=1)

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.xticks(rotation=45)
        ax.legend()
        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def plot_market_structure_rsi(ohlcv: OhlcvWrapper, symbol: str):
        dates = ohlcv.dates
        opens = ohlcv.opens
        highs = ohlcv.highs
        lows = ohlcv.lows
        closes = ohlcv.closes

        # --- Obter arrays do SuperTrend AI ---
        indicatorsUtils = TvIndicatorsUtils(ohlcv)
        #supertrend = indicatorsUtils.supertrend_ai()
        direction = indicatorsUtils.market_structure_rsi()

        #ts = res["ts"]
        #direction = res["direction"]  # 1 bullish, 0 bearish
        #perf_score = res["perf_score"]
        n = len(closes)

        #print("AQUIII", direction)

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
            print("AQUII", i, direction[i])
            # BUY: direção muda de 0 -> 1 e fechamento acima do TS anterior
            if direction[i] == 1:

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
            # SELL: direção muda de 1 -> 0 e fechamento abaixo do TS anterior
            elif direction[i] == -1 :

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

        # Candles
        for i in range(n):
            color = 'green' if direction[i] == 1 else 'red'
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)  # high-low
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=5)   # body

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.xticks(rotation=45)
        ax.legend()
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_smart_money_breakout(ohlcv, symbol):
        """
        Plota o Smart Money Breakout Channels com retângulos que cobrem
        desde o início da acumulação até o breakout.
        """
        dates = ohlcv.dates
        opens = ohlcv.opens
        highs = ohlcv.highs
        lows = ohlcv.lows
        closes = ohlcv.closes
        n = len(closes)

        # Obter resultados do indicador
        indicatorsUtils = TvIndicatorsUtils(ohlcv)
        res = indicatorsUtils.smart_money_breakout_channels()

        top = res["top"]
        bottom = res["bottom"]
        new_channel = res["new_channel"]
        bull_break = res["bull_break"]
        bear_break = res["bear_break"]
        duration = res["duration"] # Importante: deve vir do retorno da função

        fig, ax = plt.subplots(figsize=(18, 7))
        ax.set_title(f"Smart Money Breakout Channels - {symbol}")
        ax.set_facecolor('#ffffff')

        # --- 1. Desenho de Candles ---
        for i in range(n):
            color = '#00ffbb' if closes[i] >= opens[i] else '#ff1100'
            # Pavio (High-Low)
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=0.8, zorder=3)
            # Corpo (Open-Close)
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=4, solid_capstyle='butt', zorder=4)

        # --- 2. Desenho das Caixas (Canais) ---
        for i in range(n):
            if new_channel[i]:
                # Ponto de início: barra atual menos a duração da acumulação
                d = int(duration[i])
                start_idx = max(0, i - d)
                
                # Ponto de fim: procurar o primeiro breakout após o índice i
                end_idx = n - 1
                for j in range(i, n):
                    if bull_break[j] or bear_break[j]:
                        end_idx = j
                        break
                
                # Converter datas para o formato do Matplotlib
                x0 = mdates.date2num(dates[start_idx])
                x1 = mdates.date2num(dates[end_idx])
                width = x1 - x0
                
                # Altura baseada nos valores capturados no nascimento do canal
                box_top = top[i]
                box_bottom = bottom[i]
                height = box_top - box_bottom
                
                if not np.isnan(box_top) and not np.isnan(box_bottom):
                    # Canal Principal (Azul Translúcido)
                    rect = Rectangle((x0, box_bottom), width, height,
                                    facecolor='#4A90E2', edgecolor='#1f4e79', 
                                    alpha=0.12, linewidth=0.7, zorder=1)
                    ax.add_patch(rect)
                
                    # Margem visual para Supply/Demand (15% da altura)
                    margin = height * 0.15
                    
                    # Supply Zone (Topo - Vermelho)
                    ax.add_patch(Rectangle((x0, box_top - margin), width, margin,
                                        facecolor='#ff1100', alpha=0.18, linewidth=0, zorder=2))

                    # Demand Zone (Base - Verde)
                    ax.add_patch(Rectangle((x0, box_bottom), width, margin,
                                        facecolor='#00ffbb', alpha=0.18, linewidth=0, zorder=2))

        # --- 3. Sinais de Breakout (Setas) ---
        for i in range(n):
            if bull_break[i]:
                ax.text(dates[i], highs[i], "▲", color='#00a67d', fontsize=14, 
                        ha='center', va='bottom', fontweight='bold', zorder=5)
            if bear_break[i]:
                ax.text(dates[i], lows[i], "▼", color='#d91e18', fontsize=14, 
                        ha='center', va='top', fontweight='bold', zorder=5)

        # --- 4. Formatação de Eixos ---
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=30)
        ax.grid(True, linestyle=':', alpha=0.4)
        
        # Ajuste dinâmico de escala Y
        y_min = np.min(lows) * 0.998
        y_max = np.max(highs) * 1.002
        ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_sha_macd_raw_signals(ohlcv, symbol: str):
        dates = ohlcv.dates
        opens, highs, lows, closes = ohlcv.opens, ohlcv.highs, ohlcv.lows, ohlcv.closes
        n = len(closes)

        # 1. Indicadores
        indicatorsUtils = TvIndicatorsUtils(ohlcv)
        # SHA para cores das velas
        trend_sha, o2, h2, l2, c2 = indicatorsUtils.smoothed_heikin_ashi(len1=34, len2=34)
        # MACD para os sinais em bruto
        ha_o, ha_c, signal, _ = indicatorsUtils.standardized_macd_ha(fast=12, slow=26, sig_len=9)

        # 2. Configuração base
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        dates_num = mdates.date2num(dates)
        dates_array = np.array(dates)
        delta = dates_array[1] - dates_array[0] if n > 1 else 0.0005
        width = delta * 0.8

        # --- QUADRO 1: VELAS DE PREÇO (Cor baseada no SHA) ---
        for i in range(n):
            sha_color = 'lime' if c2[i] > o2[i] else 'red'
            
            ax1.plot([dates_array[i], dates_array[i]], [lows[i], highs[i]], color='black', linewidth=0.8, zorder=2)
            bottom, height = min(opens[i], closes[i]), abs(closes[i] - opens[i])
            ax1.bar(dates_array[i], max(height, 0.001), bottom=bottom, width=width, 
                    color=sha_color, edgecolor='black', linewidth=0.3, zorder=3)

        # --- QUADRO 2: MACD COM BANDAS ---
        ax2.axhline(100, color='red', linestyle='--', alpha=0.3)
        ax2.axhline(-100, color='cyan', linestyle='--', alpha=0.3)
        ax2.axhline(0, color='gray', linewidth=0.8, alpha=0.5)

        for i in range(n):
            if np.isnan(ha_o[i]): continue
            macd_col = '#00bcd4' if ha_c[i] > ha_o[i] else '#fc1f1f'
            m_bottom, m_height = min(ha_o[i], ha_c[i]), abs(ha_c[i] - ha_o[i])
            ax2.bar(dates_array[i], max(m_height, 0.1), bottom=m_bottom, width=width, 
                    color=macd_col, alpha=0.9, zorder=2)

        ax2.plot(dates_num, signal, color='black', linewidth=1.2, alpha=0.7, zorder=4)

        # --- SINAIS EM BRUTO (Sem filtro de tendência) ---
        for i in range(1, n):
            # Cruzamento de ALTA em bruto
            if ha_c[i] > signal[i] and ha_c[i-1] <= signal[i-1]:
                # Marcador verde em ambos os quadros
                ax1.scatter(dates_array[i], (lows[i] * 0.99), color='green', marker='^', s=100, zorder=10)
                ax2.scatter(dates_array[i], signal[i], color='green', marker='o', s=40, zorder=10)
            
            # Cruzamento de BAIXA em bruto
            elif ha_c[i] < signal[i] and ha_c[i-1] >= signal[i-1]:
                # Marcador vermelho em ambos os quadros
                ax1.scatter(dates_array[i], (highs[i] * 1.01), color='red', marker='v', s=100, zorder=10)
                ax2.scatter(dates_array[i], signal[i], color='red', marker='o', s=40, zorder=10)

        # Estética Final
        ax1.set_title(f"{symbol} - Preço (Trend Color) + MACD Raw Signals")
        ax1.grid(True, alpha=0.1)
        ax2.grid(True, alpha=0.1)
        ax2.set_ylim(-250, 250)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_double_rsi_signals(ohlcv, symbol: str):
        dates = ohlcv.dates
        opens, highs, lows, closes = ohlcv.opens, ohlcv.highs, ohlcv.lows, ohlcv.closes
        n = len(closes)
        dates_array = np.array(dates)

        # 1. Indicadores
        indicatorsUtils = TvIndicatorsUtils(ohlcv)
        
        
        # Obtemos o novo Double RSI que criámos (certifica-te que o método retorna este dicionário)
        # Se o teu método atual só retorna a lista de sinais, ajusta-o para retornar (signals, fast, slow)
        double_rsi_data = indicatorsUtils.double_rsi_() 
        signals = double_rsi_data['signals']
        rsi_fast = double_rsi_data['rsi_fast']
        rsi_slow = double_rsi_data['rsi_slow']
        ema25 = double_rsi_data['ema25']
        ema50 = double_rsi_data['ema50']
        ema100 = double_rsi_data['ema100']


        # 2. Configuração base
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, 
                                    gridspec_kw={'height_ratios': [2, 1]})
        
        dates_num = mdates.date2num(dates)
        width = (dates_array[1] - dates_array[0]) * 0.8 if n > 1 else 0.1

        # --- QUADRO 1: VELAS DE PREÇO ---
        for i in range(n):
            # Cor simples para o preço, ou usa a tua lógica de Heikin Ashi aqui
            color = 'lime' if closes[i] >= opens[i] else 'red'
            ax1.plot([dates_array[i], dates_array[i]], [lows[i], highs[i]], color='black', linewidth=0.8)
            ax1.bar(dates_array[i], abs(closes[i] - opens[i]), bottom=min(opens[i], closes[i]), 
                    width=width, color=color, edgecolor='black', linewidth=0.3)

        # EMAS    
        ax1.plot(dates, ema25, label="Ema 25", color="green", linewidth=1)
        ax1.plot(dates, ema50, label="Ema 50", color="orange", linewidth=1)
        ax1.plot(dates, ema100, label="Ema 50", color="red", linewidth=1)

        # --- QUADRO 2: DOUBLE RSI ---
        # Linhas de referência de Sobrecompra / Sobrevenda
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (75)')
        ax2.axhline(30, color='blue', linestyle='--', alpha=0.5, label='Oversold (25)')
        ax2.axhline(50, color='gray', linestyle='-', alpha=0.2)

        # Plot das linhas RSI
        ax2.plot(dates_num, rsi_slow, color='black', linewidth=1.5, label='RSI 21 (Slow)', alpha=0.8)
        ax2.plot(dates_num, rsi_fast, color='orange', linewidth=1.2, label='RSI 5 (Fast)', alpha=0.9)
        #ax2.plot(dates_num, rsi, color='red', linewidth=1.2, label='RSI 14', alpha=0.9)

        # --- SINAIS DE ENTRADA (Sincronizados com o Gráfico) ---
        for i in range(1, n):
            # Sinal de COMPRA (Verde)
            if signals[i] == Signal.BUY:
                ax1.scatter(dates_array[i], (lows[i] * 0.98), color='green', marker='^', s=150, edgecolors='black', zorder=10)
                ax2.scatter(dates_array[i], rsi_fast[i], color='green', marker='o', s=60, edgecolors='black', zorder=10)
            
            # Sinal de VENDA (Vermelho)
            elif signals[i] == Signal.SELL:
                ax1.scatter(dates_array[i], (highs[i] * 1.02), color='red', marker='v', s=150, edgecolors='black', zorder=10)
                ax2.scatter(dates_array[i], rsi_fast[i], color='red', marker='o', s=60, edgecolors='black', zorder=10)

        # Estética Final
        ax1.set_title(f"{symbol} - Preço + Sinais Double RSI (Scalping Setup)")
        ax1.grid(True, alpha=0.1)
        ax2.grid(True, alpha=0.1)
        ax2.set_ylim(0, 100) # RSI é sempre 0-100
        ax2.legend(loc='upper right')
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_rsi_trendline_full_visual(ohlcv, symbol: str):
        opens, highs, lows, closes = ohlcv.opens, ohlcv.highs, ohlcv.lows, ohlcv.closes
        n = len(closes)
        indices = np.arange(n) # Usamos índices simples 0, 1, 2...

        # 1. Indicadores
        indicatorsUtils = TvIndicatorsUtils(ohlcv)
        rsi6 = indicatorsUtils.rsi(8)
        bb_upper, _, bb_lower = indicatorsUtils.bollinger_bands()
        
        signals_buy_idx, signals_buy_val = [], []
        signals_sell_idx, signals_sell_val = [], []
        trendlines = [] 

        # --- LÓGICA DE DETEÇÃO (Mantida igual) ---
        for i in range(25, n):
            # 1. Identificar se estamos em zona de exaustão
            rsi_slice = np.array(rsi6[i-10:i])
            rsi_overbought = any(val > 70 for val in rsi_slice)
            rsi_oversold = any(val < 30 for val in rsi_slice)

            # --- LÓGICA DE VENDA (SHORT) ---
            if highs[i] >= bb_upper[i] and rsi_overbought:
                # O "degrau": procuramos o valor mais baixo do RSI entre o pico e agora
                # Se o RSI atual quebrar esse valor, a "linha de tendência" foi rompida
                last_low_rsi = min(rsi6[i-4:i]) 
                
                if rsi6[i] < last_low_rsi and rsi6[i-1] >= last_low_rsi:
                    signals_sell_idx.append(i)
                    signals_sell_val.append(highs[i] * 1.002)
                    # Desenha uma linha horizontal curta para mostrar o "rompimento"
                    trendlines.append(((i-4, i), (last_low_rsi, last_low_rsi), 'red'))

            # --- LÓGICA DE COMPRA (LONG) ---
            elif lows[i] <= bb_lower[i] and rsi_oversold:
                # O "degrau": procuramos o valor mais alto do RSI entre o fundo e agora
                last_high_rsi = max(rsi6[i-4:i])
                
                if rsi6[i] > last_high_rsi and rsi6[i-1] <= last_high_rsi:
                    signals_buy_idx.append(i)
                    signals_buy_val.append(lows[i] * 0.998)
                    # Desenha uma linha horizontal curta para mostrar o "rompimento"
                    trendlines.append(((i-4, i), (last_high_rsi, last_high_rsi), 'green'))

        # --- PLOTAGEM ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # 2. Desenho Manual dos Candles
        width = 0.6 # Largura fixa para índices inteiros
        for i in range(n):
            color = '#26a69a' if closes[i] >= opens[i] else '#ef5350' # Cores TradingView
            # Pavio (High/Low)
            ax1.plot([i, i], [lows[i], highs[i]], color='white', linewidth=1, alpha=0.6)
            # Corpo (Open/Close)
            ax1.bar(i, abs(closes[i] - opens[i]), bottom=min(opens[i], closes[i]), 
                    width=width, color=color, edgecolor=color, linewidth=0.1, zorder=3)

        # 3. Bandas e Sinais
        ax1.plot(indices, bb_upper, color='royalblue', alpha=0.3, label='BB Upper')
        ax1.plot(indices, bb_lower, color='royalblue', alpha=0.3, label='BB Lower')
        ax1.scatter(signals_buy_idx, signals_buy_val, marker='^', color='lime', s=150, zorder=5, label='BUY')
        ax1.scatter(signals_sell_idx, signals_sell_val, marker='v', color='red', s=150, zorder=5, label='SELL')
        
        ax1.set_facecolor('#131722')
        ax1.grid(True, color='#2a2e39', alpha=0.5)
        ax1.legend()

        # 4. RSI e Trendlines
        ax2.plot(indices, rsi6, color='#9b59b6', linewidth=2)
        ax2.axhline(70, color='#ef5350', linestyle='--', alpha=0.3)
        ax2.axhline(30, color='#26a69a', linestyle='--', alpha=0.3)
        
        for coords_x, coords_y, col in trendlines:
            ax2.plot(coords_x, coords_y, color=col, linestyle='-', linewidth=2, alpha=0.9)

        ax2.set_facecolor('#131722')
        ax2.grid(True, color='#2a2e39', alpha=0.5)
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_double_bb_rsi_logic(ohlcv, symbol: str):
        opens, highs, lows, closes = ohlcv.opens, ohlcv.highs, ohlcv.lows, ohlcv.closes
        n = len(closes)
        indices = np.arange(n)

        # 1. Indicadores
        indicatorsUtils = CustomIndicatorsUtils(ohlcv)
        rsi, rsi_ema = indicatorsUtils.rsi(14)

        double_bb = indicatorsUtils.double_bb_rsi_logic()
        bb20_up = double_bb['bbshort_up']
        bb20_low = double_bb['bbshort_low']
        bb20_mid = double_bb['bbshort_mid']
        bb80_up = double_bb['bblong_up']
        bb80_low = double_bb['bblong_low']
        signals = double_bb['signals']
        entry_buy_idx = double_bb['entry_buy_idx']
        entry_buy_val = double_bb['entry_buy_val']
        entry_sell_idx = double_bb['entry_sell_idx']
        entry_sell_val = double_bb['entry_sell_val']

        context_buy_idx = double_bb['context_buy_idx']
        context_buy_val = double_bb['context_buy_val']
        context_sell_idx = double_bb['context_sell_idx']
        context_sell_val = double_bb['context_sell_val']

        super_score, ema_score = indicatorsUtils.calculate_super_score()

        bull_pct, bear_pct, slope_ma_14, polarity_osc, signal_line = indicatorsUtils.calculate_lux_bb_oscillator()

        # --- PLOTAGEM ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # Desenho dos Candles (Simplificado)
        width = 0.6
        for i in range(n):
            color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
            ax1.plot([i, i], [lows[i], highs[i]], color='white', linewidth=1, alpha=0.4)
            ax1.bar(i, abs(closes[i] - opens[i]), bottom=min(opens[i], closes[i]), width=width, color=color, zorder=3)

        # Plot das duas Bandas
        ax1.plot(indices, bb20_up, color='cyan', alpha=0.3, label='BB 20 (15m)')
        ax1.plot(indices, bb20_low, color='cyan', alpha=0.3)
        ax1.plot(indices, bb20_mid, color='cyan', alpha=0.3)
        ax1.plot(indices, bb80_up, color='orange', linewidth=1.5, alpha=0.6, label='BB 80 (1h Projection)')
        ax1.plot(indices, bb80_low, color='orange', linewidth=1.5, alpha=0.6)
        
        # Preenchimento entre as bandas da 1h (zona de exaustão extrema)
        ax1.fill_between(indices, bb80_low, bb80_up, color='orange', alpha=0.03)

        ax1.scatter(context_buy_idx, context_buy_val, marker='^', color='lime', s=150, zorder=5, label='CONFLUENCE BUY')
        ax1.scatter(context_sell_idx, context_sell_val, marker='v', color='red', s=150, zorder=5, label='CONFLUENCE SELL')

        for idx, val in zip(context_buy_idx, context_buy_val):
            ax1.text(
                idx, 
                val, 
                f"Idx {idx}", # Usamos o idx real do ponto
                fontsize=8,
                ha='center',
                va='top',
                color='darkgreen',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'),
                zorder=6
            )

        for idx, val in zip(context_sell_idx, context_sell_val):
            ax1.text(
                idx, 
                val, 
                f"Idx {idx}", 
                fontsize=8,
                ha='center',
                va='bottom', # Para o sell, talvez seja melhor 'bottom' para ficar acima da seta
                color='darkred',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'),
                zorder=6
            )

        # 2. Plot dos Círculos (Entrada Real/Gatilho) - Símbolo sólido e marcante
        ax1.scatter(entry_buy_idx, entry_buy_val, marker='o', color='#00ff00', 
                    s=150, edgecolors='black', linewidth=1.5, label='ENTRADA (Evolução)')
        ax1.scatter(entry_sell_idx, entry_sell_val, marker='o', color='#ff0000', 
                    s=150, edgecolors='black', linewidth=1.5)
        
        for idx, val in zip(entry_buy_idx, entry_buy_val):
            ax1.text(
                idx, 
                val, 
                f"Idx {idx}", # Usamos o idx real do ponto
                fontsize=8,
                ha='center',
                va='top',
                color='darkgreen',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'),
                zorder=6
            )

        for idx, val in zip(entry_sell_idx, context_sell_val):
            ax1.text(
                idx, 
                val, 
                f"Idx {idx}", 
                fontsize=8,
                ha='center',
                va='bottom', # Para o sell, talvez seja melhor 'bottom' para ficar acima da seta
                color='darkred',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'),
                zorder=6
            )

        ax1.set_facecolor('#131722')
        ax1.grid(True, color='#2a2e39', alpha=0.5)
        ax1.legend()

        # RSI 6
        """
        ax2.plot(indices, rsi, color='#9b59b6', linewidth=2)
        ax2.plot(indices, rsi_ema, color="#368921", linewidth=2)
        ax2.axhline(70, color='red', linestyle='--', alpha=0.3)
        ax2.axhline(50, color='yellow', linestyle='--', alpha=0.3)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.3)
        ax2.set_facecolor('#131722')
        ax2.set_ylim(0, 100)
        """

        """"
        # Super score
        ax2.plot(indices, super_score, color='#9b59b6', linewidth=2, label='Super Score')
        ax2.plot(indices, ema_score, color="#137169", linewidth=2, label='Ema score')
        # 1. Zonas de Gatilho (Onde o sinal é validado)
        ax2.axhline(60, color='lime', linestyle='--', alpha=0.5)   # Gatilho de Compra
        ax2.axhline(-60, color='red', linestyle='--', alpha=0.5)  # Gatilho de Venda
        # 2. Linha de Equilíbrio (Zero)
        ax2.axhline(0, color='white', linestyle='-', alpha=0.2)
        # 3. Limites Reais (O indicador vai de -100 a 100)
        # Deixamos uma margem de 110 para a linha não bater no teto
        ax2.set_ylim(-110, 110)
        # 4. Preenchimento de Cor (Opcional, mas ajuda muito a ver a força)
        ax2.fill_between(indices, 0, super_score, where=(super_score >= 0), color='green', alpha=0.2)
        ax2.fill_between(indices, 0, super_score, where=(super_score < 0), color='red', alpha=0.2)
        """


        #
        ax2.plot(indices, bull_pct, color="#0f471b", linewidth=2, label='Super Score')
        ax2.plot(indices, bear_pct, color="#A83507", linewidth=2, label='Ema score')
        ax2.plot(indices, signal_line, color="#ECECF3", linewidth=2, label='Ema score')
        
        
        # 2. Linha de Equilíbrio (Zero)
        ax2.axhline(50, color='white', linestyle='-', alpha=0.2)
        ax2.axhline(20, color='orange', linestyle='-', alpha=0.2)
        # 3. Limites Reais (O indicador vai de -100 a 100)
        # Deixamos uma margem de 110 para a linha não bater no teto
        ax2.set_ylim(0, 100)

        ax2.set_facecolor('#131722')
        

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_soheil_pko_strategy(ohlcv, symbol: str):
        opens, highs, lows, closes, volumes = ohlcv.opens, ohlcv.highs, ohlcv.lows, ohlcv.closes, ohlcv.volumes
        n = len(closes)
        indices = np.arange(n)

        # 1. Indicadores (Canais de EMA)
        utils = BaseIndicatorsUtils(ohlcv)
        ema5_h = utils.ema_list(highs, 5)
        ema5_l = utils.ema_list(lows, 5)
        ema100_h = utils.ema_list(highs, 100)
        ema100_l = utils.ema_list(lows, 100)
        ema200 = utils.ema_list(closes, 200)
        rsi, rsi_ema = utils.rsi()

        # Listas para Sinais
        buy_idx, buy_val = [], []
        sell_idx, sell_val = [], []
        
        # Variáveis de Estado (Memória do Pullback)
        waiting_buy = False
        waiting_sell = False

        # 2. Lógica de Sinais
        for i in range(200, n):
                # 1. FILTRO DE INCLINAÇÃO (SLOPE)
                # Medimos a subida/descida da EMA 200 nos últimos 5 candles
                # Se for quase 0, o mercado está lateral e ignoramos
                slope_200 = ema200[i] - ema200[i-5]
                min_slope = ema200[i] * 0.0001 # Filtro de inclinação mínima (ajustável)

                # 2. LÓGICA DE COMPRA (LONG)
                if closes[i] > ema200[i] and slope_200 > min_slope:
                    # Verificamos se o RSI está acima de 50 (Confirmação de força)
                    if rsi[i] > 30 and rsi[i] > rsi_ema[i]:
                        if lows[i] <= ema100_h[i]: # Toque no canal de Pullback
                            waiting_buy = True
                        
                        if waiting_buy and closes[i] > ema5_h[i]:
                            # Filtro extra: O candle de sinal deve ter volume acima da média
                            if volumes[i] > np.mean(volumes[i-20:i]):
                                buy_idx.append(i)
                                buy_val.append(lows[i])
                                waiting_buy = False

                # 3. LÓGICA DE VENDA (SHORT)
                elif closes[i] < ema200[i] and slope_200 < -min_slope:
                    if rsi[i] < 70 and rsi[i] < rsi_ema[i]:
                        if highs[i] >= ema100_l[i]:
                            waiting_sell = True
                        
                        if waiting_sell and closes[i] < ema5_l[i]:
                            if volumes[i] > np.mean(volumes[i-20:i]):
                                sell_idx.append(i)
                                sell_val.append(highs[i])
                                waiting_sell = False

                # 4. INVALIDAÇÃO AUTOMÁTICA
                # Se o preço cruzar a EMA 200 antes do sinal de entrada, o pullback faliu
                if waiting_buy and closes[i] < ema200[i]: waiting_buy = False
                if waiting_sell and closes[i] > ema200[i]: waiting_sell = False

        # 3. Desenho do Gráfico
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.set_facecolor('#131722')

        # Candles
        for i in range(n):
            color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
            ax1.plot([i, i], [lows[i], highs[i]], color='white', linewidth=0.8, alpha=0.3)
            ax1.bar(i, abs(closes[i] - opens[i]), bottom=min(opens[i], closes[i]), width=0.6, color=color)

        # Plot das EMAs
        # Canal Branco (5)
        ax1.plot(indices, ema5_h, color='white', alpha=0.5, linewidth=1, label='EMA 5 (H/L)')
        ax1.plot(indices, ema5_l, color='white', alpha=0.5, linewidth=1)
        ax1.fill_between(indices, ema5_l.tolist(), ema5_h.tolist(), color='white', alpha=0.05)

        # Canal Verde (100)
        ax1.plot(indices, ema100_h, color='#26a69a', alpha=0.8, linewidth=1.5, label='EMA 100 (H/L)')
        ax1.plot(indices, ema100_l, color='#26a69a', alpha=0.8, linewidth=1.5)
        ax1.fill_between(indices, ema100_l.tolist(), ema100_h.tolist(), color='#26a69a', alpha=0.1)

        # Linha de Tendência (200)
        ax1.plot(indices, ema200, color='orange', linewidth=2, label='EMA 200 (Trend)')

        # Plot dos Sinais
        ax1.scatter(buy_idx, buy_val, marker=MarkerStyle('^'), color='lime', s=200, edgecolors='black', label='BUY Signal', zorder=5)
        ax1.scatter(sell_idx, sell_val, marker=MarkerStyle('v'), color='red', s=200, edgecolors='black', label='SELL Signal', zorder=5)

        ax1.set_title(f"Soheil PKO Strategy - {symbol}", color='white', fontsize=14)
        ax1.legend(facecolor='#1b1e2e', edgecolor='white', labelcolor='white')
        ax1.grid(True, color='#2a2e39', alpha=0.3)
        
        plt.tight_layout()
        plt.show()


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

        ohlcv = await PlotTrades.get_historical_ohlcv(pair, TimeframeEnum.M15, 740)

        #PlotTrades.plot_supertrend_with_signals(ohlcv, pair.symbol)
        #PlotTrades.plot_smart_money_flow_cloud(ohlcv, pair.symbol)
        #PlotTrades.plot_custom_supertrend(ohlcv, pair.symbol)
        #PlotTrades.plot_volumatic_vidya(ohlcv, pair.symbol)
        #PlotTrades.plot_smart_money_breakout(ohlcv, pair.symbol)
        #PlotTrades.plot_sha_macd_raw_signals(ohlcv, pair.symbol)
        #PlotTrades.plot_double_rsi_signals(ohlcv, pair.symbol)
        PlotTrades.plot_double_bb_rsi_logic(ohlcv, pair.symbol)
        #PlotTrades.plot_soheil_pko_strategy(ohlcv, pair.symbol)
        
        

if __name__ == "__main__":
    asyncio.run(main())