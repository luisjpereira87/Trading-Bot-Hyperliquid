import asyncio
import logging
import os
import sys

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from ccxt.async_support import hyperliquid
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
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
                print(f"[WARNING] Trade com índice inválido: {index}, max index = {len(dates) - 1}")
                continue

            dt = dates[trade["index"]]
            price = trade["price"]

            if trade["type"] == "entry":
                ax.scatter(dt, trade["sl"], color='red', marker='v', label='SL' if trade["index"] == 0 else "",
                           alpha=0.6)
                ax.scatter(dt, trade["tp"], color='green', marker='^', label='TP' if trade["index"] == 0 else "",
                           alpha=0.6)

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
                    ax.vlines(x=x, ymin=lows[index], ymax=circle_y, linestyles='dashed', colors=color, linewidth=1.2,
                              zorder=5)
                    # Texto um pouco abaixo do círculo
                    ax.text(
                        x,
                        circle_y,  # ligeiramente abaixo do círculo
                        f"Buy ({score:.2f}), Index {index}",
                        fontsize=9,
                        ha='center',
                        va='top',  # alinhado pelo topo do texto
                        color=color,
                        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                        zorder=6
                    )

                elif signal_type == Signal.SELL:
                    color = 'red'
                    circle_y = highs[index] * 1.005
                    ax.scatter(x, circle_y, marker='o', color=color, s=100, label='Sell')
                    ax.vlines(x=x, ymin=highs[index], ymax=circle_y, linestyles='dashed', colors=color, linewidth=1.2,
                              zorder=5)
                    # Texto um pouco acima do círculo
                    ax.text(
                        x,
                        circle_y,  # ligeiramente acima do círculo
                        f"Sell ({score:.2f}), Index {index}",
                        fontsize=9,
                        ha='center',
                        va='bottom',  # alinhado pela base do texto
                        color=color,
                        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                        zorder=6
                    )

                current_signal = signal_type

    @staticmethod
    def plot_trades(symbol, ohlcv: OhlcvWrapper, signals, trades):
        # Preparar dados para plot
        # dates = [datetime.fromtimestamp(c[0] / 1000) for c in candles]
        # opens = [c[1] for c in candles]
        # highs = [c[2] for c in candles]
        # lows = [c[3] for c in candles]
        # closes = [c[4] for c in candles]

        dates = ohlcv.dates
        opens = ohlcv.opens
        highs = ohlcv.highs
        lows = ohlcv.lows
        closes = ohlcv.closes

        indicators_utils = IndicatorsUtils(ohlcv)

        # fig, ax = plt.subplots(figsize=(15, 7))
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax.set_title(f"Backtest Trades - {symbol}")

        # Plotar candles com linhas de alta/baixa
        for i in range(len(closes)):
            color = 'green' if closes[i] >= opens[i] else 'red'
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color=color)  # linha high-low
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=5)  # corpo candle

        # EMA 50
        ema800 = indicators_utils.ema(800)
        ax.plot(dates, ema800, color='black', linestyle='-', linewidth=1.5, label='EMA 800')

        ema200 = indicators_utils.ema(200)
        ax.plot(dates, ema200, color='blue', linestyle='-', linewidth=1.5, label='EMA 200')

        ema50 = indicators_utils.ema(50)
        ax.plot(dates, ema50, color='red', linestyle='-', linewidth=1.5, label='EMA 50')

        ema21 = indicators_utils.ema(20)
        ax.plot(dates, ema21, color='green', linestyle='-', linewidth=1.5, label='EMA 21')

        # Plotar entradas e saídas
        PlotTrades.build_trade(ax, dates, trades)

        # Plotar pontuação (score) sobre os candles
        PlotTrades.build_signals(ax, dates, signals, highs, lows)

        rsi, rsi_ema = indicators_utils.rsi()
        ax2.plot(dates, rsi, color='#9b59b6', linewidth=2)
        # ax2.plot(indices, rsi_higher, color='#368921', linewidth=2)
        ax2.plot(dates, rsi_ema, color="#368921", linewidth=2)
        # ax2.plot(indices, rsi_ema, color="#368921", linewidth=2)
        ax2.axhline(70, color='red', linestyle='--', alpha=0.3)
        ax2.axhline(50, color='yellow', linestyle='--', alpha=0.3)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.3)
        ax2.set_facecolor('#131722')
        ax2.set_ylim(0, 100)

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
        supertrend, trend, upperband, lowerband, supertrend_smooth, direction, _ = indicatorsUtils.supertrend()
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
        stop_line, trend_stop_atr = indicatorsUtils.stop_atr_tradingview()
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
            # ax.hlines(sl, dates_num-0.2, dates_num+0.2, colors='red', linestyles='dashed', linewidth=1)
            # ax.hlines(tp, dates_num-0.2, dates_num+0.2, colors='green', linestyles='dashed', linewidth=1)

            if ema_cross_signal[i] == Signal.BUY:
                ax.scatter(dates_num, (0.985 * lows[i]), color='green', s=100, zorder=5)
                ax.vlines(dates_num, ymin=lows[i], ymax=(0.985 * lows[i]), linestyles='dashed', color='green',
                          linewidth=1.2, zorder=5)

                ax.text(
                    dates_num,
                    (0.985 * lows[i]),  # ligeiramente abaixo do círculo
                    f"Index {i}",
                    fontsize=9,
                    ha='center',
                    va='top',  # alinhado pelo topo do texto
                    color='green',
                    bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                    zorder=6
                )
            elif ema_cross_signal[i] == Signal.SELL:
                ax.scatter(dates_num, (1.015 * highs[i]), color='red', s=100, zorder=5)
                ax.vlines(dates_num, ymin=highs[i], ymax=(1.015 * highs[i]), linestyles='dashed', color='red',
                          linewidth=1.2, zorder=5)

                ax.text(
                    dates_num,
                    (1.015 * highs[i]),  # ligeiramente abaixo do círculo
                    f"Index {i}",
                    fontsize=9,
                    ha='center',
                    va='top',  # alinhado pelo topo do texto
                    color='red',
                    bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                    zorder=6
                )
            elif ema_cross_signal[i] == Signal.CLOSE:
                ax.scatter(dates_num, (1.015 * highs[i]), color='brown', s=100, zorder=5)
                ax.vlines(dates_num, ymin=highs[i], ymax=(1.015 * highs[i]), linestyles='dashed', color='brown',
                          linewidth=1.2, zorder=5)

                ax.text(
                    dates_num,
                    (1.015 * highs[i]),  # ligeiramente abaixo do círculo
                    f"Index {i}",
                    fontsize=9,
                    ha='center',
                    va='top',  # alinhado pelo topo do texto
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
        # ax.plot(dates, upper, color="purple", linestyle="--", linewidth=1, label="Bollinger Upper")
        # ax.plot(dates, mid, color="gray", linestyle="--", linewidth=1, label="Bollinger Mid")
        # ax.plot(dates, lower, color="purple", linestyle="--", linewidth=1, label="Bollinger Lower")

        # Plot stop atr
        """
        ax.plot(dates, stop_line, color='orange', linestyle='--', linewidth=1.5, label='Stop ATR')
        """
        up_trend_mask = trend_stop_atr == 1
        down_trend_mask = trend_stop_atr == -1
        # ax.plot(dates, stop_line, color='green', linestyle='-', linewidth=1.5, label='Stop ATR (Uptrend)')
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
        # ts = res["ts"]
        # direction = res["direction"]  # 1 bullish, 0 bearish
        # perf_score = res["perf_score"]
        n = len(closes)

        # print("AQUIII", direction)

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
            if direction[i - 1] == -1 and direction[i] == 1 and closes[i] > ts[i - 1]:

                ax.scatter(dates_num, (0.985 * lows[i]), color='green', s=100, zorder=5)
                ax.vlines(dates_num, ymin=lows[i], ymax=(0.985 * lows[i]), linestyles='dashed', color='green',
                          linewidth=1.2, zorder=5)

                ax.text(
                    dates_num,
                    (0.985 * lows[i]),  # ligeiramente abaixo do círculo
                    f"Index {i}, {perf_score[i]}",
                    fontsize=9,
                    ha='center',
                    va='top',  # alinhado pelo topo do texto
                    color='green',
                    bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                    zorder=6
                )

                # buy_signals.append((dates[i], closes[i]))
                trend_change_scores.append((dates[i], perf_score[i]))
            # SELL: direção muda de 1 -> 0 e fechamento abaixo do TS anterior
            elif direction[i - 1] == 1 and direction[i] == -1 and closes[i] < ts[i - 1]:
                # sell_signals.append((dates[i], closes[i]))
                trend_change_scores.append((dates[i], perf_score[i]))

                ax.scatter(dates_num, (1.015 * highs[i]), color='red', s=100, zorder=5)
                ax.vlines(dates_num, ymin=highs[i], ymax=(1.015 * highs[i]), linestyles='dashed', color='red',
                          linewidth=1.2, zorder=5)

                ax.text(
                    dates_num,
                    (1.015 * highs[i]),  # ligeiramente abaixo do círculo
                    f"Index {i}, {perf_score[i]}",
                    fontsize=9,
                    ha='center',
                    va='top',  # alinhado pelo topo do texto
                    color='red',
                    bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                    zorder=6
                )

            # Trailing stop para plot contínuo
            ts_long[i] = ts[i] if direction[i] == 1 else ts_long[i - 1]
            ts_short[i] = ts[i] if direction[i] == -1 else ts_short[i - 1]

        # Candles
        for i in range(n):
            color = 'green' if direction[i] == 1 else 'red'
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)  # high-low
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=5)  # body

        # for dt, score in trend_change_scores:
        #    ax.text(dt, score, str(score), color='blue', fontsize=10, ha='center', va='bottom')

        # SuperTrend contínuo colorido

        # ax.plot(dates, ts_long, color='blue', linewidth=1, label='TS Long')
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
        # ts = res["ts"]
        # direction = res["direction"]  # 1 bullish, 0 bearish
        # perf_score = res["perf_score"]
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
            if direction[i - 1] == -1 and direction[i] == 1:

                ax.scatter(dates_num, (0.985 * lows[i]), color='green', s=100, zorder=5)
                ax.vlines(dates_num, ymin=lows[i], ymax=(0.985 * lows[i]), linestyles='dashed', color='green',
                          linewidth=1.2, zorder=5)

                ax.text(
                    dates_num,
                    (0.985 * lows[i]),  # ligeiramente abaixo do círculo
                    f"Index {i}, {perf_score[i]}",
                    fontsize=9,
                    ha='center',
                    va='top',  # alinhado pelo topo do texto
                    color='green',
                    bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                    zorder=6
                )

                # buy_signals.append((dates[i], closes[i]))
                trend_change_scores.append((dates[i], perf_score[i]))
            # SELL: direção muda de 1 -> 0 e fechamento abaixo do TS anterior
            elif direction[i - 1] == 1 and direction[i] == -1:
                # sell_signals.append((dates[i], closes[i]))
                trend_change_scores.append((dates[i], perf_score[i]))

                ax.scatter(dates_num, (1.015 * highs[i]), color='red', s=100, zorder=5)
                ax.vlines(dates_num, ymin=highs[i], ymax=(1.015 * highs[i]), linestyles='dashed', color='red',
                          linewidth=1.2, zorder=5)

                ax.text(
                    dates_num,
                    (1.015 * highs[i]),  # ligeiramente abaixo do círculo
                    f"Index {i}, {perf_score[i]}",
                    fontsize=9,
                    ha='center',
                    va='top',  # alinhado pelo topo do texto
                    color='red',
                    bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                    zorder=6
                )

            # Trailing stop para plot contínuo
            # ts_long[i] = ts[i] if direction[i] == 1 else ts_long[i-1]
            # ts_short[i] = ts[i] if direction[i] == 0 else ts_short[i-1]

        # Candles
        for i in range(n):
            color = 'green' if direction[i] == 1 else 'red'
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)  # high-low
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=5)  # body

        # for dt, score in trend_change_scores:
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
            x = [mdates.date2num(dates[i - 1]), mdates.date2num(dates[i])]
            y = [band[i - 1], band[i]]
            segments.append(list(zip(x, y)))
            colors.append(color_up if trend[i - 1] == 1 else color_down)

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
                ax1.axvspan(dates_array[i - 1], dates_array[i], color="green", alpha=0.03)
            else:
                ax1.axvspan(dates_array[i - 1], dates_array[i], color="red", alpha=0.03)

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
        # supertrend = indicatorsUtils.supertrend_ai()
        res = indicatorsUtils.smart_money_flow_cloud()
        signal = res['signal']
        strength = res['strength']
        upper = res['upper']
        lower = res['lower']
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

            if ((signal[i - 1] == -1 and signal[i] == 1) or bull_retest[i]):

                ax.scatter(dates_num, (0.985 * lows[i]), color='green', s=100, zorder=5)
                ax.vlines(dates_num, ymin=lows[i], ymax=(0.985 * lows[i]), linestyles='dashed', color='green',
                          linewidth=1.2, zorder=5)

                ax.text(
                    dates_num,
                    (0.985 * lows[i]),  # ligeiramente abaixo do círculo
                    f"Index {i}, {round(strength[i], 2)}",
                    fontsize=9,
                    ha='center',
                    va='top',  # alinhado pelo topo do texto
                    color='green',
                    bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                    zorder=6
                )
            # SELL: direção muda de 1 -> 0 e fechamento abaixo do TS anterior
            elif ((signal[i - 1] == 1 and signal[i] == -1) or bear_retest[i]):

                ax.scatter(dates_num, (1.015 * highs[i]), color='red', s=100, zorder=5)
                ax.vlines(dates_num, ymin=highs[i], ymax=(1.015 * highs[i]), linestyles='dashed', color='red',
                          linewidth=1.2, zorder=5)

                ax.text(
                    dates_num,
                    (1.015 * highs[i]),  # ligeiramente abaixo do círculo
                    f"Index {i}, {round(strength[i], 2)}",
                    fontsize=9,
                    ha='center',
                    va='top',  # alinhado pelo topo do texto
                    color='red',
                    bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                    zorder=6
                )

        # Candles
        for i in range(n):
            color = 'green' if signal[i] == 1 else 'red'
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)  # high-low
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=5)  # body

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
        # supertrend = indicatorsUtils.supertrend_ai()
        direction = indicatorsUtils.market_structure_rsi()

        # ts = res["ts"]
        # direction = res["direction"]  # 1 bullish, 0 bearish
        # perf_score = res["perf_score"]
        n = len(closes)

        # print("AQUIII", direction)

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
            if direction[i] == 1:

                ax.scatter(dates_num, (0.985 * lows[i]), color='green', s=100, zorder=5)
                ax.vlines(dates_num, ymin=lows[i], ymax=(0.985 * lows[i]), linestyles='dashed', color='green',
                          linewidth=1.2, zorder=5)

                ax.text(
                    dates_num,
                    (0.985 * lows[i]),  # ligeiramente abaixo do círculo
                    f"Index {i}",
                    fontsize=9,
                    ha='center',
                    va='top',  # alinhado pelo topo do texto
                    color='green',
                    bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                    zorder=6
                )
            # SELL: direção muda de 1 -> 0 e fechamento abaixo do TS anterior
            elif direction[i] == -1:

                ax.scatter(dates_num, (1.015 * highs[i]), color='red', s=100, zorder=5)
                ax.vlines(dates_num, ymin=highs[i], ymax=(1.015 * highs[i]), linestyles='dashed', color='red',
                          linewidth=1.2, zorder=5)

                ax.text(
                    dates_num,
                    (1.015 * highs[i]),  # ligeiramente abaixo do círculo
                    f"Index {i}",
                    fontsize=9,
                    ha='center',
                    va='top',  # alinhado pelo topo do texto
                    color='red',
                    bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                    zorder=6
                )

        # Candles
        for i in range(n):
            color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)  # high-low
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=5)  # body
            # ax.plot([i, i], [lows[i], highs[i]], color='white', linewidth=1, alpha=0.4)
            # ax.bar(i, abs(closes[i] - opens[i]), bottom=min(opens[i], closes[i]), width=width, color=color, zorder=3)

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
        duration = res["duration"]  # Importante: deve vir do retorno da função

        fig, ax = plt.subplots(figsize=(18, 7))
        ax.set_title(f"Smart Money Breakout Channels - {symbol}")
        ax.set_facecolor('#ffffff')

        # --- 1. Desenho de Candles ---
        for i in range(n):
            color = '#00ffbb' if closes[i] >= opens[i] else '#ff1100'
            # Pavio (High-Low)
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=0.8, zorder=3)
            # Corpo (Open-Close)
            ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=4, solid_capstyle='butt',
                    zorder=4)

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
        indicators_utils = IndicatorsUtils(ohlcv)
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

        ema200 = indicators_utils.ema(200)
        ax1.plot(dates, ema200, color='blue', linestyle='-', linewidth=1.5, label='EMA 200')

        ema50 = indicators_utils.ema(50)
        ax1.plot(dates, ema50, color='red', linestyle='-', linewidth=1.5, label='EMA 50')

        ema21 = indicators_utils.ema(21)
        ax1.plot(dates, ema21, color='green', linestyle='-', linewidth=1.5, label='EMA 21')

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
            if ha_c[i] > signal[i] and ha_c[i - 1] <= signal[i - 1]:
                # Marcador verde em ambos os quadros
                ax1.scatter(dates_array[i], (lows[i] * 0.99), color='green', marker='^', s=100, zorder=10)
                ax2.scatter(dates_array[i], signal[i], color='green', marker='o', s=40, zorder=10)

            # Cruzamento de BAIXA em bruto
            elif ha_c[i] < signal[i] and ha_c[i - 1] >= signal[i - 1]:
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
        # ax2.plot(dates_num, rsi, color='red', linewidth=1.2, label='RSI 14', alpha=0.9)

        # --- SINAIS DE ENTRADA (Sincronizados com o Gráfico) ---
        for i in range(1, n):
            # Sinal de COMPRA (Verde)
            if signals[i] == Signal.BUY:
                ax1.scatter(dates_array[i], (lows[i] * 0.98), color='green', marker='^', s=150, edgecolors='black',
                            zorder=10)
                ax2.scatter(dates_array[i], rsi_fast[i], color='green', marker='o', s=60, edgecolors='black', zorder=10)

            # Sinal de VENDA (Vermelho)
            elif signals[i] == Signal.SELL:
                ax1.scatter(dates_array[i], (highs[i] * 1.02), color='red', marker='v', s=150, edgecolors='black',
                            zorder=10)
                ax2.scatter(dates_array[i], rsi_fast[i], color='red', marker='o', s=60, edgecolors='black', zorder=10)

        # Estética Final
        ax1.set_title(f"{symbol} - Preço + Sinais Double RSI (Scalping Setup)")
        ax1.grid(True, alpha=0.1)
        ax2.grid(True, alpha=0.1)
        ax2.set_ylim(0, 100)  # RSI é sempre 0-100
        ax2.legend(loc='upper right')

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_rsi_trendline_full_visual(ohlcv, symbol: str):
        opens, highs, lows, closes = ohlcv.opens, ohlcv.highs, ohlcv.lows, ohlcv.closes
        n = len(closes)
        indices = np.arange(n)  # Usamos índices simples 0, 1, 2...

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
            rsi_slice = np.array(rsi6[i - 10:i])
            rsi_overbought = any(val > 70 for val in rsi_slice)
            rsi_oversold = any(val < 30 for val in rsi_slice)

            # --- LÓGICA DE VENDA (SHORT) ---
            if highs[i] >= bb_upper[i] and rsi_overbought:
                # O "degrau": procuramos o valor mais baixo do RSI entre o pico e agora
                # Se o RSI atual quebrar esse valor, a "linha de tendência" foi rompida
                last_low_rsi = min(rsi6[i - 4:i])

                if rsi6[i] < last_low_rsi and rsi6[i - 1] >= last_low_rsi:
                    signals_sell_idx.append(i)
                    signals_sell_val.append(highs[i] * 1.002)
                    # Desenha uma linha horizontal curta para mostrar o "rompimento"
                    trendlines.append(((i - 4, i), (last_low_rsi, last_low_rsi), 'red'))

            # --- LÓGICA DE COMPRA (LONG) ---
            elif lows[i] <= bb_lower[i] and rsi_oversold:
                # O "degrau": procuramos o valor mais alto do RSI entre o fundo e agora
                last_high_rsi = max(rsi6[i - 4:i])

                if rsi6[i] > last_high_rsi and rsi6[i - 1] <= last_high_rsi:
                    signals_buy_idx.append(i)
                    signals_buy_val.append(lows[i] * 0.998)
                    # Desenha uma linha horizontal curta para mostrar o "rompimento"
                    trendlines.append(((i - 4, i), (last_high_rsi, last_high_rsi), 'green'))

        # --- PLOTAGEM ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # 2. Desenho Manual dos Candles
        width = 0.6  # Largura fixa para índices inteiros
        for i in range(n):
            color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'  # Cores TradingView
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
        rsi, rsi_ema = indicatorsUtils.rsi()
        # rsi1, rsi_ema1 = indicatorsUtils1.rsi()
        # last_HL, last_LH = tvIndicatorsUtils.market_structure(3, 2)
        # --- CÁLCULO DOS CANAIS ---
        # Podes usar 3.0 para exaustão extrema ou 2.0 para zona de alerta
        # k_up, k_mid, k_low = indicatorsUtils.keltner_channels( period=20, multiplier=3.0)
        # k_up80, k_mid80, k_low80 = indicatorsUtils.keltner_channels(period=80, multiplier=3.0)

        double_bb = indicatorsUtils.double_bb_rsi_logic()
        bb20_up = double_bb['bbshort_up']
        bb20_low = double_bb['bbshort_low']
        bb20_mid = double_bb['bbshort_mid']
        bb80_up = double_bb['bblong_up']
        bb80_low = double_bb['bblong_low']
        bb80_mid = double_bb['bblong_mid']
        signals = double_bb['signals']
        entry_buy_idx = double_bb['entry_buy_idx']
        entry_buy_val = double_bb['entry_buy_val']
        entry_sell_idx = double_bb['entry_sell_idx']
        entry_sell_val = double_bb['entry_sell_val']

        super_score, ema_score = indicatorsUtils.calculate_super_score()
        ema200 = indicatorsUtils.ema(200)

        # bull_pct, bear_pct, slope_ma_14, polarity_osc, signal_line = indicatorsUtils.calculate_lux_bb_oscillator()
        # base_mult=2, min_mult=1.5, base_length=7
        supertrend, trend, upperband, lowerband, supertrend_smooth, direction, perf_score = indicatorsUtils.supertrend()
        #tv_indicators = TvIndicatorsUtils(ohlcv)

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
        ax1.plot(indices, bb80_mid, color='orange', linewidth=1.5, alpha=0.6)
        # Linha de Tendência (200)
        ax1.plot(indices, ema200, color='red', linewidth=1.5, alpha=0.6, label='EMA 200 (Trend)')

        ax1.plot(indices, bb80_mid, color='orange', linewidth=1.5, alpha=0.6)
        ax1.fill_between(indices, bb80_low, bb80_up, color='orange', alpha=0.03)

        # TOPO: BB20 Upper ACIMA da BB80 Upper
        topo_extremo = bb20_up > bb80_up
        ax1.fill_between(range(len(bb20_up)), bb80_up, bb20_up,
                         where=topo_extremo, color='red', alpha=0.3, label='Exaustão Topo')

        # FUNDO: BB20 Low ABAIXO da BB80 Low (Aqui é onde costuma falhar)
        fundo_extremo = bb20_low < bb80_low
        ax1.fill_between(range(len(bb20_low)), bb80_low, bb20_low,
                         where=fundo_extremo, color='green', alpha=0.3, label='Exaustão Fundo')

        for i in range(1, n):
            # Verifica se estamos perante um sinal de Compra válido e se o índice existe nos arrays de valores
            if signals[i] == 1:
                val = entry_buy_val[i] if i < len(entry_buy_val) and entry_buy_val[i] is not None else closes[i]
                idx_label = entry_buy_idx[i] if i < len(entry_buy_idx) else i

                ax1.scatter(i, val, marker='^', color='#00ff00',
                            s=150, edgecolors='black', linewidth=1.5, zorder=5)
                ax1.text(
                    i, val, f"Idx {idx_label}",
                    fontsize=8, ha='center', va='bottom',
                    color='darkgreen',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'),
                    zorder=6
                )

            # Verifica se estamos perante um sinal de Venda válido
            elif signals[i] == -1:
                val = entry_sell_val[i] if i < len(entry_sell_val) and entry_sell_val[i] is not None else closes[i]
                idx_label = entry_sell_idx[i] if i < len(entry_sell_idx) else i

                ax1.scatter(i, val, marker='v', color='#ff0000',
                            s=150, edgecolors='black', linewidth=1.5, zorder=5)
                ax1.text(
                    i, val, f"Idx {idx_label}",
                    fontsize=8, ha='center', va='top',
                    color='darkred',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'),
                    zorder=6
                )




        ax1.set_facecolor('#131722')
        ax1.grid(True, color='#2a2e39', alpha=0.5)
        ax1.legend()

        # RSI 6

        ax2.plot(indices, rsi, color='#9b59b6', linewidth=2)
        # ax2.plot(indices, rsi_higher, color='#368921', linewidth=2)
        ax2.plot(indices, rsi_ema, color="#368921", linewidth=2)
        # ax2.plot(indices, rsi_ema, color="#368921", linewidth=2)
        ax2.axhline(70, color='red', linestyle='--', alpha=0.3)
        ax2.axhline(50, color='yellow', linestyle='--', alpha=0.3)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.3)
        ax2.set_facecolor('#131722')
        ax2.set_ylim(0, 100)

        """
        # Andean
        ax2.plot(indices, andean_bear, color='red', linewidth=2, label='Andean Bear')
        ax2.plot(indices, andean_bull, color="green", linewidth=2, label='Andean Bull')
        ax2.plot(indices, andean_signal, color="yellow", linewidth=2, label='Andean Signal')
        # 1. Zonas de Gatilho (Onde o sinal é validado)
        # ax2.axhline(60, color='lime', linestyle='--', alpha=0.5)  # Gatilho de Compra
        # ax2.axhline(-60, color='red', linestyle='--', alpha=0.5)  # Gatilho de Venda
        # 2. Linha de Equilíbrio (Zero)
        ax2.axhline(0, color='white', linestyle='-', alpha=0.2)
        # 3. Limites Reais (O indicador vai de -100 a 100)
        # Deixamos uma margem de 110 para a linha não bater no teto
        ax2.set_ylim(-1, 2)

        print(andean_signal)

        """
        """
        # Super score
        ax2.plot(indices, super_score, color='#9b59b6', linewidth=2, label='Super Score')
        ax2.plot(indices, ema_score, color="#137169", linewidth=2, label='Ema score')
        # 1. Zonas de Gatilho (Onde o sinal é validado)
        ax2.axhline(60, color='lime', linestyle='--', alpha=0.5)  # Gatilho de Compra
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
         """
        ax2.set_facecolor('#131722')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_soheil_pko_strategy__(ohlcv, symbol: str):
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
            slope_200 = ema200[i] - ema200[i - 5]
            min_slope = ema200[i] * 0.0001  # Filtro de inclinação mínima (ajustável)

            # 2. LÓGICA DE COMPRA (LONG)
            if closes[i] > ema200[i] and slope_200 > min_slope:
                # Verificamos se o RSI está acima de 50 (Confirmação de força)
                # if rsi[i] > 30 and rsi[i] > rsi_ema[i]:
                if lows[i] <= ema100_h[i]:  # Toque no canal de Pullback
                    waiting_buy = True

                if waiting_buy and closes[i] > ema5_h[i]:
                    # Filtro extra: O candle de sinal deve ter volume acima da média
                    # if volumes[i] > np.mean(volumes[i - 20:i]):
                    buy_idx.append(i)
                    buy_val.append(lows[i])
                    waiting_buy = False

            # 3. LÓGICA DE VENDA (SHORT)
            elif closes[i] < ema200[i] and slope_200 < -min_slope:
                # if rsi[i] < 70 and rsi[i] < rsi_ema[i]:
                if highs[i] >= ema100_l[i]:
                    waiting_sell = True

                if waiting_sell and closes[i] < ema5_l[i]:
                    # if volumes[i] > np.mean(volumes[i - 20:i]):
                    sell_idx.append(i)
                    sell_val.append(highs[i])
                    waiting_sell = False

            # 4. INVALIDAÇÃO AUTOMÁTICA
            # Se o preço cruzar a EMA 200 antes do sinal de entrada, o pullback faliu
            if waiting_buy and closes[i] < ema200[i]: waiting_buy = False
            if waiting_sell and closes[i] > ema200[i]: waiting_sell = False

        # 3. Desenho do Gráfico
        # fig, ax1 = plt.subplots(figsize=(16, 8))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
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
        ax1.scatter(buy_idx, buy_val, marker=MarkerStyle('^'), color='lime', s=200, edgecolors='black',
                    label='BUY Signal', zorder=5)
        ax1.scatter(sell_idx, sell_val, marker=MarkerStyle('v'), color='red', s=200, edgecolors='black',
                    label='SELL Signal', zorder=5)

        ax1.set_title(f"Soheil PKO Strategy - {symbol}", color='white', fontsize=14)
        ax1.legend(facecolor='#1b1e2e', edgecolor='white', labelcolor='white')
        ax1.grid(True, color='#2a2e39', alpha=0.3)

        adx = utils.adx()

        # ax2.plot(indices, rsi, color='#9b59b6', linewidth=2, label='RSI')
        ax2.plot(indices, adx, color='#9b59b6', linewidth=2, label='ADX')
        # ax2.plot(indices, rsi_ema, color="#368921", linewidth=1.5, label='RSI EMA')
        ax2.axhline(70, color='red', linestyle='--', alpha=0.3)
        ax2.axhline(50, color='yellow', linestyle='--', alpha=0.3)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.3)
        ax2.set_facecolor('#131722')
        ax2.set_ylim(0, 100)
        ax2.tick_params(colors='white')
        ax2.grid(True, color='#2a2e39', alpha=0.2)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_soheil_pko_strategy(ohlcv, symbol: str):
        opens, highs, lows, closes, volumes = ohlcv.opens, ohlcv.highs, ohlcv.lows, ohlcv.closes, ohlcv.volumes
        n = len(closes)
        indices = np.arange(n)

        # 1. Indicadores Base (Canais de EMA e Momentum)
        utils = BaseIndicatorsUtils(ohlcv)
        ema5_h = utils.ema_list(highs, 5)
        ema5_l = utils.ema_list(lows, 5)
        ema100_h = utils.ema_list(highs, 100)
        ema100_l = utils.ema_list(lows, 100)
        ema200 = utils.ema_list(closes, 200)
        ema50 = utils.ema_list(closes, 50)
        adx = utils.adx()  # Extraído aqui para ser usado tanto na lógica como no plot

        # Listas para Armazenamento de Sinais
        buy_idx, buy_val = [], []
        sell_idx, sell_val = [], []

        # Variáveis de Estado (Memória e Timers do Pullback)
        waiting_buy = False
        waiting_sell = False
        buy_timer = 0
        sell_timer = 0

        # Restrição de proximidade para evitar sinais colados no mesmo movimento
        last_buy_bar = -20
        last_sell_bar = -20

        # 2. Lógica Algorítmica de Sinais
        for i in range(200, n):
            # Filtro de Inclinação da Tendência (Slope)
            slope_200 = ema200[i] - ema200[i - 5]
            min_slope = ema200[i] * 0.0001  # Filtro adaptativo baseado no preço do ativo

            # Filtro de Força Direcional (ADX > 22 indica tendência clara)
            adx_forte = adx[i] > 22

            # Incrementar os cronómetros se o alerta de pullback estiver ativo
            if waiting_buy: buy_timer += 1
            if waiting_sell: sell_timer += 1

            # 🟢 LÓGICA DE COMPRA (LONG)
            if ema50[i] > ema200[i] and slope_200 > min_slope:
                # Condição 1: A mínima do candle vem testar o canal de Pullback (EMA 100 alta)
                if lows[i] <= ema100_h[i]:
                    waiting_buy = True
                    buy_timer = 0  # Faz reset ao cronómetro para exigir reação rápida

                # Condição 2: Reação imediata com fecho acima da EMA 5 alta dentro do limite de tempo
                if waiting_buy and closes[i] > ema5_h[i] and buy_timer <= 7:
                    # Condição 3: Volume acima da média e respeito pelo cooldown de barras
                    # if volumes[i] > np.mean(volumes[i - 20:i]) and (i - last_buy_bar > 10):
                    buy_idx.append(i)
                    buy_val.append(lows[i])
                    last_buy_bar = i
                    waiting_buy = False

            # 🔴 LÓGICA DE VENDA (SHORT)
            elif ema50[i] < ema200[i] and slope_200 < -min_slope:
                # Condição 1: A máxima do candle vem testar o canal de Pullback (EMA 100 baixa)
                if highs[i] >= ema100_l[i]:
                    waiting_sell = True
                    sell_timer = 0

                # Condição 2: Reação rápida com fecho abaixo da EMA 5 baixa dentro do limite de tempo
                if waiting_sell and closes[i] < ema5_l[i] and sell_timer <= 7:
                    # Condição 3: Volume acima da média e respeito pelo cooldown de barras
                    # if volumes[i] > np.mean(volumes[i - 20:i]) and (i - last_sell_bar > 10):
                    sell_idx.append(i)
                    sell_val.append(highs[i])
                    last_sell_bar = i
                    waiting_sell = False

            # ❌ INVALIDAÇÃO AUTOMÁTICA & TIMEOUTS
            # O alerta é desligado se o preço romper a EMA 200 contra a tendência ou se demorar mais de 7 velas a reagir
            if waiting_buy and (closes[i] < ema200[i] or buy_timer > 7):
                waiting_buy = False
            if waiting_sell and (closes[i] > ema200[i] or sell_timer > 7):
                waiting_sell = False

        # 3. Desenho Estruturado do Gráfico (Duplo Painel)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax1.set_facecolor('#131722')
        fig.patch.set_facecolor('#131722')

        # Renderização dos Candles Japoneses
        for i in range(n):
            candle_color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
            ax1.plot([i, i], [lows[i], highs[i]], color='white', linewidth=0.8, alpha=0.3)
            ax1.bar(i, abs(closes[i] - opens[i]), bottom=min(opens[i], closes[i]), width=0.6, color=candle_color)

        # Plot dos Canais de EMA
        # Canal Micro Reação (EMA 5 High/Low em Branco)
        ax1.plot(indices, ema5_h, color='white', alpha=0.4, linewidth=1, label='EMA 5 (H/L)')
        ax1.plot(indices, ema5_l, color='white', alpha=0.4, linewidth=1)
        ax1.fill_between(indices, ema5_l, ema5_h, color='white', alpha=0.03)

        # Canal Pullback Estrutural (EMA 100 High/Low em Verde Água)
        ax1.plot(indices, ema100_h, color='#26a69a', alpha=0.7, linewidth=1.5, label='EMA 100 (H/L)')
        ax1.plot(indices, ema100_l, color='#26a69a', alpha=0.7, linewidth=1.5)
        ax1.fill_between(indices, ema100_l, ema100_h, color='#26a69a', alpha=0.08)

        # Média Móvel Macro (EMA 200 em Laranja)
        ax1.plot(indices, ema200, color='orange', linewidth=2, label='EMA 200 (Trend Direction)')
        ax1.plot(indices, ema50, color='red', alpha=0.7,linewidth=2, label='EMA 50 (Trend Direction)')

        # Impressão Visual dos Sinais Executados
        ax1.scatter(buy_idx, buy_val, marker=MarkerStyle('^'), color='lime', s=180, edgecolors='black',
                    label='PKO BUY Trigger', zorder=5)
        ax1.scatter(sell_idx, sell_val, marker=MarkerStyle('v'), color='red', s=180, edgecolors='black',
                    label='PKO SELL Trigger', zorder=5)

        ax1.set_title(f"Soheil PKO Strategy Pro - {symbol}", color='white', fontsize=14, pad=15)
        ax1.legend(facecolor='#1b1e2e', edgecolor='white', labelcolor='white', loc='upper left')
        ax1.grid(True, color='#2a2e39', alpha=0.3)
        ax1.tick_params(colors='white')

        # Painel Inferior: Força de Tendência (ADX)
        ax2.set_facecolor('#131722')
        ax2.plot(indices, adx, color='#9b59b6', linewidth=2, label='ADX Matrix')

        # Níveis horizontais clássicos de validação do ADX
        ax2.axhline(22, color='yellow', linestyle='--', alpha=0.4, label='Trend Threshold (22)')
        ax2.axhline(40, color='red', linestyle=':', alpha=0.3)

        ax2.set_ylim(0, 70)
        ax2.tick_params(colors='white')
        ax2.grid(True, color='#2a2e39', alpha=0.2)
        ax2.legend(facecolor='#1b1e2e', edgecolor='white', labelcolor='white', loc='upper left')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_test_(ohlcv, ohlcvHigher, symbol: str):
        opens, highs, lows, closes, volumes = ohlcv.opens, ohlcv.highs, ohlcv.lows, ohlcv.closes, ohlcv.volumes
        n = len(closes)
        indices = np.arange(n)

        # 1. Indicadores Base (Canais de EMA e Momentum)
        utils = BaseIndicatorsUtils(ohlcv)
        utils_higher = BaseIndicatorsUtils(ohlcvHigher)
        ema21 = utils.ema(21)
        ema50 = utils.ema(50)
        ema200 = utils.ema_list(closes, 200)
        adx = utils.adx()  # Extraído aqui para ser usado tanto na lógica como no plot
        atr = utils.atr()

        rsi, rsi_ema = utils.rsi(period=14, signal_period=14)
        rsi_h, rsi_ema_h = utils_higher.rsi(period=14, signal_period=14)

        ema21 = utils.kalman_filter(ema21)
        ema50 = utils.kalman_filter(ema50)
        ema200 = utils.kalman_filter(ema200)

        # print(ema21, ema50)

        # Listas para Armazenamento de Sinais
        buy_idx, buy_val = [], []
        sell_idx, sell_val = [], []

        # Variáveis de Estado (Memória e Timers do Pullback)
        waiting_buy = False
        waiting_sell = False
        buy_timer = 0
        sell_timer = 0

        # Restrição de proximidade para evitar sinais colados no mesmo movimento
        last_buy_bar = -20
        last_sell_bar = -20

        lower_band = np.full(n, np.nan)
        upper_band = np.full(n, np.nan)

        lookback = 14
        is_enter = True
        lower_band_raw = None
        upper_band_raw = None

        top_idx, top_val = [], []
        bottom_idx, bottom_val = [], []

        # State memory flags
        waiting_buy_confirmation = False
        waiting_sell_confirmation = False
        EMA_PROXIMITY_THRESHOLD = 0.001
        for i in range(lookback, n):

            current_price = closes[i]
            is_green_candle = closes[i] > opens[i]
            is_red_candle = closes[i] < opens[i]

            # --- CALCULAR PROXIMIDADE DAS EMAs ---
            # dist_ema21 = abs(current_price - ema21[i]) / current_price
            # dist_ema50 = abs(current_price - ema50[i]) / current_price
            # is_near_ema = (dist_ema21 <= EMA_PROXIMITY_THRESHOLD) or (dist_ema50 <= EMA_PROXIMITY_THRESHOLD)

            # --- 1. VALIDATE CANDLE N+1 (CONTINUATION OR REVERSAL) ---

            candle_range = highs[i] - lows[i]

            is_top_wick_rejection = False
            if candle_range > 0:
                # Tamanho da sombra superior (Upper Wick)
                upper_wick = highs[i] - max(opens[i], closes[i])
                wick_ratio = upper_wick / candle_range

                # Se a sombra superior for mais de 45% do tamanho total do candle -> Rejeição / Topo
                is_top_wick_rejection = wick_ratio > 0.45

            """
            if waiting_buy_confirmation:
                if is_red_candle:  # ✅ Continuation: Trend entry confirmed
                    if rsi[i - 1] > 70 and rsi[i] < rsi[i - 1]:
                        print("SELL", i, rsi[i - 1], rsi[i], is_top_wick_rejection)
                        # top_idx.append(i)
                        # top_val.append(highs[i])
                waiting_buy_confirmation = False  # Reset state

            elif waiting_sell_confirmation:
                if is_green_candle:  # ✅ Continuation: Trend entry confirmed
                    if rsi[i - 1] < 30 and rsi[i] > rsi[i - 1]:
                        print("BUY", i, rsi[i - 1], rsi[i], is_top_wick_rejection)
                        # bottom_idx.append(i)
                        # bottom_val.append(lows[i])
                waiting_sell_confirmation = False  # Reset state
            """
            # --- 2. DETECT INITIAL BREAKOUT ---
            if lower_band_raw is not None and upper_band_raw is not None:

                if current_price > upper_band_raw:

                    ema21_inside = (lows[i] <= ema21[i] <= highs[i]) or (
                            abs(lows[i] - ema21[i]) / lows[i] <= EMA_PROXIMITY_THRESHOLD)
                    ema50_inside = (lows[i] <= ema50[i] <= highs[i]) or (
                            abs(lows[i] - ema50[i]) / lows[i] <= EMA_PROXIMITY_THRESHOLD)
                    is_near_ema = ema21_inside or ema50_inside
                    is_trend_up = closes[i] > ema200[i] and ema50[i] < ema21[i]
                    if is_trend_up:
                        buy_idx.append(i)
                        buy_val.append(lows[i])
                        print(f"BUY index: {i}, rsi: {rsi[i]}")

                    waiting_buy_confirmation = True  # Trigger check for the next candle
                    is_enter = True
                    lower_band_raw = None
                    upper_band_raw = None

                elif current_price < lower_band_raw:

                    ema21_inside = (lows[i] <= ema21[i] <= highs[i]) or (
                            abs(highs[i] - ema21[i]) / highs[i] <= EMA_PROXIMITY_THRESHOLD)
                    ema50_inside = (lows[i] <= ema50[i] <= highs[i]) or (
                            abs(highs[i] - ema50[i]) / highs[i] <= EMA_PROXIMITY_THRESHOLD)
                    is_near_ema = ema21_inside or ema50_inside
                    is_trend_down = closes[i] < ema200[i] and ema50[i] > ema21[i]
                    if is_trend_down:
                        sell_idx.append(i)
                        sell_val.append(highs[i])
                        print(f"SELL index: {i}, rsi: {rsi[i]}")

                    waiting_sell_confirmation = True  # Trigger check for the next candle
                    is_enter = True
                    lower_band_raw = None
                    upper_band_raw = None

            last_highs = highs[i - lookback + 1: i + 1]
            last_lows = lows[i - lookback + 1: i + 1]

            channel_high = np.max(last_highs)
            channel_low = np.min(last_lows)
            channel_width = channel_high - channel_low

            current_price = closes[i]

            if current_price == 0:
                continue  # Usa 'continue' em vez de return para o loop não morrer a meio do histórico

            range_percent = channel_width / current_price

            # 2. CAPTURA E BLOQUEIO (Gatilhada quando o mercado comprime < 2%)
            if range_percent < 0.01 and is_enter:
                range_percent_half = range_percent / 2
                # CORRIGIDO: Uma banda subtrai do preço, a outra soma!
                lower_band_raw = current_price - (current_price * range_percent_half)
                upper_band_raw = current_price + (current_price * range_percent_half)
                is_enter = False

            # 3. PREENCHIMENTO DO HISTÓRICO
            if lower_band_raw is not None and upper_band_raw is not None:
                # Se estiver trancado na consolidação, mantém os valores fixos (linhas horizontais)
                lower_band[i] = lower_band_raw
                upper_band[i] = upper_band_raw
            else:
                # Se estiver em modo dinâmico, acompanha o preço da vela atual
                lower_band[i] = lows[i]
                upper_band[i] = highs[i]

        # 3. Desenho Estruturado do Gráfico (Duplo Painel)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax1.set_facecolor('#131722')
        fig.patch.set_facecolor('#131722')

        # Renderização dos Candles Japoneses
        for i in range(n):
            candle_color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
            ax1.plot([i, i], [lows[i], highs[i]], color='white', linewidth=0.8, alpha=0.3)
            ax1.bar(i, abs(closes[i] - opens[i]), bottom=min(opens[i], closes[i]), width=0.6, color=candle_color)

        # Plot dos Canais de EMA
        # Canal Micro Reação (EMA 5 High/Low em Branco)
        ax1.plot(indices, lower_band, color='white', alpha=0.4, linewidth=1, label='Lower band')
        ax1.plot(indices, upper_band, color='white', alpha=0.4, linewidth=1, label='Upper band')
        # ax1.fill_between(indices, ema5_l, ema5_h, color='white', alpha=0.03)

        # Canal Pullback Estrutural (EMA 100 High/Low em Verde Água)
        ax1.plot(indices, ema21, color='#26a69a', alpha=0.7, linewidth=1.5, label='EMA 21')
        ax1.plot(indices, ema50, color='#26a69a', alpha=0.7, linewidth=1.5, label='EMA 50')
        ax1.fill_between(indices, ema21, ema50, color='#26a69a', alpha=0.08)

        # Média Móvel Macro (EMA 200 em Laranja)
        ax1.plot(indices, ema200, color='orange', linewidth=2, label='EMA 200 (Trend Direction)')

        # Impressão Visual dos Sinais Executados
        ax1.scatter(buy_idx, buy_val, marker=MarkerStyle('^'), color='lime', s=180, edgecolors='black',
                    label='PKO BUY Trigger', zorder=5)
        ax1.scatter(sell_idx, sell_val, marker=MarkerStyle('v'), color='red', s=180, edgecolors='black',
                    label='PKO SELL Trigger', zorder=5)

        ax1.scatter(bottom_idx, bottom_val, marker=MarkerStyle('o'), color='lime', s=180, edgecolors='black',
                    label='PKO BUY Trigger', zorder=5)
        ax1.scatter(top_idx, top_val, marker=MarkerStyle('o'), color='red', s=180, edgecolors='black',
                    label='PKO SELL Trigger', zorder=5)

        ax1.set_title(f"Soheil PKO Strategy Pro - {symbol}", color='white', fontsize=14, pad=15)
        ax1.legend(facecolor='#1b1e2e', edgecolor='white', labelcolor='white', loc='upper left')
        ax1.grid(True, color='#2a2e39', alpha=0.3)
        ax1.tick_params(colors='white')

        # Painel Inferior: Força de Tendência (ADX)
        """
        ax2.set_facecolor('#131722')
        ax2.plot(indices, adx, color='#9b59b6', linewidth=2, label='ADX Matrix')

        # Níveis horizontais clássicos de validação do ADX
        # ax2.axhline(22, color='yellow', linestyle='--', alpha=0.4, label='Trend Threshold (22)')
        # ax2.axhline(40, color='red', linestyle=':', alpha=0.3)
    
        ax2.set_ylim(0, 70)
        ax2.tick_params(colors='white')
        ax2.grid(True, color='#2a2e39', alpha=0.2)
        ax2.legend(facecolor='#1b1e2e', edgecolor='white', labelcolor='white', loc='upper left')
        """

        ax2.plot(indices, rsi, color='#9b59b6', linewidth=2)
        # ax2.plot(indices, rsi_higher, color='#368921', linewidth=2)
        ax2.plot(indices, rsi_ema, color="#368921", linewidth=2)
        # ax2.plot(indices, rsi_ema, color="#368921", linewidth=2)
        ax2.axhline(70, color='red', linestyle='--', alpha=0.3)
        ax2.axhline(50, color='yellow', linestyle='--', alpha=0.3)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.3)
        ax2.set_facecolor('#131722')
        ax2.set_ylim(0, 100)

        ax2.plot(indices, rsi_h, color='#9b59b6', linewidth=2)
        # ax2.plot(indices, rsi_higher, color='#368921', linewidth=2)
        ax2.plot(indices, rsi_ema_h, color="#368921", linewidth=2)
        # ax2.plot(indices, rsi_ema, color="#368921", linewidth=2)
        ax2.axhline(70, color='red', linestyle='--', alpha=0.3)
        ax2.axhline(50, color='yellow', linestyle='--', alpha=0.3)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.3)
        ax2.set_facecolor('#131722')
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_test(ohlcv, ohlcvHigher, symbol: str):
        opens, highs, lows, closes, volumes, dates = ohlcv.opens, ohlcv.highs, ohlcv.lows, ohlcv.closes, ohlcv.volumes, ohlcv.dates
        n = len(closes)
        indices = np.arange(n)

        # 1. Indicadores Base (Canais de EMA e Momentum)
        utils = BaseIndicatorsUtils(ohlcv)
        utils_higher = BaseIndicatorsUtils(ohlcvHigher)
        ema21 = utils.ema(21)
        ema50 = utils.ema(50)
        ema200 = utils.ema_list(closes, 200)
        adx = utils.adx()
        atr = utils.atr()

        # RSIs Micro (15m) e Macro (1h)
        rsi, rsi_ema = utils.rsi(period=14, signal_period=14)
        rsi_h, rsi_ema_h = utils_higher.rsi(period=14, signal_period=14)

        # --- A TUA PRIMEIRA IDEIA: MAPEAMENTO EM DEGRAU (4 CANDLES POR VELA DE 1H) ---
        ratio = 4  # 1 candle de 1h = 4 candles de 15m

        # 1. Repete cada valor de 1h 4 vezes consecutivas
        rsi_h_stepped = np.repeat(rsi_h, ratio)
        rsi_ema_h_stepped = np.repeat(rsi_ema_h, ratio)

        # 2. Corta ou ajusta diretamente ao tamanho 'n' de 15m (SEM SHIFT MANUAL)
        # O timestamp do fecho da vela de 1h já alinha com o fecho do 4º candle de 15m
        n_valid = min(len(rsi_h_stepped), n)

        rsi_h_aligned = np.full(n, np.nan)
        rsi_ema_h_aligned = np.full(n, np.nan)

        rsi_h_aligned[:n_valid] = rsi_h_stepped[:n_valid]
        rsi_ema_h_aligned[:n_valid] = rsi_ema_h_stepped[:n_valid]
        # Filtro de Suavização (Kalman)
        ema21 = utils.kalman_filter(ema21)
        ema50 = utils.kalman_filter(ema50)
        ema200 = utils.kalman_filter(ema200)

        # Listas para Armazenamento de Sinais
        buy_idx, buy_val = [], []
        sell_idx, sell_val = [], []

        lower_band = np.full(n, np.nan)
        upper_band = np.full(n, np.nan)

        lookback = 10
        is_enter = True
        lower_band_raw = None
        upper_band_raw = None

        top_idx, top_val = [], []
        bottom_idx, bottom_val = [], []

        EMA_PROXIMITY_THRESHOLD = 0.001

        # --- LOOP DE EXECUÇÃO DA ESTRATÉGIA ---
        for i in range(lookback, n):
            current_price = closes[i]

            if current_price == 0:
                continue

            # --- DETETAR BREAKOUT DAS BANDAS ---
            if lower_band_raw is not None and upper_band_raw is not None:

                if current_price > upper_band_raw:
                    ema21_inside = (lows[i] <= ema21[i] <= highs[i]) or (
                            abs(lows[i] - ema21[i]) / lows[i] <= EMA_PROXIMITY_THRESHOLD)
                    ema50_inside = (lows[i] <= ema50[i] <= highs[i]) or (
                            abs(lows[i] - ema50[i]) / lows[i] <= EMA_PROXIMITY_THRESHOLD)
                    is_near_ema = ema21_inside or ema50_inside
                    is_trend_up = closes[i] > ema200[i] and ema50[i] < ema21[i]

                    if is_trend_up:
                        buy_idx.append(i)
                        buy_val.append(lows[i])
                        print(f"BUY index: {i}, rsi_15m: {rsi[i]:.1f}, rsi_1h: {rsi_h_aligned[i]:.1f}")

                    is_enter = True
                    lower_band_raw = None
                    upper_band_raw = None

                elif current_price < lower_band_raw:
                    ema21_inside = (lows[i] <= ema21[i] <= highs[i]) or (
                            abs(highs[i] - ema21[i]) / highs[i] <= EMA_PROXIMITY_THRESHOLD)
                    ema50_inside = (lows[i] <= ema50[i] <= highs[i]) or (
                            abs(highs[i] - ema50[i]) / highs[i] <= EMA_PROXIMITY_THRESHOLD)
                    is_near_ema = ema21_inside or ema50_inside
                    is_trend_down = closes[i] < ema200[i] and ema50[i] > ema21[i]

                    if is_trend_down:
                        sell_idx.append(i)
                        sell_val.append(highs[i])
                        print(f"SELL index: {i}, rsi_15m: {rsi[i]:.1f}, rsi_1h: {rsi_h_aligned[i]:.1f}")

                    is_enter = True
                    lower_band_raw = None
                    upper_band_raw = None

            # --- DETETAR COMPRESSÃO (PANELA DE PRESSÃO) ---
            last_highs = highs[i - lookback + 1: i + 1]
            last_lows = lows[i - lookback + 1: i + 1]

            channel_high = np.max(last_highs)
            channel_low = np.min(last_lows)
            channel_width = channel_high - channel_low

            range_percent = channel_width / current_price

            # Captura quando o mercado comprime < 1%
            if range_percent < 0.01 and is_enter:
                range_percent_half = range_percent / 2
                lower_band_raw = current_price - (current_price * range_percent_half)
                upper_band_raw = current_price + (current_price * range_percent_half)
                is_enter = False

            # Preenchimento do Histórico das Bandas
            if lower_band_raw is not None and upper_band_raw is not None:
                lower_band[i] = lower_band_raw
                upper_band[i] = upper_band_raw
            else:
                lower_band[i] = lows[i]
                upper_band[i] = highs[i]

        # --- DESENHO ESTRUTURADO DO GRÁFICO (DUPLO PAINEL) ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax1.set_facecolor('#131722')
        fig.patch.set_facecolor('#131722')

        # Renderização das Velas Japonesas
        for i in range(n):
            candle_color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
            # ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color='black', linewidth=1)  # high-low
            # ax.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=5)

            ax1.plot([i, i], [lows[i], highs[i]], color='white', linewidth=0.8, alpha=0.3)
            ax1.bar(i, abs(closes[i] - opens[i]), bottom=min(opens[i], closes[i]), width=0.6, color=candle_color)

        # Plot das Bandas e Médias Móveis no Painel Principal
        ax1.plot(indices, lower_band, color='white', alpha=0.4, linewidth=1, label='Lower band')
        ax1.plot(indices, upper_band, color='white', alpha=0.4, linewidth=1, label='Upper band')
        ax1.plot(indices, ema21, color='#26a69a', alpha=0.7, linewidth=1.5, label='EMA 21')
        ax1.plot(indices, ema50, color='#26a69a', alpha=0.7, linewidth=1.5, label='EMA 50')
        ax1.fill_between(indices, ema21, ema50, color='#26a69a', alpha=0.08)
        ax1.plot(indices, ema200, color='orange', linewidth=2, label='EMA 200 (Trend Direction)')

        # Marcadores Visuais de Compra e Venda
        ax1.scatter(buy_idx, buy_val, marker=MarkerStyle('^'), color='lime', s=180, edgecolors='black',
                    label='BUY Trigger', zorder=5)
        ax1.scatter(sell_idx, sell_val, marker=MarkerStyle('v'), color='red', s=180, edgecolors='black',
                    label='SELL Trigger', zorder=5)

        ax1.set_title(f"Soheil PKO Strategy Pro - Multi-RSI (15m/1h) - {symbol}", color='white', fontsize=14, pad=15)
        ax1.legend(facecolor='#1b1e2e', edgecolor='white', labelcolor='white', loc='upper left')
        ax1.grid(True, color='#2a2e39', alpha=0.3)
        ax1.tick_params(colors='white')

        # --- PAINEL INFERIOR: PAINEL DUAL RSI (15m & 1h) ---
        ax2.set_facecolor('#131722')

        # 1. RSI 15m (Oscilação rápida em roxo)
        ax2.plot(indices, rsi, color='#9b59b6', linewidth=1.5, label='RSI 15m (Micro)', alpha=0.85)

        # 2. RSI 1h (Degraus sólidos em Laranja / Estilo TradingView)
        ax2.plot(indices, rsi_h_aligned, color='#f39c12', linewidth=2.0, label='RSI 1h (Macro Step)',
                 drawstyle='steps-post')

        # Níveis Críticos Horizontais
        ax2.axhline(70, color='#ef5350', linestyle='--', alpha=0.6, label='Sobrecompra (70)')
        ax2.axhline(50, color='gray', linestyle=':', alpha=0.3)
        ax2.axhline(30, color='#26a69a', linestyle='--', alpha=0.6, label='Sobrevenda (30)')

        # Sombreamento visual para zonas de confluência
        ax2.fill_between(indices, 70, 100, color='#ef5350', alpha=0.08)
        ax2.fill_between(indices, 0, 30, color='#26a69a', alpha=0.08)

        ax2.set_ylim(0, 100)
        ax2.tick_params(colors='white')
        ax2.grid(True, color='#2a2e39', alpha=0.2)
        ax2.legend(facecolor='#1b1e2e', edgecolor='white', labelcolor='white', loc='upper left', fontsize=9)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_high_volume_pivots_bigbeluga__(ohlcv, symbol: str):
        opens, highs, lows, closes, volumes = ohlcv.opens, ohlcv.highs, ohlcv.lows, ohlcv.closes, ohlcv.volumes
        n = len(closes)
        indices = np.arange(n)

        # -------------------------------------------------------------------------
        # 1. PARÂMETROS & INDICADORES BASE
        # -------------------------------------------------------------------------
        res_len = 40
        sup_len = 40
        vol_avg_len = 20
        vol_mult = 1.2

        # ATR (Usando 200 períodos para padding idêntico ao Pine Script)
        utils = BaseIndicatorsUtils(ohlcv)
        atr = utils.atr(200) if hasattr(utils, 'atr') else [np.std(closes[max(0, i - 200):i + 1]) for i in range(n)]

        # Validação de Volume Superior
        vol_sma = [np.mean(volumes[max(0, i - vol_avg_len + 1):i + 1]) for i in range(n)]

        # Listas para guardar as coordenadas das caixas de Suporte/Resistência
        res_boxes = []
        sup_boxes = []

        # Listas para os Marcadores de Sinais de Estrutura no Gráfico
        res_break_idx, res_break_val = [], []
        sup_break_idx, sup_break_val = [], []
        res_retest_idx, res_retest_val = [], []
        sup_retest_idx, sup_retest_val = [], []

        # Sinais EXECUTÁVEIS (Os teus Triângulos)
        buy_signals_idx, buy_signals_val = [], []
        sell_signals_idx, sell_signals_val = [], []

        # Estados das Zonas
        last_res_idx, last_res_top, last_res_bot = None, None, None
        last_sup_idx, last_sup_top, last_sup_bot = None, None, None
        res_broken, sup_broken = False, False

        # Controladores para evitar spam de sinais na mesma zona
        last_buy_idx = -5
        last_sell_idx = -5

        # -------------------------------------------------------------------------
        # 2. LÓGICA DE DETEÇÃO DE PIVOTS & QUEBRAS
        # -------------------------------------------------------------------------
        for i in range(max(res_len, sup_len), n):

            # --- DETEÇÃO DE PIVOT HIGH (RESISTÊNCIA) ---
            is_pivot_high = True
            target_idx_hi = i - res_len
            for k in range(target_idx_hi - res_len, i + 1):
                if highs[k] > highs[target_idx_hi]:
                    is_pivot_high = False
                    break

            valid_pivot_high = is_pivot_high and (volumes[target_idx_hi] > vol_sma[target_idx_hi] * vol_mult)

            if valid_pivot_high:
                res_broken = False
                last_res_idx = target_idx_hi
                pivot_body_top = max(opens[target_idx_hi], closes[target_idx_hi])
                last_res_top = pivot_body_top + atr[target_idx_hi]
                last_res_bot = pivot_body_top
                res_boxes.append(
                    {"left": last_res_idx, "right": i, "top": last_res_top, "bottom": last_res_bot, "active": True})

            # --- DETEÇÃO DE PIVOT LOW (SUPORTE) ---
            is_pivot_low = True
            target_idx_lo = i - sup_len
            for k in range(target_idx_lo - sup_len, i + 1):
                if lows[k] < lows[target_idx_lo]:
                    is_pivot_low = False
                    break

            valid_pivot_low = is_pivot_low and (volumes[target_idx_lo] > vol_sma[target_idx_lo] * vol_mult)

            if valid_pivot_low:
                sup_broken = False
                last_sup_idx = target_idx_lo
                pivot_body_bot = min(opens[target_idx_lo], closes[target_idx_lo])
                last_sup_top = pivot_body_bot
                last_sup_bot = pivot_body_bot - atr[target_idx_lo]
                sup_boxes.append(
                    {"left": last_sup_idx, "right": i, "top": last_sup_top, "bottom": last_sup_bot, "active": True})

            # --- ATUALIZAÇÃO EM TEMPO REAL DAS CAIXAS EXISTENTES & RETESTES ---
            if last_res_idx is not None and len(res_boxes) > 0:
                res_boxes[-1]["right"] = i
                if highs[i - 1] >= last_res_bot and highs[i] < last_res_bot:
                    res_retest_idx.append(i)
                    res_retest_val.append(highs[i])
                if closes[i] > last_res_top and not res_broken:
                    res_break_idx.append(i)
                    res_break_val.append(closes[i])
                    res_broken = True
                    res_boxes[-1]["active"] = False

            if last_sup_idx is not None and len(sup_boxes) > 0:
                sup_boxes[-1]["right"] = i
                if lows[i - 1] <= last_sup_top and lows[i] > last_sup_top:
                    sup_retest_idx.append(i)
                    sup_retest_val.append(lows[i])
                if closes[i] < last_sup_bot and not sup_broken:
                    sup_break_idx.append(i)
                    sup_break_val.append(closes[i])
                    sup_broken = True
                    sup_boxes[-1]["active"] = False

            # --- GATILHOS DOS TEUS TRIÂNGULOS EM TEMPO REAL ---
            # 🟢 COMPRA: Rejeição Controlada no teto do Suporte Ativo
            for box in sup_boxes:
                if box["active"] and i > box["left"] + sup_len:
                    if lows[i] <= box["top"] and closes[i] > box["top"] and closes[i] > opens[i]:
                        if i - last_buy_idx >= 3:  # Cooldown para limpar o gráfico
                            buy_signals_idx.append(i)
                            buy_signals_val.append(lows[i])
                            last_buy_idx = i
                            break

                            # 🔴 VENDA: Rejeição Controlada na base da Resistência Ativa
            for box in res_boxes:
                if box["active"] and i > box["left"] + res_len:
                    if highs[i] >= box["bottom"] and closes[i] < box["bottom"] and closes[i] < opens[i]:
                        if i - last_sell_idx >= 3:
                            sell_signals_idx.append(i)
                            sell_signals_val.append(highs[i])
                            last_sell_idx = i
                            break

        # -------------------------------------------------------------------------
        # 3. MATPLOTLIB RENDER
        # -------------------------------------------------------------------------
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.set_facecolor('#131722')

        # Desenho dos Candles
        for i in range(n):
            color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
            ax1.plot([i, i], [lows[i], highs[i]], color='white', linewidth=0.8, alpha=0.3)
            ax1.bar(i, abs(closes[i] - opens[i]), bottom=min(opens[i], closes[i]), width=0.6, color=color)

        # PINTAR AS CAIXAS DE RESISTÊNCIA
        for box in res_boxes:
            width = box["right"] - box["left"]
            height = box["top"] - box["bottom"]
            box_color = '#2196f3' if box["active"] else '#ffc13b'
            alpha_fill = 0.12 if box["active"] else 0.02
            line_style = '-' if box["active"] else '--'
            ax1.bar(box["left"] + width / 2, height, bottom=box["bottom"], width=width,
                    color=box_color, alpha=alpha_fill, edgecolor=box_color, linewidth=1, linestyle=line_style)

        # PINTAR AS CAIXAS DE SUPORTE
        for box in sup_boxes:
            width = box["right"] - box["left"]
            height = box["top"] - box["bottom"]
            box_color = '#ffc13b' if box["active"] else '#2196f3'
            alpha_fill = 0.12 if box["active"] else 0.02
            line_style = '-' if box["active"] else '--'
            ax1.bar(box["left"] + width / 2, height, bottom=box["bottom"], width=width,
                    color=box_color, alpha=alpha_fill, edgecolor=box_color, linewidth=1, linestyle=line_style)

        # PLOT DOS MARCADORES DE ESTRUTURA BASE
        if res_break_idx:
            ax1.scatter(res_break_idx, res_break_val, marker=MarkerStyle('^'), color='#ffc13b', s=120,
                        edgecolors='black', label='Res Breakout', zorder=5)
        if sup_break_idx:
            ax1.scatter(sup_break_idx, sup_break_val, marker=MarkerStyle('v'), color='#2196f3', s=120,
                        edgecolors='black', label='Sup Breakdown', zorder=5)
        if res_retest_idx:
            ax1.scatter(res_retest_idx, res_retest_val, marker=MarkerStyle('o'), color='#ef5350', s=60,
                        edgecolors='black', label='Res Hold Retest', zorder=5)
        if sup_retest_idx:
            ax1.scatter(sup_retest_idx, sup_retest_val, marker=MarkerStyle('o'), color='#26a69a', s=60,
                        edgecolors='black', label='Sup Hold Retest', zorder=5)

        # -------------------------------------------------------------------------
        # PLOT DOS SINAIS DE EXECUÇÃO (CORRIGIDO: FORA DOS BLOCKS DE IDENTAÇÃO)
        # -------------------------------------------------------------------------
        if buy_signals_idx:
            ax1.scatter(buy_signals_idx, buy_signals_val, marker=MarkerStyle('^'),
                        color='#00ff88', s=250, edgecolors='black', linewidths=1.5,
                        label='EXECUTE BUY (Sup Hold)', zorder=6)

        if sell_signals_idx:
            ax1.scatter(sell_signals_idx, sell_signals_val, marker=MarkerStyle('v'),
                        color='#ff0055', s=250, edgecolors='black', linewidths=1.5,
                        label='EXECUTE SELL (Res Rejection)', zorder=6)

        # Configurações de Design finais (Cá fora para englobar tudo de forma limpa)
        ax1.set_title(f"High Volume Pivot Zones [BigBeluga Fixed] - {symbol}", color='white', fontsize=14)
        ax1.legend(facecolor='#1b1e2e', edgecolor='white', labelcolor='white', loc='upper left')
        ax1.grid(True, color='#2a2e39', alpha=0.3)

        ax1.set_xlim(-5, n + 15)
        ax1.tick_params(colors='white')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_high_volume_pivots_bigbeluga_old(ohlcv, symbol: str):
        opens, highs, lows, closes, volumes = ohlcv.opens, ohlcv.highs, ohlcv.lows, ohlcv.closes, ohlcv.volumes
        n = len(closes)

        # -------------------------------------------------------------------------
        # 1. PARÂMETROS BASE & VOLATILIDADE
        # -------------------------------------------------------------------------
        period = 40

        # Cálculo limpo do ATR para definir a altura da zona de tolerância
        try:
            utils = BaseIndicatorsUtils(ohlcv)
            atr = utils.atr(14)
        except Exception:
            # Fallback simples caso não encontre a classe utils
            atr = [np.std(closes[max(0, i - 14):i + 1]) for i in range(n)]

        # Listas para armazenar os sinais de execução reais
        buy_signals_idx, buy_signals_val = [], []
        sell_signals_idx, sell_signals_val = [], []

        # Listas para desenhar as linhas das zonas de forma limpa no gráfico
        res_line_top, res_line_bot = np.full(n, np.nan), np.full(n, np.nan)
        sup_line_top, sup_line_bot = np.full(n, np.nan), np.full(n, np.nan)

        res_dinamica, sup_dinamico = np.full(n, np.nan), np.full(n, np.nan)

        # Controladores de Cooldown
        last_buy_idx = -5
        last_sell_idx = -5

        last_confirmed_sell_rsi = 50.0
        last_confirmed_buy_rsi = 50.0

        rsi, rsi_ema = utils.rsi()

        pending_sell_trigger_idx = -1
        pending_buy_trigger_idx = -1
        max_wait_candles = 5
        ema8 = utils.ema(8)

        ema21 = utils.ema(21)
        ema9 = utils.ema(9)

        # -------------------------------------------------------------------------
        # 2. LOOP ÚNICO TEMPORAL (EXATAMENTE COMO EM LIVE TRADING)
        # -------------------------------------------------------------------------
        for i in range(period, n):
            # Calcular os extremos estritos das últimas 40 velas (exclui a vela atual 'i')
            max_40 = np.max(highs[i - period:i])
            min_40 = np.min(lows[i - period:i])
            current_atr = atr[i]

            # Guardar as coordenadas das zonas para o gráfico desenhar no presente
            res_line_top[i] = max_40 + current_atr
            res_line_bot[i] = max_40
            sup_line_top[i] = min_40
            sup_line_bot[i] = min_40 - current_atr

            preco_medio = ema21[i]
            current_atr = atr[i]

            # 2. A linha de resistência afasta-se ou aproxima-se dependendo do ATR
            # Se o ATR estiver gigante (foguete), a linha vai para o céu.
            # Se o ATR encolher (exaustão), a linha desce e o preço toca-lhe.
            res_dinamica[i] = max_40 + (1.2 * current_atr)
            sup_dinamico[i] = min_40 - (1.2 * current_atr)

            # 🔴 GATILHO DE VENDA (Rejeição com confirmação na Zona de Resistência)
            # 1. A máxima testou a base da zona (max_40)
            # 2. O fecho respeitou o teto da zona (max_40 + ATR)
            # 3. O fecho recuou para baixo da linha de gatilho e o candle fechou vermelho
            """
            if highs[i] >= max_40 and closes[i] <= (max_40 + current_atr):
                if closes[i] < max_40 and closes[i] < opens[i]:
                    # if i - last_sell_idx >= 4:
                    sell_signals_idx.append(i)
                    sell_signals_val.append(highs[i])
                    last_sell_idx = i

            # 🟢 GATILHO DE COMPRA (Defesa com confirmação na Zona de Suporte)
            # 1. A mínima testou o teto da zona (min_40)
            # 2. O fecho respeitou o fundo da zona (min_40 - ATR)
            # 3. O fecho segurou acima da linha de gatilho e o candle fechou verde
            if lows[i] <= min_40 and closes[i] >= (min_40 - current_atr):
                if closes[i] > min_40 and closes[i] > opens[i]:
                    # if i - last_buy_idx >= 4:
                    buy_signals_idx.append(i)
                    buy_signals_val.append(lows[i])
                    last_buy_idx = i
            """
            """
            if highs[i] >= max_40:
                # if closes[i] < max_40:
                # if i - last_sell_idx >= 4:
                sell_signals_idx.append(i)
                sell_signals_val.append(highs[i])
                last_sell_idx = i

                # 🟢 GATILHO DE COMPRA (Defesa com confirmação na Zona de Suporte)
                # 1. A mínima testou o teto da zona (min_40)
                # 2. O fecho respeitou o fundo da zona (min_40 - ATR)
                # 3. O fecho segurou acima da linha de gatilho e o candle fechou verde
            if lows[i] <= min_40:
                # if closes[i] > min_40:
                # if i - last_buy_idx >= 4:
                buy_signals_idx.append(i)
                buy_signals_val.append(lows[i])
                last_buy_idx = i
            """

            if highs[i] >= max_40:
                pending_sell_trigger_idx = i  # Alarme ativo

            if lows[i] <= min_40:
                pending_buy_trigger_idx = i  # Alarme ativo

            # 🔴 Filtro para VENDA (Falsos Topos)
            if pending_sell_trigger_idx != -1:
                if i - pending_sell_trigger_idx <= 5:

                    # CONDICIONAL GEOMÉTRICA PERFEITA:
                    # 1. O preço recuou e testou a EMA rápida (lows[i] <= ema_rapida)
                    # 2. FILTRO: Só vende se NÃO for uma tendência de alta parabólica
                    #    (ou seja, permitimos o trade se a EMA rápida já estiver a curvar/abaixo da lenta,
                    #    OU se o preço atual já conseguiu fechar abaixo da EMA lenta, quebrando a tendência)
                    if lows[i] <= ema9[i] and (ema9[i] < ema21[i] or closes[i] < ema21[i]):
                        if i - last_sell_idx >= 6:
                            sell_signals_idx.append(i)
                            sell_signals_val.append(highs[i])
                            last_sell_idx = i
                        pending_sell_trigger_idx = -1  # Reset
                else:
                    pending_sell_trigger_idx = -1  # Janela expirou (Falso Topo evitado)

            # 🟢 Filtro para COMPRA (Falsos Fundos)
            if pending_buy_trigger_idx != -1:
                if i - pending_buy_trigger_idx <= 5:

                    # 1. O preço recuperou e testou a EMA rápida para cima
                    # 2. FILTRO: Só compra se a EMA rápida estiver acima da lenta OU se o fecho já superou a lenta
                    if highs[i] >= ema9[i] and (ema9[i] > ema21[i] or closes[i] > ema21[i]):
                        if i - last_buy_idx >= 6:
                            buy_signals_idx.append(i)
                            buy_signals_val.append(lows[i])
                            last_buy_idx = i
                        pending_buy_trigger_idx = -1
                else:
                    pending_buy_trigger_idx = -1

        # -------------------------------------------------------------------------
        # 3. RENDERING DO GRÁFICO
        # -------------------------------------------------------------------------
        # fig, (ax1,ax2) = plt.subplots(figsize=(16, 8))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax1.set_facecolor('#131722')

        # Desenho dos Candles convencionais
        for i in range(n):
            color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
            ax1.plot([i, i], [lows[i], highs[i]], color='white', linewidth=0.8, alpha=0.3)
            ax1.bar(i, abs(closes[i] - opens[i]), bottom=min(opens[i], closes[i]), width=0.6, color=color)

        # Desenhar as Zonas de Liquidez como bandas contínuas sombreadas
        ax1.plot(res_dinamica, color='#2196f3', linestyle='--', linewidth=1, alpha=0.5, label='Res Trigger Line 2')
        # ax1.plot(res_line_bot, color='#2196f3', linestyle='--', linewidth=1, alpha=0.5, label='Res Trigger Line')
        # ax1.fill_between(range(n), res_line_bot, res_line_top, color='#2196f3', alpha=0.08, label='Resist. ATR Zone')

        ax1.plot(sup_dinamico, color='#ffc13b', linestyle='--', linewidth=1, alpha=0.5, label='Sup Trigger Line 2')
        # ax1.plot(sup_line_bot, color='#2196f3', linestyle='--', linewidth=1, alpha=0.5, label='Res Trigger Line')
        # ax1.fill_between(range(n), sup_line_bot, sup_line_top, color='#ffc13b', alpha=0.08, label='Support ATR Zone')

        # Desenhar os Triângulos Grandes nos Sinais Reais Filtrados
        if buy_signals_idx:
            ax1.scatter(buy_signals_idx, buy_signals_val, marker=MarkerStyle('^'),
                        color='#00ff88', s=250, edgecolors='black', linewidths=1.5,
                        label='EXECUTE BUY (Rejection)', zorder=6)

        if sell_signals_idx:
            ax1.scatter(sell_signals_idx, sell_signals_val, marker=MarkerStyle('v'),
                        color='#ff0055', s=250, edgecolors='black', linewidths=1.5,
                        label='EXECUTE SELL (Rejection)', zorder=6)

        indices = np.arange(n)
        ema200 = utils.ema(200)
        # ax1.plot(indices, ema200, color='blue', linestyle='-', linewidth=1.5, label='EMA 200')

        ema50 = utils.ema(50)
        # ax1.plot(indices, ema50, color='red', linestyle='-', linewidth=1.5, label='EMA 50')

        # ema21 = utils.ema(21)
        ax1.plot(indices, ema21, color='green', linestyle='-', linewidth=1.5, label='EMA 21')

        # ema9 = utils.ema(8)
        ax1.plot(indices, ema9, color='orange', linestyle='-', linewidth=1.5, label='EMA 8')

        ema5 = utils.ema(3)
        # ax1.plot(indices, ema5, color='brown', linestyle='-', linewidth=1.5, label='EMA 5')

        ax1.plot(indices, ema5, color='brown', linestyle='-', linewidth=1.5, label='EMA 5')

        # Layout de Produção
        ax1.set_title(f"Dynamic Donchian ATR Rejection Bot - {symbol}", color='white', fontsize=14)
        ax1.legend(facecolor='#1b1e2e', edgecolor='white', labelcolor='white', loc='upper left')
        ax1.grid(True, color='#2a2e39', alpha=0.2)
        ax1.set_xlim(period - 5, n + 5)
        ax1.tick_params(colors='white')

        indicators_utils = IndicatorsUtils(ohlcv)

        rsi, rsi_ema = indicators_utils.rsi()

        ax2.plot(indices, rsi, color='#9b59b6', linewidth=2)
        # ax2.plot(indices, rsi_higher, color='#368921', linewidth=2)
        ax2.plot(indices, rsi_ema, color="#368921", linewidth=2)
        # ax2.plot(indices, rsi_ema, color="#368921", linewidth=2)
        ax2.axhline(70, color='red', linestyle='--', alpha=0.3)
        ax2.axhline(50, color='yellow', linestyle='--', alpha=0.3)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.3)
        ax2.set_facecolor('#131722')
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_high_volume_pivots_bigbeluga(ohlcv, symbol: str):
        import numpy as np
        import matplotlib.pyplot as plt

        opens, highs, lows, closes, volumes = ohlcv.opens, ohlcv.highs, ohlcv.lows, ohlcv.closes, ohlcv.volumes
        n = len(closes)

        # -------------------------------------------------------------------------
        # 1. PARÂMETROS BASE & VOLATILIDADE (Configurações do BigBeluga)
        # -------------------------------------------------------------------------
        period = 200  # Janela de análise da regressão polinomial

        # Instanciação unificada dos utilitários para evitar NameError nas EMAs
        utils = BaseIndicatorsUtils(ohlcv)
        try:
            atr = utils.atr(14)
        except Exception:
            atr = [np.std(closes[max(0, i - 14):i + 1]) for i in range(n)]

        # Extração dos dados pré-calculados do teu pacote nativo da TV
        tv_utils = TvIndicatorsUtils(ohlcv)
        calculate_poly_regression_channel = tv_utils.calculate_poly_regression_channel()
        baselines = calculate_poly_regression_channel['baseline']
        chans_high = calculate_poly_regression_channel['chan_high']
        chans_low = calculate_poly_regression_channel['chan_low']
        pocs_level = calculate_poly_regression_channel['poc_level']
        is_bullish_arr = calculate_poly_regression_channel['is_bullish']

        # Listas para Sinais de EXAUSTÃO (Caçador de Topos e Fundos - Triângulos)
        buy_exhaust_idx, buy_exhaust_val = [], []
        sell_exhaust_idx, sell_exhaust_val = [], []

        # Listas para Sinais de RETESTE (Seguidor de Tendência na Baseline - Círculos)
        buy_retest_idx, buy_retest_val = [], []
        sell_retest_idx, sell_retest_val = [], []

        # Controladores de Cooldown e Gatilhos
        last_exhaust_buy = -10
        last_exhaust_sell = -10
        last_retest_buy = -10
        last_retest_sell = -10

        pending_sell_trigger_idx = -1
        pending_buy_trigger_idx = -1

        ema21 = utils.ema(21)
        ema9 = utils.ema(9)
        indicators_utils = IndicatorsUtils(ohlcv)
        rsi, rsi_ema = indicators_utils.rsi()

        # -------------------------------------------------------------------------
        # 2. LOOP ÚNICO TEMPORAL (MIMICA PERFEITA DE LIVE TRADING)
        # -------------------------------------------------------------------------
        for i in range(period, n):
            print(f" i = {i} pocs_level = {pocs_level[i]}, is_bullish_arr = {is_bullish_arr[i]}")
            # Filtro de segurança: Ignora se os indicadores ainda não estiverem calculados (NaN)
            if np.isnan(baselines[i]) or np.isnan(chans_high[i]) or np.isnan(chans_low[i]):
                continue

            # Cache de variáveis locais para evitar conflitos de sintaxe
            val_ema9 = ema9[i]
            val_ema21 = ema21[i]
            val_rsi = rsi[i]

            # -------------------------------------------------------------------------
            # ESTRATÉGIA A: CAÇADOR DE TOPOS E FUNDOS (Exaustão em +/- 3 SD)
            # -------------------------------------------------------------------------
            if highs[i] >= chans_high[i]:
                pending_sell_trigger_idx = i  # Ativa alarme de Topo

            if lows[i] <= chans_low[i]:
                pending_buy_trigger_idx = i  # Ativa alarme de Fundo

            # 🔴 Confirmação de Venda por Exaustão (Gatilho Curto)
            if pending_sell_trigger_idx != -1:
                if i - pending_sell_trigger_idx <= 5:
                    if (closes[i] < pocs_level[i] or lows[i] <= val_ema9) and (
                            val_ema9 < val_ema21 or closes[i] < val_ema21):
                        if i - last_exhaust_sell >= 8:  # Cooldown
                            sell_exhaust_idx.append(i)
                            sell_exhaust_val.append(highs[i])
                            last_exhaust_sell = i
                        pending_sell_trigger_idx = -1
                else:
                    pending_sell_trigger_idx = -1

            # 🟢 Confirmação de Compra por Exaustão
            if pending_buy_trigger_idx != -1:
                if i - pending_buy_trigger_idx <= 5:
                    if (closes[i] > pocs_level[i] or highs[i] >= val_ema9) and (
                            val_ema9 > val_ema21 or closes[i] > val_ema21):
                        if i - last_exhaust_buy >= 8:
                            buy_exhaust_idx.append(i)
                            buy_exhaust_val.append(lows[i])
                            last_exhaust_buy = i
                        pending_buy_trigger_idx = -1
                else:
                    pending_buy_trigger_idx = -1

            # -------------------------------------------------------------------------
            # ESTRATÉGIA B: SEGUIDOR DE TENDÊNCIA (Reteste / Pullback na Baseline)
            # -------------------------------------------------------------------------
            # Usamos o array nativo is_bullish extraído diretamente do teu indicador da TV
            # state_bullish = is_bullish_arr[i - 1] and val_ema9 > val_ema21
            # state_bearish = not is_bullish_arr[i - 1] and val_ema9 < val_ema21

            # 🟢 COMPRA NO RETESTE: Tendência de alta + Mínima testa a baseline + Vela fecha verde a rejeitar
            # -------------------------------------------------------------------------
            # ESTRATÉGIA B: SEGUIDOR DE TENDÊNCIA (Rompimento + Reteste da Baseline)
            # -------------------------------------------------------------------------

            # Olhamos para uma janela curta anterior (ex: últimas 5 velas antes da atual)
            # para verificar se o preço veio de BAIXO para CIMA da baseline (Rompimento Recente)
            veio_de_baixo = any(closes[i - k] < baselines[i - k] for k in range(2, 6))
            rompeu_recentemente_alta = veio_de_baixo and closes[i - 1] > baselines[i - 1]

            # Inverso para o cenário Bearish (Vem de CIMA, rompe para BAIXO)
            veio_de_cima = any(closes[i - k] > baselines[i - k] for k in range(2, 6))
            rompeu_recentemente_baixa = veio_de_cima and closes[i - 1] < baselines[i - 1]

            # 🟢 COMPRA NO RETESTE APÓS ROMPIMENTO:
            # 1. Rompeu recentemente de baixo para cima
            # 2. A mínima da vela atual vem testar/retestar a baseline (low <= baseline)
            # 3. O fecho atual segura-se acima da baseline (close > baseline) e a vela é verde
            if is_bullish_arr[i]:
                if closes[i] > baselines[i] and closes[i - 1] <= baselines[i - 1]:
                    # if rompeu_recentemente_alta and closes[i] > baselines[i] and closes[i] > opens[i]:
                    # Filtros adicionais de momentum e cooldown
                    # if val_ema9 > val_ema21 and val_rsi < 65 and i - last_retest_buy >= 6:
                    buy_retest_idx.append(i)
                    buy_retest_val.append(lows[i])
                    last_retest_buy = i

            # 🔴 VENDA NO RETESTE APÓS ROMPIMENTO:
            # 1. Rompeu recentemente de cima para baixo
            # 2. A máxima da vela atual vem testar/retestar a baseline (high >= baseline)
            # 3. O fecho atual segura-se abaixo da baseline (close < baseline) e a vela é vermelha
            if not is_bullish_arr[i]:
                if closes[i] < baselines[i] and closes[i - 1] >= baselines[i - 1]:
                    # if rompeu_recentemente_baixa and closes[i] < baselines[i] and closes[i] < opens[i]:
                    # Filtros adicionais de momentum e cooldown
                    # if val_ema9 < val_ema21 and val_rsi > 35 and i - last_retest_sell >= 6:
                    sell_retest_idx.append(i)
                    sell_retest_val.append(highs[i])
                    last_retest_sell = i

        # -------------------------------------------------------------------------
        # 3. RENDERING DO GRÁFICO (Estilo TradingView Pro)
        # -------------------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax1.set_facecolor('#131722')
        fig.patch.set_facecolor('#131722')

        # Candlesticks
        for i in range(n):
            color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
            ax1.plot([i, i], [lows[i], highs[i]], color='white', linewidth=0.8, alpha=0.3)
            ax1.bar(i, abs(closes[i] - opens[i]), bottom=min(opens[i], closes[i]), width=0.6, color=color)

        indices = np.arange(n)
        idx_validos = indices >= period

        # Plot das Curvas do Sistema BigBeluga
        ax1.plot(indices[idx_validos], baselines[idx_validos], color='#ffffff', linestyle='-', linewidth=1.5, alpha=0.6,
                 label='Baseline Polinomial')
        ax1.plot(indices[idx_validos], chans_high[idx_validos], color='#ff4444', linestyle='--', linewidth=1.0,
                 alpha=0.5, label='Canal Extremo (+3 SD)')
        ax1.plot(indices[idx_validos], chans_low[idx_validos], color='#44ff44', linestyle='--', linewidth=1.0,
                 alpha=0.5, label='Canal Extremo (-3 SD)')
        ax1.plot(indices[idx_validos], pocs_level[idx_validos], color='#ffa600', linestyle='-', linewidth=1.8,
                 alpha=0.8, label='POC de Volume')

        ax1.fill_between(indices[idx_validos], chans_low[idx_validos], chans_high[idx_validos], color='#888888',
                         alpha=0.03)

        # Médias de Tendência Auxiliares
        # ax1.plot(indices, ema21, color='#2196f3', linestyle='-', linewidth=1.0, alpha=0.7, label='EMA 21')
        # ax1.plot(indices, ema9, color='#ff9800', linestyle='-', linewidth=1.0, alpha=0.7, label='EMA 9')

        # 🎯 PLOT DOS SINAIS DE EXAUSTÃO
        if buy_exhaust_idx:
            ax1.scatter(buy_exhaust_idx, buy_exhaust_val, marker='^', color='#00ff88', s=220, edgecolors='black',
                        linewidths=1.5, label='EXHAUSTION BUY (Fundo)', zorder=6)
        if sell_exhaust_idx:
            ax1.scatter(sell_exhaust_idx, sell_exhaust_val, marker='v', color='#ff0055', s=220, edgecolors='black',
                        linewidths=1.5, label='EXHAUSTION SELL (Topo)', zorder=6)

        # 🎯 PLOT DOS SINAIS DE RETESTE
        if buy_retest_idx:
            ax1.scatter(buy_retest_idx, buy_retest_val, marker='o', color='#00bcff', s=140, edgecolors='white',
                        linewidths=1.2, label='TREND RETEST BUY', zorder=5)
        if sell_retest_idx:
            ax1.scatter(sell_retest_idx, sell_retest_val, marker='o', color='#e67e22', s=140, edgecolors='white',
                        linewidths=1.2, label='TREND RETEST SELL', zorder=5)

        ax1.set_title(f"Hybrid Polynomial Regression & Vol. Profile Bot - {symbol}", color='white', fontsize=14, pad=15)
        ax1.legend(facecolor='#1b1e2e', edgecolor='#333333', labelcolor='white', loc='upper left', ncol=2)
        ax1.grid(True, color='#2a2e39', alpha=0.3, linestyle=':')
        ax1.set_xlim(period - 10, n + 5)
        ax1.tick_params(colors='white')

        # Subplot 2: RSI
        ax2.plot(indices, rsi, color='#9b59b6', linewidth=2, label='RSI')
        ax2.plot(indices, rsi_ema, color="#368921", linewidth=1.5, label='RSI EMA')
        ax2.axhline(70, color='red', linestyle='--', alpha=0.3)
        ax2.axhline(50, color='yellow', linestyle='--', alpha=0.3)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.3)
        ax2.set_facecolor('#131722')
        ax2.set_ylim(0, 100)
        ax2.tick_params(colors='white')
        ax2.grid(True, color='#2a2e39', alpha=0.2)

        plt.tight_layout()
        plt.show()

    @staticmethod
    async def get_historical_ohlcv(pair: PairConfig, timeframe: TimeframeEnum, limit: int = 50) -> OhlcvWrapper:

        # Configura sua exchange Hyperliquid
        exchange = hyperliquid({
            "enableRateLimit": True,
            "testnet": False,
        })  # type: ignore

        try:
            # Busca candles OHLCV históricos (timestamp, open, high, low, close, volume)

            exchange.fetch_spot_markets = safe_fetch_spot_markets.__get__(exchange, exchange.__class__)
            await exchange.load_markets()

            ohlcv = await exchange.fetch_ohlcv(pair.symbol, timeframe, limit=limit)
            return OhlcvWrapper(ohlcv)
        finally:
            await exchange.close()


async def safe_fetch_spot_markets(self, params={}):
    """Versão corrigida para evitar o erro NoneType + str na Testnet"""
    try:
        request = {'type': 'spotMetaAndAssetCtxs'}
        response = await self.publicPostInfo(self.extend(request, params))

        # Se a resposta for inválida, retornamos lista vazia
        if not response or len(response) < 1:
            return []

        universe = self.safe_list(response[0], 'universe', [])
        tokens = self.safe_list(response[0], 'tokens', [])
        markets = []

        for i in range(len(universe)):
            market_data = universe[i]
            name = self.safe_string(market_data, 'name')

            # O FIX: Se o nome for None ou não tiver '/', ignoramos
            if not name or '/' not in name:
                continue

            # Deixamos o CCXT processar o resto se o nome for válido
            # Mas para este teste, podemos simplesmente ignorar Spot
            pass

        return []  # Retornamos vazio para o Spot não atrapalhar os Swaps
    except Exception:
        return []  # Se falhar qualquer coisa, não crasha o bot


# Execução principal
async def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    pair = get_pair_by_symbol("SOL/USDC:USDC")

    if pair:
        ohlcv = await PlotTrades.get_historical_ohlcv(pair, TimeframeEnum.M15, 2000)
        ohlcvHigher = await PlotTrades.get_historical_ohlcv(pair, TimeframeEnum.H1, 2000)

        #PlotTrades.plot_supertrend_with_signals(ohlcv, pair.symbol)
        #PlotTrades.plot_smart_money_flow_cloud(ohlcv, pair.symbol)
        #PlotTrades.plot_custom_supertrend(ohlcv, pair.symbol)
        #PlotTrades.plot_volumatic_vidya(ohlcv, pair.symbol)
        #PlotTrades.plot_smart_money_breakout(ohlcv, pair.symbol)
        # PlotTrades.plot_sha_macd_raw_signals(ohlcv, pair.symbol)
        # PlotTrades.plot_double_rsi_signals(ohlcv, pair.symbol)
        PlotTrades.plot_double_bb_rsi_logic(ohlcv, pair.symbol)
        #PlotTrades.plot_soheil_pko_strategy(ohlcv, pair.symbol)
        # PlotTrades.plot_market_structure_rsi(ohlcv, pair.symbol)

        #PlotTrades.plot_high_volume_pivots_bigbeluga(ohlcv, pair.symbol)

        #PlotTrades.plot_test(ohlcv, ohlcvHigher, pair.symbol)


if __name__ == "__main__":
    asyncio.run(main())
