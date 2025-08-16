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
from commons.utils.ai_supertrend.ai_super_trend_utils import AISuperTrendUtils
from commons.utils.config_loader import PairConfig, get_pair_by_symbol
from commons.utils.ohlcv_wrapper import OhlcvWrapper


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
        supertrend, trend, upperband, lowerband, supertrend_smooth, trend_signal = AISuperTrendUtils(ohlcv).get_supertrend()

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
        supertrend, trend, upperband, lowerband, supertrend_smooth, trend_signal = AISuperTrendUtils(ohlcv).get_supertrend()

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

            if trend_signal[i] == Signal.BUY:
                ax.scatter(dates_num, (0.985*lows[i]), color='green', s=100, zorder=5)
                ax.vlines(dates_num, ymin=lows[i], ymax=(0.985*lows[i]), linestyles='dashed', color='green', linewidth=1.2, zorder=5)
            
                ax.text(
                        dates_num,
                        (0.985*lows[i]),    # ligeiramente abaixo do círculo
                        f"Index {i}",
                        fontsize=9,
                        ha='center',
                        va='top',           # alinhado pelo topo do texto
                        color=color,
                        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                        zorder=6
                    )
            elif trend_signal[i] == Signal.SELL:
                ax.scatter(dates_num, (1.015*highs[i]), color='red', s=100, zorder=5)
                ax.vlines(dates_num, ymin=highs[i], ymax=(1.015*highs[i]), linestyles='dashed', color='red', linewidth=1.2, zorder=5)

                ax.text(
                        dates_num,
                        (1.015*highs[i]),    # ligeiramente abaixo do círculo
                        f"Index {i}",
                        fontsize=9,
                        ha='center',
                        va='top',           # alinhado pelo topo do texto
                        color=color,
                        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'),
                        zorder=6
                    )

        # SuperTrend Up / Down
        st_up = np.where(trend == 1, supertrend, np.nan)
        st_down = np.where(trend == -1, supertrend, np.nan)
        ax.plot(dates, st_up, color='blue', label='SuperTrend Up', linewidth=2)
        ax.plot(dates, st_down, color='orange', label='SuperTrend Down', linewidth=2)

        #print("st_up",st_up)
        #print("st_down",st_down)
        #print("supertrend_smooth",supertrend_smooth)

        # UpperBand / LowerBand
        ax.plot(dates, upperband, color='purple', linestyle='--', label='Upper Band', alpha=0.7)
        ax.plot(dates, lowerband, color='brown', linestyle='--', label='Lower Band', alpha=0.7)
        
        # SuperTrend Smooth
        #ax.plot(dates, supertrend_smooth, color='cyan', linewidth=2, label='SuperTrend Smooth')

        # Pontos de entrada da tendência

        """
        for i in range(1, len(trend)):
            if trend[i] != trend[i-1]:  # mudança de tendência
                dates_num = mdates.date2num(dates[i])
                if trend[i] == 1:  # entrada de compra
                    ax.scatter(dates_num, lows[i] - (0.001*closes[i]), color='green', s=100, zorder=5)
                elif trend[i] == -1:  # entrada de venda
                    ax.scatter(dates_num, highs[i] + (0.001*closes[i]), color='red', s=100, zorder=5)
        """
        #print("TREND", supertrend, trend)


        """
        # Marca início real de tendência baseado no SuperTrend
        for i in range(1, len(supertrend)):
            if trend[i] == 1 and trend[i-1] == -1:
                dates_num = mdates.date2num(dates[i])
                # Mudou de baixa para alta (SuperTrend desceu para baixo do preço)
                ax.scatter(dates_num, lows[i] - (0.002 * closes[i]),
                        color='green', s=120, edgecolors='black', zorder=5, label="Início Tendência Alta")
            elif trend[i] == -1 and trend[i-1] == 1:
                # Mudou de alta para baixa (SuperTrend subiu para cima do preço)
                ax.scatter(dates_num, highs[i] + (0.002 * closes[i]),
                        color='red', s=120, edgecolors='black', zorder=5, label="Início Tendência Baixa")
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
                "testnet": True,
            })

        try:
            # Busca candles OHLCV históricos (timestamp, open, high, low, close, volume)
            
            ohlcv =  await exchange.fetch_ohlcv(pair.symbol, timeframe, limit=limit)
            return OhlcvWrapper(ohlcv)
        finally:
            await exchange.close()

# Execução principal
async def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    pair = get_pair_by_symbol("SOL/USDC:USDC")

    if pair:

        ohlcv = await PlotTrades.get_historical_ohlcv(pair, TimeframeEnum.M15, 250)

        PlotTrades.plot_supertrend_with_signals(ohlcv, pair.symbol)

if __name__ == "__main__":
    asyncio.run(main())