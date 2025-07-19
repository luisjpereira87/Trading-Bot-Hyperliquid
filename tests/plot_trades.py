from datetime import datetime

import matplotlib.dates as mdates
from matplotlib import pyplot as plt

from commons.enums.signal_enum import Signal


class PlotTrades:
    
    @staticmethod
    def build_trade(ax, dates, trades):
        # Plotar entradas e saídas
        
        for trade in trades:
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
                        f"Buy ({score:.2f})",
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
                        f"Sell ({score:.2f})",
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