import asyncio
import logging
from datetime import datetime

from strategies.ai_supertrend import AISuperTrend
from strategies.ml_strategy import MLStrategy
from tests.strategy_evaluator import StrategyEvaluator


class StrategyManager:
    def __init__(self, exchange, symbol, timeframe, initial_capital=1000.0, leverage=10):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe

        # Dicionário de estratégias, pode adicionar/remover facilmente
        self.strategies = {
            'ml': MLStrategy(exchange, symbol, timeframe),
            **self.create_supertrend_variations(self.exchange, self.symbol, self.timeframe)
            #'supertrend_default': AISuperTrend(exchange, symbol, timeframe),
            # Exemplos de várias AISuperTrend com parâmetros diferentes:
            # 'supertrend_param1': AISuperTrend(exchange, symbol, timeframe, param1=123),
            # 'supertrend_param2': AISuperTrend(exchange, symbol, timeframe, param1=456),
        }

        self.evaluator = StrategyEvaluator()
        for name in self.strategies:
            self.evaluator.register_strategy(name)

        self.capital = initial_capital
        self.leverage = leverage
        self.used_margin = 0.0

        self.position_sizes = {name: 0.0 for name in self.strategies}
        self.last_position = {name: None for name in self.strategies}
        self.orders = {name: [] for name in self.strategies}

        self.signal_counts = {name: {'buy': 0, 'sell': 0, 'hold': 0} for name in self.strategies}
        self.order_counts = {name: {'opened': 0, 'closed': 0} for name in self.strategies}

    def create_supertrend_variations(self, exchange, symbol, timeframe):
        strategies = {}

        modes = ['conservative', 'aggressive']
        multipliers = [1.0, 1.2, 1.5]
        adx_thresholds = [15, 20, 25]
        rsi_buy_thresholds = [30, 40, 50]
        rsi_sell_thresholds = [50, 60, 70]

        for mode in modes:
            for multiplier in multipliers:
                for adx_th in adx_thresholds:
                    for rsi_buy in rsi_buy_thresholds:
                        for rsi_sell in rsi_sell_thresholds:
                            name = f"supertrend_{mode}_m{multiplier}_adx{adx_th}_rsiBuy{rsi_buy}_rsiSell{rsi_sell}"
                            strategies[name] = AISuperTrend(
                                exchange,
                                symbol,
                                timeframe,
                                mode=mode,
                                multiplier=multiplier,
                                adx_threshold=adx_th,
                                rsi_buy_threshold=rsi_buy,
                                rsi_sell_threshold=rsi_sell
                            )

        return strategies

    def normalize_signal(self, signal):
        if signal not in ['buy', 'sell', 'hold']:
            logging.warning(f"Signal inválido recebido: {signal}. Usando 'hold' como fallback.")
            return 'hold'
        return signal

    def calculate_profit(self, side, entry_price, current_price):
        if side == 'buy':
            return current_price - entry_price
        elif side == 'sell':
            return entry_price - current_price
        else:
            return 0.0

    def calculate_quantity(self, price):
        max_position_value = self.capital * self.leverage - self.used_margin
        quantity = max_position_value / price
        return quantity if quantity > 0 else 0

    def log_position_change(self, strategy_name, action, side=None, profit=None, price=None):
        msg = f"[{strategy_name}] {action}"
        if side:
            msg += f" posição {side.upper()}"
        if profit is not None:
            msg += f" com lucro {profit:.4f}"
        if price is not None:
            msg += f" a preço {price:.4f}"
        logging.info(msg)

    def open_order(self, strategy_name, side, price):
        quantity = self.calculate_quantity(price)
        if quantity <= 0:
            logging.warning(f"[{strategy_name}] Capital insuficiente para abrir nova ordem")
            return None

        order = {
            'side': side,
            'entry_price': price,
            'quantity': quantity,
            'open_time': datetime.utcnow(),
            'close_time': None,
            'profit': None,
            'status': 'open'
        }
        self.orders[strategy_name].append(order)

        self.used_margin += price * quantity / self.leverage
        self.position_sizes[strategy_name] += quantity
        self.order_counts[strategy_name]['opened'] += 1

        logging.debug(f"[{strategy_name}] Ordem aberta: {order}")
        return order

    def close_order(self, strategy_name, order, close_price):
        if order is None or order['status'] != 'open':
            return
        order['close_time'] = datetime.utcnow()
        order['profit'] = self.calculate_profit(order['side'], order['entry_price'], close_price) * order['quantity']
        order['status'] = 'closed'

        self.capital += order['profit']
        self.used_margin -= order['entry_price'] * order['quantity'] / self.leverage
        self.position_sizes[strategy_name] -= order['quantity']

        self.order_counts[strategy_name]['closed'] += 1

        logging.info(f"[{strategy_name}] Fechando ordem com lucro {order['profit']:.4f}. Capital agora: {self.capital:.2f}")
        self.evaluator.record_trade(strategy_name, order['profit'], timestamp=order['close_time'])

        logging.debug(f"[{strategy_name}] Ordem fechada: {order}")

    def update_profit(self, strategy_name, signal, current_price):
        if signal in self.signal_counts[strategy_name]:
            self.signal_counts[strategy_name][signal] += 1

        last_pos = self.last_position[strategy_name]
        logging.debug(f"[{strategy_name}] last_pos={last_pos}, signal={signal}, current_price={current_price}")

        if signal == 'hold':
            if last_pos is not None and last_pos['status'] == 'open':
                self.close_order(strategy_name, last_pos, current_price)
                self.log_position_change(strategy_name, "Fechando", last_pos['side'], last_pos['profit'], current_price)
                self.last_position[strategy_name] = None
            else:
                logging.debug(f"[{strategy_name}] Nenhuma posição aberta para fechar.")

        else:
            if last_pos is not None and last_pos['status'] == 'open':
                if last_pos['side'] != signal:
                    self.close_order(strategy_name, last_pos, current_price)
                    self.log_position_change(strategy_name, "Fechando", last_pos['side'], last_pos['profit'], current_price)

                    nova_ordem = self.open_order(strategy_name, signal, current_price)
                    if nova_ordem:
                        self.last_position[strategy_name] = nova_ordem
                        self.log_position_change(strategy_name, "Abrindo nova", signal, price=current_price)
                    else:
                        self.last_position[strategy_name] = None
                else:
                    logging.debug(f"[{strategy_name}] Mantendo posição atual {signal.upper()} aberta.")
            else:
                nova_ordem = self.open_order(strategy_name, signal, current_price)
                if nova_ordem:
                    self.last_position[strategy_name] = nova_ordem
                    self.log_position_change(strategy_name, "Abrindo nova", signal, price=current_price)
                else:
                    self.last_position[strategy_name] = None

    async def run_strategies(self):
        try:
            # Executa todos os sinais em paralelo
            coros = []
            for strategy in self.strategies.values():
                if hasattr(strategy, 'run'):
                    coros.append(strategy.run())
                else:
                    coros.append(strategy.get_signal())
            results = await asyncio.gather(*coros)

            signals = {name: self.normalize_signal(signal) for name, signal in zip(self.strategies.keys(), results)}

            ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe)
            if not ohlcv:
                logging.warning("Nenhum dado OHLCV retornado pelo exchange")
                return {name: 'hold' for name in self.strategies}

            current_price = ohlcv[-1][4]

            for name, signal in signals.items():
                self.update_profit(name, signal, current_price)

            return signals

        except Exception as e:
            logging.error(f"Erro ao executar run_strategies: {e}", exc_info=True)
            return {name: 'hold' for name in self.strategies}

    def report(self):
        #self.evaluator.report()

        self.evaluator.report_summary()

        logging.info("=== Contadores de sinais ===")
        for strat, counts in self.signal_counts.items():
            logging.info(f"Strategy {strat}: {counts}")

        logging.info("=== Contadores de ordens ===")
        for strat, counts in self.order_counts.items():
            logging.info(f"Strategy {strat}: {counts}")


