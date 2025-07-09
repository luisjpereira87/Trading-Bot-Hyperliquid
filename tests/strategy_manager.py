import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

from commons.enums.signal_enum import Signal
from strategies.ai_supertrend import AISuperTrend
from strategies.ml_strategy import MLModelType, MLStrategy
from tests.strategy_evaluator import StrategyEvaluator

# Assumindo SignalResult tem atributos signal, sl, tp
# from strategies.signal_result import SignalResult 



class StrategyManager:
    def __init__(self, exchange, symbol, timeframe, initial_capital=1000.0, leverage=10):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe

        self.strategies = {
            'ml-random-forest': MLStrategy(exchange, symbol, timeframe),
            'ml-random-forest-false': MLStrategy(exchange, symbol, timeframe),
            'ml-xgboost': MLStrategy(exchange, symbol, timeframe, model_type=MLModelType.XGBOOST),
            'ml-xgboost-false': MLStrategy(exchange, symbol, timeframe, model_type=MLModelType.XGBOOST),
            #'ml-mlp': MLStrategy(exchange, symbol, timeframe, model_type=MLModelType.MLP)
            #'ml-aggresive': MLStrategy(exchange, symbol, timeframe, aggressive_mode=True)
        }

        self.evaluator = StrategyEvaluator()
        for name in self.strategies:
            self.evaluator.register_strategy(name)

        self.capital = initial_capital
        self.leverage = leverage
        self.used_margin = 0.0

        self.position_sizes = {name: 0.0 for name in self.strategies}
        self.last_position: Dict[str, Any] = {name: None for name in self.strategies}

        self.orders = {name: [] for name in self.strategies}
        self.signal_counts = {name: {'buy': 0, 'sell': 0, 'hold': 0} for name in self.strategies}
        self.order_counts = {name: {'opened': 0, 'closed': 0} for name in self.strategies}

        # Armazenar o último SignalResult completo
        self.last_signals: Dict[str, Any] = {name: None for name in self.strategies}

        # NOVO: controle para ativar/desativar exit logic por estratégia
        self.exit_logic_enabled = {name: True for name in self.strategies}  # padrão: ativado para todas

        self.exit_logic_enabled['ml-xgboost-false'] = False
        self.exit_logic_enabled['ml-random-forest-false'] = False

    def normalize_signal(self, signal):
        if signal not in [Signal.BUY, Signal.SELL, Signal.HOLD]:
            logging.warning(f"Signal inválido recebido: {signal}. Usando 'hold' como fallback.")
            return 'hold'
        return signal

    def calculate_profit(self, side, entry_price, current_price):
        if side == Signal.BUY:
            return current_price - entry_price
        elif side == Signal.SELL:
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
            msg += f" posição {side}"
        if profit is not None:
            msg += f" com lucro {profit:.4f}"
        if price is not None:
            msg += f" a preço {price:.4f}"
        logging.info(msg)

    def open_order(self, strategy_name, side, price, sl=None, tp=None):
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
            'status': 'open',
            'sl': sl,   # stop loss
            'tp': tp    # take profit
        }
        self.orders[strategy_name].append(order)

        self.used_margin += price * quantity / self.leverage
        self.position_sizes[strategy_name] += quantity
        self.order_counts[strategy_name]['opened'] += 1

        logging.debug(f"[{strategy_name}] Ordem aberta: {order}")
        return order

    def close_order(self, strategy_name, order, close_price, reason=''):
        if order is None or order['status'] != 'open':
            return
        order['close_time'] = datetime.utcnow()
        order['profit'] = self.calculate_profit(order['side'], order['entry_price'], close_price) * order['quantity']
        order['status'] = 'closed'

        self.capital += order['profit']
        self.used_margin -= order['entry_price'] * order['quantity'] / self.leverage
        self.position_sizes[strategy_name] -= order['quantity']

        self.order_counts[strategy_name]['closed'] += 1

        logging.info(f"[{strategy_name}] Fechando ordem com lucro {order['profit']:.4f} por {reason}. Capital agora: {self.capital:.2f}")
        self.evaluator.record_trade(strategy_name, order['profit'], timestamp=order['close_time'])

        logging.debug(f"[{strategy_name}] Ordem fechada: {order}")

    def check_liquidation(self):
        # Se capital <= 0, fecha todas posições abertas forçando liquidação
        if self.capital <= 0:
            logging.warning("Capital <= 0! Forçando fechamento de todas posições (liquidação)")
            for strat in self.strategies.keys():
                last_pos = self.last_position[strat]
                if last_pos and last_pos['status'] == 'open':
                    self.close_order(strat, last_pos, last_pos['entry_price'], reason='liquidação por capital zero')
                    self.last_position[strat] = None

    """
    def update_profit(self, strategy_name, signal_result, current_price):
        # signal_result é o objeto SignalResult com .signal, .sl e .tp
        signal = getattr(signal_result, 'signal', Signal.HOLD)
        sl = getattr(signal_result, 'sl', None)
        tp = getattr(signal_result, 'tp', None)

        if signal.value in self.signal_counts[strategy_name]:
            self.signal_counts[strategy_name][signal.value] += 1

        last_pos = self.last_position[strategy_name]

        # Primeiro, checar se a posição aberta bateu SL ou TP
        if last_pos and last_pos['status'] == 'open':
            if last_pos['sl'] is not None:
                # Checar stop loss: depende do lado da posição
                if last_pos['side'] == 'buy' and current_price <= last_pos['sl']:
                    self.close_order(strategy_name, last_pos, last_pos['sl'], reason='stop loss atingido')
                    self.log_position_change(strategy_name, "Fechando", last_pos['side'], last_pos['profit'], last_pos['sl'])
                    self.last_position[strategy_name] = None
                    last_pos = None
                elif last_pos['side'] == 'sell' and current_price >= last_pos['sl']:
                    self.close_order(strategy_name, last_pos, last_pos['sl'], reason='stop loss atingido')
                    self.log_position_change(strategy_name, "Fechando", last_pos['side'], last_pos['profit'], last_pos['sl'])
                    self.last_position[strategy_name] = None
                    last_pos = None

            if last_pos and last_pos['tp'] is not None:
                # Checar take profit
                if last_pos['side'] == 'buy' and current_price >= last_pos['tp']:
                    self.close_order(strategy_name, last_pos, last_pos['tp'], reason='take profit atingido')
                    self.log_position_change(strategy_name, "Fechando", last_pos['side'], last_pos['profit'], last_pos['tp'])
                    self.last_position[strategy_name] = None
                    last_pos = None
                elif last_pos['side'] == 'sell' and current_price <= last_pos['tp']:
                    self.close_order(strategy_name, last_pos, last_pos['tp'], reason='take profit atingido')
                    self.log_position_change(strategy_name, "Fechando", last_pos['side'], last_pos['profit'], last_pos['tp'])
                    self.last_position[strategy_name] = None
                    last_pos = None

        # Checa liquidação
        self.check_liquidation()

        # Atualiza posição baseado no signal
        if signal == Signal.HOLD:
            if last_pos is not None and last_pos['status'] == 'open':
                self.close_order(strategy_name, last_pos, current_price, reason='sinal hold')
                self.log_position_change(strategy_name, "Fechando", last_pos['side'], last_pos['profit'], current_price)
                self.last_position[strategy_name] = None
            else:
                logging.debug(f"[{strategy_name}] Nenhuma posição aberta para fechar.")

        else:
            if last_pos is not None and last_pos['status'] == 'open':
                if last_pos['side'] != signal:
                    self.close_order(strategy_name, last_pos, current_price, reason='mudança de sinal')
                    self.log_position_change(strategy_name, "Fechando", last_pos['side'], last_pos['profit'], current_price)

                    nova_ordem = self.open_order(strategy_name, signal, current_price, sl, tp)
                    if nova_ordem:
                        self.last_position[strategy_name] = nova_ordem
                        self.log_position_change(strategy_name, "Abrindo nova", signal, price=current_price)
                    else:
                        self.last_position[strategy_name] = None
                else:
                    logging.debug(f"[{strategy_name}] Mantendo posição atual {signal} aberta.")
            else:
                nova_ordem = self.open_order(strategy_name, signal, current_price, sl, tp)
                if nova_ordem:
                    self.last_position[strategy_name] = nova_ordem
                    self.log_position_change(strategy_name, "Abrindo nova", signal, price=current_price)
                else:
                    self.last_position[strategy_name] = None

        # Guarda o signal completo para uso futuro
        self.last_signals[strategy_name] = signal_result
    """

    # Novo método para alterar exit logic em runtime
    def set_exit_logic(self, strategy_name: str, enabled: bool):
        if strategy_name in self.exit_logic_enabled:
            self.exit_logic_enabled[strategy_name] = enabled
            logging.info(f"[{strategy_name}] Exit logic {'ativado' if enabled else 'desativado'}")
        else:
            logging.warning(f"Tentativa de configurar exit logic para estratégia desconhecida: {strategy_name}")


    def update_profit(self, strategy_name, signal_result, current_price):
        signal = getattr(signal_result, 'signal', Signal.HOLD)
        sl = getattr(signal_result, 'sl', None)
        tp = getattr(signal_result, 'tp', None)

        if signal.value in self.signal_counts[strategy_name]:
            self.signal_counts[strategy_name][signal.value] += 1

        last_pos = self.last_position[strategy_name]

        # --- 1. Se EXIT LOGIC estiver ativado, processa SL, TP, HOLD ---
        if self.exit_logic_enabled.get(strategy_name, True):
            if last_pos and last_pos['status'] == 'open':
                # Stop Loss
                if last_pos['sl'] is not None:
                    if last_pos['side'] == 'buy' and current_price <= last_pos['sl']:
                        self.close_order(strategy_name, last_pos, last_pos['sl'], reason='stop loss atingido')
                        self.log_position_change(strategy_name, "Fechando", last_pos['side'], last_pos['profit'], last_pos['sl'])
                        self.last_position[strategy_name] = None
                        last_pos = None
                    elif last_pos['side'] == 'sell' and current_price >= last_pos['sl']:
                        self.close_order(strategy_name, last_pos, last_pos['sl'], reason='stop loss atingido')
                        self.log_position_change(strategy_name, "Fechando", last_pos['side'], last_pos['profit'], last_pos['sl'])
                        self.last_position[strategy_name] = None
                        last_pos = None

                # Take Profit
                if last_pos and last_pos['tp'] is not None:
                    if last_pos['side'] == 'buy' and current_price >= last_pos['tp']:
                        self.close_order(strategy_name, last_pos, last_pos['tp'], reason='take profit atingido')
                        self.log_position_change(strategy_name, "Fechando", last_pos['side'], last_pos['profit'], last_pos['tp'])
                        self.last_position[strategy_name] = None
                        last_pos = None
                    elif last_pos['side'] == 'sell' and current_price <= last_pos['tp']:
                        self.close_order(strategy_name, last_pos, last_pos['tp'], reason='take profit atingido')
                        self.log_position_change(strategy_name, "Fechando", last_pos['side'], last_pos['profit'], last_pos['tp'])
                        self.last_position[strategy_name] = None
                        last_pos = None

            self.check_liquidation()

            # HOLD: fecha posição se houver
            if signal == Signal.HOLD:
                if last_pos and last_pos['status'] == 'open':
                    self.close_order(strategy_name, last_pos, current_price, reason='sinal hold')
                    self.log_position_change(strategy_name, "Fechando", last_pos['side'], last_pos['profit'], current_price)
                    self.last_position[strategy_name] = None

        # --- 2. Sempre processa reversão ou nova entrada, independentemente do exit_logic ---
        if signal != Signal.HOLD:
            if last_pos and last_pos['status'] == 'open':
                # Reversão de posição
                if last_pos['side'] != signal:
                    self.close_order(strategy_name, last_pos, current_price, reason='mudança de sinal')
                    self.log_position_change(strategy_name, "Fechando", last_pos['side'], last_pos['profit'], current_price)

                    nova_ordem = self.open_order(strategy_name, signal, current_price, sl, tp)
                    if nova_ordem:
                        self.last_position[strategy_name] = nova_ordem
                        self.log_position_change(strategy_name, "Abrindo nova", signal, price=current_price)
                    else:
                        self.last_position[strategy_name] = None
                else:
                    logging.debug(f"[{strategy_name}] Mantendo posição atual {signal} aberta.")
            else:
                # Nova entrada
                nova_ordem = self.open_order(strategy_name, signal, current_price, sl, tp)
                if nova_ordem:
                    self.last_position[strategy_name] = nova_ordem
                    self.log_position_change(strategy_name, "Abrindo nova", signal, price=current_price)
                else:
                    self.last_position[strategy_name] = None

        # Guarda o signal completo
        self.last_signals[strategy_name] = signal_result

    async def run_strategies(self):
        try:
            coros = []
            for strategy in self.strategies.values():
                coros.append(strategy.get_signal())
            results = await asyncio.gather(*coros)

            signals = {}
            for name, result in zip(self.strategies.keys(), results):
                # extrai o sinal string para normalizar e atualizar contadores
                if hasattr(result, 'signal'):
                    signal_str = result.signal
                else:
                    signal_str = str(result)

                signals[name] = self.normalize_signal(signal_str)

            ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe)
            if not ohlcv:
                logging.warning("Nenhum dado OHLCV retornado pelo exchange")
                return {name: 'hold' for name in self.strategies}

            current_price = ohlcv[-1][4]

            for name, result in zip(self.strategies.keys(), results):
                self.update_profit(name, result, current_price)

            return signals

        except Exception as e:
            logging.error(f"Erro ao executar run_strategies: {e}", exc_info=True)
            return {name: 'hold' for name in self.strategies}

    def report(self):
        self.evaluator.report_summary()

        logging.info("=== Contadores de sinais ===")
        for strat, counts in self.signal_counts.items():
            logging.info(f"Strategy {strat}: {counts}")

        logging.info("=== Contadores de ordens ===")
        for strat, counts in self.order_counts.items():
            logging.info(f"Strategy {strat}: {counts}")



