import asyncio
import logging
import math
import time
from datetime import datetime
from decimal import Decimal

from eth_account import Account
from eth_account.signers.local import LocalAccount
from nado_protocol.client import (NadoClient, NadoClientContext,
                                  NadoClientMode, create_nado_client)
from nado_protocol.engine_client.types import (CancelOrdersParams,
                                               MarketOrderParams, OrderParams,
                                               PlaceMarketOrderParams,
                                               PlaceOrderParams,
                                               SubaccountInfoData)
from nado_protocol.indexer_client.types.models import \
    IndexerCandlesticksGranularity
from nado_protocol.indexer_client.types.query import (
    IndexerCandlesticksParams, IndexerSubaccountHistoricalOrdersParams,
    IndexerSubaccountsParams)
from nado_protocol.trigger_client.types.execute import (
    CancelProductOrdersParams, CancelProductTriggerOrdersParams,
    CancelTriggerOrdersParams, PlaceTriggerOrderParams)
from nado_protocol.trigger_client.types.models import (LastPriceAbove,
                                                       LastPriceBelow,
                                                       PriceTrigger,
                                                       PriceTriggerData,
                                                       TriggerCriteria)
from nado_protocol.utils.expiration import OrderType
from nado_protocol.utils.nonce import gen_order_nonce
from nado_protocol.utils.order import OrderAppendixTriggerType, build_appendix
from nado_protocol.utils.subaccount import Subaccount, SubaccountParams

from commons.enums.signal_enum import Signal
from commons.enums.timeframe_enum import TimeframeEnum
from commons.helpers.trading_helpers import TradingHelpers
from commons.helpers.trailing_stop_helpers import TrailingStopHelpers
from commons.models.ohlcv_format_dclass import OhlcvFormat
from commons.models.open_position_dclass import OpenPosition
from commons.models.opened_order_dclass import OpenedOrder
from commons.utils.config_loader import (PairConfig,
                                         get_pair_by_configs_symbol,
                                         get_pair_by_symbol, load_pair_configs)
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from trading_bot.exchange_base import ExchangeBase

# Teus enums e classes permanecem iguais
# ... 

class NadoExchangeClient(ExchangeBase):
    def __init__(self, private_key, rpc_url, wallet_address, pairs: list[PairConfig], subaccount_id=0):
        # 1. Criar o signer
        self.signer: LocalAccount = Account.from_key(private_key)
        self.wallet_address = self.signer.address
        
        # 2. Criar o cliente de forma simples
        # O helper 'create_nado_client' já instancia internamente o Indexer, Engine, etc.
        # Substitui "devnet" por "mainnet" (ou o nome da rede Ink) quando fores para produção
        
        self.client = create_nado_client(NadoClientMode.TESTNET, self.signer)

        self.wallet_address = wallet_address
        self.subaccount_id = subaccount_id
        self.helpers = TradingHelpers()
        self._market_map = {} # Cache para mapear "BTC-PERP" -> ID

        self.indexer = self.client.context.indexer_client
        self.engine = self.client.context.engine_client
        self.X18_SCALE = 10**18  # ADICIONA ESTA LINHA AQUI

        self._pairs_cache = pairs
        self._market_map = {}

        self.active_trailing_levels = {}

    
    async def _get_market_id(self, symbol: str) -> int | None:
        try:
            # 1. Busca no cache de mapeamento (já carregado no init)
            pair = get_pair_by_configs_symbol(self._pairs_cache, symbol) if self._pairs_cache else None
            if not pair:
                return None
            
            symbol_nado = pair.symbol_nado

            # 2. Sincronização Única com a Engine
            if not self._market_map:
                logging.info("🔄 Sincronizando IDs e Incrementos com a Engine Nado...")
                symbols_data = self.client.context.engine_client.get_symbols()
                
                self._market_map = {}
                X18 = 10**18

                for ticker, data in symbols_data.symbols.items():
                    # Extração direta dos campos que vimos no teu log
                    # Note: size_increment às vezes não tem o sufixo _x18 no nome mas o valor é X18
                    s_inc = int(data.size_increment)
                    p_inc = int(data.price_increment_x18)
                    m_notional = int(data.min_size)

                    self._market_map[ticker] = {
                        "id": int(data.product_id),
                        "size_step": s_inc / X18,
                        "price_step": p_inc / X18,
                        "min_notional": m_notional / X18
                    }
                
                logging.info(f"✅ Market Map: {len(self._market_map)} ativos carregados.")

            # 3. Resolução instantânea por dicionário
            data = self._market_map.get(symbol_nado)
            return data["id"] if data else None

        except Exception as e:
            logging.error(f"❌ Erro ao resolver Market ID para {symbol}: {e}")
            return None
        
    async def get_market_meta(self, symbol: str):
        """Retorna o dicionário de metadados para o símbolo."""
        # Garante que o mapa está carregado
        if not self._market_map:
            await self._get_market_id(symbol)
        
        pair = get_pair_by_configs_symbol(self._pairs_cache, symbol)
        return self._market_map.get(pair.symbol_nado) if pair else None

    async def get_size_increment(self, symbol: str) -> float:
        meta = await self.get_market_meta(symbol)
        return meta["size_step"] if meta else 0.01

    async def get_price_increment(self, symbol: str) -> float:
        meta = await self.get_market_meta(symbol)
        return meta["price_step"] if meta else 0.01
    

    async def fetch_ohlcv(self, symbol: str, timeframe: TimeframeEnum, limit: int = 14, is_higher: bool = False) -> OhlcvFormat:
        try:
            product_id = await self._get_market_id(symbol)
            if product_id is None:
                raise ValueError(f"ID não encontrado para {symbol}")

            # 1. Mapeamento de Granularidade (Certifica-te que estes nomes batem com o teu Enum)
            granularity_map = {
                "1m": IndexerCandlesticksGranularity.ONE_MINUTE,
                "5m": IndexerCandlesticksGranularity.FIVE_MINUTES,
                "15m": IndexerCandlesticksGranularity.FIFTEEN_MINUTES,
                "1h": IndexerCandlesticksGranularity.ONE_HOUR,
                "4h": IndexerCandlesticksGranularity.FOUR_HOURS, # Adicionado comum
                "1d": IndexerCandlesticksGranularity.ONE_DAY,
            }
            
            nado_granularity = granularity_map.get(timeframe.value)
            if not nado_granularity:
                logging.warning(f"Timeframe {timeframe.value} não mapeado, a usar 15m como default")
                nado_granularity = IndexerCandlesticksGranularity.FIFTEEN_MINUTES

            params = IndexerCandlesticksParams(
                product_id=product_id,
                granularity=nado_granularity,
                limit=limit,
                submission_idx=None,
                max_time=None
            )

            # 2. Chamada ao Indexer
            response = self.client.context.indexer_client.get_candlesticks(params)

            X18_SCALE = 10**18
            
            # 3. Formatação e Limpeza
            raw_data = []
            for c in response.candlesticks:
                if c.timestamp is not None and c.close_x18 is not None:
                    raw_data.append([
                        int(c.timestamp) * 1000, # Se a tua estratégia usa ms, multiplica por 1000 aqui
                        float(c.open_x18) / X18_SCALE,
                        float(c.high_x18) / X18_SCALE, 
                        float(c.low_x18) / X18_SCALE, 
                        float(c.close_x18) / X18_SCALE, 
                        float(c.volume) / X18_SCALE
                    ])

            # 4. ORDENAÇÃO: Garante ordem cronológica (Antigo -> Recente)
            # Isso resolve o problema de indicadores calcularem sobre velas erradas
            raw_data.sort(key=lambda x: x[0])

            if not raw_data:
                logging.error(f"⚠️ Nenhum candle retornado para {symbol}")
                return OhlcvFormat(OhlcvWrapper([]), OhlcvWrapper([]))

            logging.info(f"📈 {symbol}: {len(raw_data)} velas obtidas. Última: {datetime.fromtimestamp(raw_data[-1][0] / 1000)}")

            return OhlcvFormat(OhlcvWrapper(raw_data), OhlcvWrapper([]))

        except Exception as e:
            logging.error(f"❌ Erro fetch_ohlcv Nado para {symbol}: {e}")
            raise

    def _get_subaccount_info(self) -> SubaccountInfoData | None:
        try:
            # 1. Pegamos a info da subconta via Engine (o log que mandaste)
            # Assumindo que 'sub' é o objeto que obtiveste via get_subaccount_info
            params = IndexerSubaccountsParams(address=self.wallet_address, limit=None, start=0)
            subaccounts_list = self.client.context.indexer_client.get_subaccounts(params)
            
            if not subaccounts_list or not subaccounts_list.subaccounts:
                return None

            # Usamos a primeira subconta encontrada ou filtramos pelo ID
            sub_data = subaccounts_list.subaccounts[0]

            # Chamada à Engine para ter os dados em tempo real (spot_balances)
            return self.engine.get_subaccount_info(subaccount=sub_data.subaccount)

        except Exception as e:
            logging.error(f"Erro ao obter info de conta: {e}")
            return None

    async def print_open_orders(self, symbol: str):
        try:

            info = self._get_subaccount_info()

            if info is None:
                logging.info(f"📊 Não foi possível obter os dados")
                return 
            
            product_id = await self._get_market_id(symbol)
            # 1. Localizar o balanço do Perpétuo específico
            # O perp_balances contém o estado da posição (amount, v_quote_balance, etc)
            perp_pos = next((p for p in info.perp_balances if p.product_id == product_id), None)

            if perp_pos and float(perp_pos.balance.amount) != 0:
                # O amount na Nado representa o tamanho da posição (positivo para Long, negativo para Short)
                size = float(perp_pos.balance.amount) / 10**18
                side = "BUY" if size > 0 else "SELL"
                
                # Nota: O PnL não vem direto aqui, geralmente é calculado via v_quote_balance 
                # ou lido de info.perp_products se precisares de dados do mercado
                logging.info(f"📊 [ST] {symbol} | Lado: {side} | Size: {abs(size)}")
            else:
                logging.info(f"📊 [ST] {symbol} | Sem posição ativa.")

            # 2. Balanço USDC (Spot)
            # Normalmente o USDC é o product_id 0
            usdc_balance = next((s for s in info.spot_balances if s.product_id == 0), None)
            if usdc_balance:
                available_usdc = float(usdc_balance.balance.amount) / 10**18
                logging.info(f"💰 [ST] Saldo USDC: {available_usdc:.2f}")

        except Exception as e:
            logging.error(f"❌ Erro ao processar get_subaccount_info: {e}")

    async def print_balance(self):
        try:
            balance = await self.get_available_balance()
            logging.info(f"💰 Saldo total: {balance}")
        except Exception as e:
            logging.error(f"Erro ao buscar saldo: {e}")

    async def get_available_balance(self) -> float:
        try:
            # Chamada à Engine para ter os dados em tempo real (spot_balances)
            info =  self._get_subaccount_info()
            
            if not info or not hasattr(info, 'spot_balances'):
                return 0.0

            # 2. Procurar o Product ID 0 (USDC) dentro de spot_balances
            for spot in info.spot_balances:
                if spot.product_id == 0:
                    # O valor vem como string no log, convertemos para int e depois float
                    raw_amount = int(spot.balance.amount)
                    balance = raw_amount / self.X18_SCALE
                    logging.info(f"💰 Saldo USDC detetado: {balance:.2f}")
                    return balance

            return 0.0

        except Exception as e:
            logging.error(f"Erro ao mapear saldo do log: {e}")
            return 0.0
        
    async def get_entry_price(self, symbol: str) -> float|None:
        try:
            # 1. Primeiro, precisamos do product_id (ex: BTC-PERP = 2)
            product_id = await self._get_market_id(symbol)

            if product_id is None:
                return None
            
            # 2. Chamar o método do Indexer
            # Na Nado, o retorno costuma ser um objeto MarketPrice
            market_price_data = self.client.market.get_latest_market_price(
                product_id=product_id
            )
            
            # 3. Converter de X18 para float
            # A Nado devolve quase tudo multiplicado por 10^18
            price = float(market_price_data.ask_x18) / 10**18
            
            logging.info(f"📊 Preço Indexer para {symbol}: {price}")
            return price

        except Exception as e:
            logging.error(f"❌ Erro ao obter get_latest_market_price: {e}")
        return None

    async def get_open_position(self, symbol: str) -> (OpenPosition | None):
        try:
            product_id = await self._get_market_id(symbol)
            if product_id is None:
                return None
            positions = self._get_subaccount_info()
            if not positions: return None

            for p_balance in positions.perp_balances:
                if p_balance.product_id == product_id:
                    amount_raw = int(p_balance.balance.amount)
                    
                    # SÓ CONTINUA SE HOUVER POSIÇÃO ABERTA NA ENGINE
                    if abs(amount_raw) > 1e14: # Mais de 0.0001 para ignorar poeira
                        X18_SCALE = 10**18
                        size = abs(float(amount_raw)) / X18_SCALE
                        side = 'buy' if amount_raw > 0 else 'sell'

                        # BUSCAR PREÇO NO INDEXER
                        entry_price = 0.0
                        try:
                            hist_params = IndexerSubaccountHistoricalOrdersParams(
                                subaccounts=[positions.subaccount],
                                product_ids=[product_id],
                                limit=5,
                                submission_idx=None,
                                max_time=None,
                                trigger_types=None,
                                isolated=None
                            )
                            history = self.client.context.indexer_client.get_subaccount_historical_orders(hist_params)

                            if history and history.orders:
                                # Procuramos a ordem mais recente que combine com o nosso side
                                for order_info in history.orders:
                                    order_amount = int(order_info.amount)
                                    # Se a ordem é de compra e estamos comprados (ou vice-versa)
                                    if (order_amount > 0 and side == 'buy') or (order_amount < 0 and side == 'sell'):
                                        # Cálculo do preço real de execução (Total gasto / Quantidade)
                                        # Usamos abs porque quote_filled é negativo na compra
                                        q_filled = abs(float(order_info.quote_filled))
                                        b_filled = abs(float(order_info.base_filled))
                                        if b_filled > 0:
                                            entry_price = q_filled / b_filled
                                            break
                        except Exception as e:
                            logging.error(f"Erro ao buscar histórico: {e}")

                        # FALLBACK SE O INDEXER FALHAR
                        if entry_price == 0:
                            v_quote = float(p_balance.balance.v_quote_balance)
                            entry_price = abs(-v_quote / float(amount_raw))

                        current_market_price = await self.get_entry_price(symbol)
                        if not current_market_price: return None

                        # CÁLCULO PNL
                        diff = (current_market_price - entry_price) / entry_price
                        pnl_pct = diff if side == 'buy' else -diff
                        
                        logging.info(f"📊 {symbol} | Entry: {entry_price:.2f} | Market: {current_market_price:.2f} | PNL: {pnl_pct*100:.2f}%")

                        return OpenPosition(
                            side=side, size=size, entry_price=entry_price,
                            id=f"pos_{product_id}", notional=size * current_market_price,
                            sl=None, tp=None, unrealizedPnl=pnl_pct
                        )
            
            return None # Nenhuma posição aberta para este produto

        except Exception as e:
            logging.error(f"Erro em get_open_position para {symbol}: {e}")
            return None
        
    async def calculate_entry_amount(self, symbol: str, price_ref: float, capital_amount: float) -> float:
        """
        Calcula a quantidade (size) exata para a Nado, respeitando os 
        incrementos mínimos do protocolo para evitar erros de validação.
        """
        try:
            if price_ref <= 0 or capital_amount <= 0:
                logging.warning(f"🚫 Preço ({price_ref}) ou Capital ({capital_amount}) inválidos.")
                return 0.0

            # 1. Cálculo da quantidade bruta
            raw_quantity = capital_amount / price_ref

            # 2. Busca Dinâmica do Incremento (A CORREÇÃO ESTÁ AQUI)
            # Em vez de 0.00005 fixo, pedimos à exchange qual é a regra para este símbolo
            size_increment = await self.get_size_increment(symbol)
            
            # Fallback de segurança se a API falhar (usa 0.01 para ALTs, 0.0001 para BTC)
            if size_increment == 0:
                size_increment = 0.0001 if "BTC" in symbol else 0.01

            # 3. Arredondamento "Floor" (Corta o excesso)
            # Exemplo SOL: 1.338 -> Step 0.01 -> (133.8 // 1) * 0.01 -> 1.33
            multiplier = 1 / size_increment
            refined_quantity = math.floor(raw_quantity * multiplier) / multiplier

            # 4. Limpeza de precisão
            # Calcula quantas casas decimais o step tem para arredondar corretamente

            decimals = int(abs(math.log10(size_increment))) if size_increment < 1 else 0
            final_quantity = round(refined_quantity, decimals)

            logging.info(
                f"🧮 Cálculo de Size: Bruto: {raw_quantity:.6f} | "
                f"Ajustado: {final_quantity:.6f} (Step: {size_increment})"
            )

            return final_quantity

        except Exception as e:
            logging.error(f"❌ Erro ao calcular entry_amount: {e}")
            return 0.0
    
        
    async def open_new_position(
        self, 
        symbol: str, 
        leverage: float, 
        signal: Signal, 
        capital_amount: float, # Podemos ignorar este vindo de fora
        pair: PairConfig, 
        sl: (float | None), 
        tp: (float | None)
    ) -> (OpenedOrder | None):
        try:
            # 1. Obter Saldo Real e Preço de Referência
            # Usamos o equity total para o cálculo da percentagem
            balance_total = await self.get_available_balance()
            price_ref = await self.get_entry_price(symbol)

            if balance_total <= 0 or not price_ref:
                logging.warning(f"⚠️ Saldo ({balance_total}) ou Preço ({price_ref}) inválidos para {symbol}")
                return None

            # 2. Lógica de Capital baseada no JSON (Ex: capital=0.3, leverage=10)
            # Calculamos quanto USD queremos expor (Notional)
            target_capital = balance_total * pair.capital  # 500 se capital=0.5
            notional_value = target_capital * pair.leverage # 5000 se leverage=10
            
            # 3. Converter para Size (Quantidade de ativos)
            entry_amount = await self.calculate_entry_amount(symbol, price_ref, notional_value)

            logging.info(
                f"💰 [Nado] Balance: ${balance_total:.2f} | Alocação: {pair.capital*100}% | "
                f"Alavancagem: {pair.leverage}x | Notional Alvo: ${notional_value:.2f}"
            )

            # 4. Check de Mínimo da Nado ($100)
            if entry_amount * price_ref < 105:
                logging.warning(
                    f"🚫 Ordem abaixo do mínimo ($105). Requerido: {entry_amount * price_ref:.2f}"
                )
                return None

            # 5. Executar
            return await self.place_entry_order(
                symbol=symbol,
                leverage=pair.leverage,
                entry_amount=entry_amount,
                price_ref=price_ref,
                side=signal,
                sl_price=sl,
                tp_price=tp
            )

        except Exception as e:
            logging.error(f"❌ Erro ao calcular/abrir posição: {e}")
            return None
    
    async def place_entry_order(
        self, 
        symbol: str, 
        leverage: float, 
        entry_amount: float, 
        price_ref: float, 
        side: Signal, 
        sl_price: (float | None) = None, 
        tp_price: (float | None) = None
    ) -> OpenedOrder:
        try:
            # 1. Identificação do Produto (Mapeamento symbol -> id)
            product_id = await self._get_market_id(symbol)
            if product_id is None:
                raise ValueError(f"ID não encontrado para {symbol}")

            # 2. Configuração de Escalas e Arredondamentos (Engine Requirements)
            X18_SCALE = 10**18
            step_decimal = await self.get_size_increment(symbol)
            size_increment = int(step_decimal * X18_SCALE)

            # Incremento de Tamanho (Size Step)
            # Nota: entry_amount já deve vir arredondado pelo calculate_entry_amount
            amount_raw = int(entry_amount * X18_SCALE)
            #size_increment = 50_000_000_000_000 # Específico para BTC-PERP na Nado
            amount_adjusted = (amount_raw // size_increment) * size_increment
            
            # Na Nado: Compra (BUY) é positivo, Venda (SELL) é negativo
            amount = amount_adjusted if side == Signal.BUY else -amount_adjusted

            # 3. Cálculo de Preço Limite (Slippage de 10% sobre o price_ref)
            price_step_decimal = await self.get_price_increment(symbol)
            price_increment_x18 = int(price_step_decimal * X18_SCALE)
            if side == Signal.BUY:
                raw_price_limit = int(price_ref * 1.10 * X18_SCALE)
            else:
                raw_price_limit = int(price_ref * 0.90 * X18_SCALE)
                
            price_limit = (raw_price_limit // price_increment_x18) * price_increment_x18

            # 4. Construção do Appendix (Versão 1 obrigatória para Perps)
            # Usamos FOK (Fill or Kill) para garantir execução imediata ou cancelamento
            appendix = build_appendix(
                order_type=OrderType.FOK, 
                reduce_only=False
            )

            sender_info = SubaccountParams(
                subaccount_owner=self.wallet_address,
                subaccount_name="default" 
            )

            # 5. Montagem dos Parâmetros da Ordem
            m_order = OrderParams(
                sender=sender_info,
                amount=amount,
                priceX18=price_limit,
                nonce=None,
                expiration=int(time.time() + 60),
                appendix=appendix
            )

            # 6. Envio via Wrapper de Mercado (Resolve Erro 2018 e trata Assinatura)
            params = PlaceOrderParams(
                product_id=product_id,
                order=m_order,
                signature=None,
                id=None,
                digest=None,
                spot_leverage=None # OBRIGATÓRIO False para Perps
            )

            real_size = abs(amount) / X18_SCALE
            logging.info(f"🚀 [Nado] Enviando Ordem: {side.value} {real_size} {symbol} | Ref: {price_ref}")

            # Execução síncrona no SDK (pode ser chamada dentro do wrapper async)
            order_response = self.client.market.place_order(params)
            
            # 7. Gestão do Digest (ID da Ordem)
            order_hash = getattr(order_response, 'digest', "confirmed")

            # 8. Proteções Secundárias (Take Profit e Stop Loss)
            if sl_price or tp_price:
                logging.info(f"🛡️ Configurando ordens de proteção: SL {sl_price} | TP {tp_price}")
                await self._place_protections(
                    symbol=symbol,
                    product_id=product_id,
                    size=abs(entry_amount),
                    side=side,
                    sl_price=sl_price,
                    tp_price=tp_price
                )


            return OpenedOrder(
                id=order_hash,
                clientOrderId=None,
                timestamp=None,
                datetime=None,
                symbol=symbol,
                type="market",
                side=side.value,
                price=price_ref,
                amount=real_size,
                reduceOnly=False,
                orderType="market"
            )

        except Exception as e:
            logging.error(f"❌ Erro fatal no place_entry_order (Nado): {e}")
            raise

    
    async def _place_protections(self, symbol, product_id, size, side, sl_price, tp_price):
        try:

            # 1. Converte para string e coloca tudo em minúsculas
            side_str = str(side).lower()

            # 2. Cria um booleano simples para usar no resto da função
            is_buy = 'buy' in side_str
            
            market_data = await self.get_market_meta(symbol)
            if not market_data: return

            price_step = market_data["price_step"]
            size_step = market_data["size_step"]
            
            def to_clean_x18(value: float, step: float) -> int:
                """
                1. Limpa o valor para respeitar o tick size (price_step ou size_step).
                2. Converte para a escala X18 com precisão Decimal.
                """
                # Passo 1: Limpeza matemática (Arredondamento por baixo para o step mais próximo)
                # Ex: 2100.12345 com step 0.01 -> 2100.12
                clean_val = float(round((value // step) * step, 8))
                
                # Passo 2: A lógica oficial da Nado (Decimal + str) para evitar lixo binário
                return int(Decimal(str(clean_val)) * Decimal(10**18))

            # Subaccount comum
            subaccount = SubaccountParams(subaccount_owner=self.wallet_address, subaccount_name="default")

            # 3. Preparar a Quantidade (Amount) - Negativa para Long, Positiva para Short
            raw_amount = -size if is_buy else size
            amount_x18 = to_clean_x18(raw_amount, size_step)

            # ---------------------------------------------------------
            # 1. STOP LOSS (Trigger Order)
            # ---------------------------------------------------------
            if sl_price:

                sl_trigger_x18 = to_clean_x18(sl_price, price_step)
            
                # Preço de Execução (Limit) com Slippage de 5%
                slippage_factor = 0.95 if is_buy else 1.05
                sl_limit_price = sl_price * slippage_factor
                sl_limit_x18 = to_clean_x18(sl_limit_price, price_step)

                sl_params = PlaceTriggerOrderParams(
                    product_id=product_id,
                    order=OrderParams(
                        sender=subaccount, 
                        amount=amount_x18, 
                        priceX18=sl_limit_x18,
                        expiration=int(time.time() + 86400 * 30), # 30 dias
                        appendix=build_appendix(
                            OrderType.DEFAULT, 
                            reduce_only=True, 
                            trigger_type=OrderAppendixTriggerType.PRICE),
                        nonce=gen_order_nonce()
                    ),
                    trigger=PriceTrigger(price_trigger=PriceTriggerData(
                        price_requirement=LastPriceBelow(last_price_below=str(sl_trigger_x18)) if is_buy
                        else LastPriceAbove(last_price_above=str(sl_trigger_x18))
                    )),
                        signature=None, id=None, digest=None, spot_leverage=None
                )
                res_sl = self.client.market.place_trigger_order(params=sl_params)
                logging.info(f"🛑 SL configurado: {sl_price} (Limit X18: {sl_limit_x18})")

            await asyncio.sleep(1.2)
            # ---------------------------------------------------------
            # 2. TAKE PROFIT (Também como Trigger Order para evitar Erro 2064)
            # ---------------------------------------------------------
            
            if tp_price:

                # Preço de Gatilho
                tp_trigger_x18 = to_clean_x18(tp_price, price_step)
                
                # No TP, preço de execução costuma ser igual ao gatilho
                tp_limit_x18 = tp_trigger_x18

                tp_trigger_params = PlaceTriggerOrderParams(
                    product_id=product_id,
                    order=OrderParams(
                        sender=subaccount, amount=amount_x18, priceX18=tp_limit_x18,
                        expiration=int(time.time() + 86400 * 30),
                        appendix=build_appendix(OrderType.DEFAULT, reduce_only=True, trigger_type=OrderAppendixTriggerType.PRICE),
                        nonce=gen_order_nonce()
                    ),
                    trigger=PriceTrigger(price_trigger=PriceTriggerData(
                        price_requirement=LastPriceAbove(last_price_above=str(tp_trigger_x18)) if is_buy 
                        else LastPriceBelow(last_price_below=str(tp_trigger_x18))
                    )),
                    signature=None, id=None, digest=None, spot_leverage=None
                )
                res_tp = self.client.market.place_trigger_order(params=tp_trigger_params)

                logging.info(f"🎯 TP configurado: {tp_price} (Trigger X18: {tp_trigger_x18})")

        except Exception as e:
            logging.error(f"❌ Erro fatal no _place_protections: {e}")
            

    async def cancel_all_orders(self, symbol: str):
        try:
            product_id = await self._get_market_id(symbol)
            if product_id is None:
                return

            subaccount = SubaccountParams(
                subaccount_owner=self.wallet_address,
                subaccount_name="default"
            )

            # 1. CANCELAR TODAS AS LIMIT ORDERS (Book comum)
            # Em vez de cancel_orders com digests vazios, usamos cancel_product_orders
            try:
                logging.info(f"🚫 [Nado] Limpando Limit Orders em {symbol}...")
                self.client.market.cancel_product_orders(
                    CancelProductOrdersParams(
                        sender=subaccount,
                        productIds=[product_id],
                        signature=None,
                        nonce=None,
                        digest=None
                    )
                )
            except Exception as e:
                logging.debug(f"ℹ️ Sem limit orders para cancelar: {e}")

            # 2. CANCELAR TODAS AS TRIGGER ORDERS (SL e TP)
            trigger_client = self.client.context.trigger_client
            if trigger_client:
                try:
                    logging.info(f"🚫 [Nado] Limpando Trigger Orders (SL/TP) em {symbol}...")
                    # O método mais seguro para limpar tudo de um produto
                    params_trigger = CancelProductTriggerOrdersParams(
                        signature=None,
                        sender=SubaccountParams(
                            subaccount_owner=self.wallet_address,
                            subaccount_name="default"
                        ),
                        nonce=None,
                        productIds=[product_id],
                        digest=None
                    )
                    trigger_client.cancel_product_trigger_orders(params_trigger)
                except Exception as e:
                    logging.debug(f"ℹ️ Sem trigger orders para cancelar: {e}")

        except Exception as e:
            logging.warning(f"⚠️ Erro ao limpar ordens em {symbol}: {e}")


    async def close_position(self, symbol: str, amount: float, side: Signal):
        logging.info(f"⚖️ [Nado] Iniciando fecho nativo via SDK para {symbol}")

        try:
            # 1. Cancelar ordens pendentes (Sempre boa prática)
            await self.cancel_all_orders(symbol)

            # 2. Obter o ID do produto
            product_id = await self._get_market_id(symbol)
            if product_id is None:
                logging.error(f"❌ Não foi possível encontrar ID para {symbol}")
                return

            # 3. Preparar a subconta no formato que o SDK espera
            # O SDK costuma aceitar o owner + nome
            subaccount  = SubaccountParams(
                subaccount_owner = self.wallet_address,
                subaccount_name = "default"
            )

            # 4. Chamar o helper nativo
            # Nota: amount e side são ignorados aqui porque o SDK vai buscar o saldo real
            order_response = self.engine.close_position(
                subaccount=subaccount,
                product_id=product_id
            )
            
            digest = getattr(order_response, 'digest', 'unknown')
            self.active_trailing_levels.pop(symbol, None)
            logging.info(f"✅ Posição fechada com sucesso! Digest: {digest}")

        except Exception as e:
            logging.error(f"❌ Erro ao fechar posição com método nativo: {e}")
            # Se o erro for que a posição já é zero, o SDK pode lançar exceção
            # Podes tratar isso aqui se quiseres


    async def apply_trailing_stop(self, symbol, current_price):
        # 1. Verifica se há posição aberta
        logging.info("Aplicar trailing stop")
        pos = await self.get_open_position(symbol)
        if not pos or abs(pos.size) < 1e-8:
            self.active_trailing_levels.pop(symbol, None)
            logging.info("Sem posição ou size < 1e-8")
            return

        entry_price = float(pos.entry_price)
        # --- AJUSTE 1: Normalização do Side ---
        side_str = str(pos.side).lower()
        is_buy = 'buy' in side_str
        
        # 2. Calcula o lucro atual
        pnl_pct = (current_price - entry_price) / entry_price if is_buy else (entry_price - current_price) / entry_price

        logging.info(f"pnl_pct={pnl_pct} side={side_str} current_price={current_price} entry_price={entry_price}")

        adjustment, icon, log = TrailingStopHelpers.get_trailing_adjustment(pnl_pct)
        logging.info(log)

        # 2. O PULO DO GATO: Verificar se já aplicamos este ajuste (ou um superior)
        last_applied = self.active_trailing_levels.get(symbol, 0)

        if adjustment > last_applied:
            # 4. Calcula os novos preços baseados no ajuste uniforme
            if is_buy:
                new_sl = entry_price * (1 + adjustment)
                new_tp = entry_price * (1.05 + adjustment) # Alvo original de 5% + ajuste
            else:
                new_sl = entry_price * (1 - adjustment)
                new_tp = entry_price * (0.95 - adjustment)

            logging.info(f"🔄 [Trailing] Reajustando proteções para {symbol} (+{adjustment:.2%})")

            # Atualizamos o estado ANTES de enviar para evitar duplicidade em caso de lag
            self.active_trailing_levels[symbol] = adjustment

            # 5. O PULO DO GATO: Reutiliza o teu método de proteções
            # Primeiro cancelamos as ordens de proteção antigas para não duplicar
            await self.cancel_all_orders(symbol)

            # 2. O PULO DO GATO: Pequena pausa para a Nado atualizar o saldo disponível
            # Sem isto, o motor de risco da Nado acha que estás a duplicar ordens
            await asyncio.sleep(0.5)
            
            product_id = await self._get_market_id(symbol)
            
            # Chamamos o método que tu já tens pronto e validado!
            await self._place_protections(
                symbol=symbol,
                product_id=product_id,
                size=abs(pos.size),
                side=side_str,
                sl_price=new_sl,
                tp_price=new_tp
            )
        else:
            # Se cair aqui, significa que o SL já está no nível correto 
            # ou o lucro ainda não subiu o suficiente para o próximo degrau.
            if adjustment > 0:
                logging.info(f"✅ Trailing em {symbol} já garantido em {adjustment:.2%}. Aguardando próximo nível.")

        
    
   
