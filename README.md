# Trading Bot Hyperliquid

Bot de trading automatizado para a Hyperliquid Testnet usando Python e ccxt.

## Como usar

1. Configure suas variáveis de ambiente no arquivo `.env`:

WALLET_ADDRESS=seu_wallet_address

PRIVATE_KEY=sua_private_key

2. Instale as dependências:
pip install -r requirements.txt

3. Para treinar o modelo:
python3 main.py train

4. Execute o bot:
python3 main.py

5. Executar backtests:
python tests/backtest_runner.py;  

## Estrutura do projeto

- `trading_bot/`: código fonte do bot
- `main.py`: script principal que executa o bot
- `exchange_client.py`: interface com a exchange
- `order_manager.py`: gerenciamento de ordens
- `strategy.py`: lógica dos sinais de compra/venda