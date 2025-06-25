# Trading Bot Hyperliquid

Bot de trading automatizado para a Hyperliquid Testnet usando Python e ccxt.

## Como usar

1. Configure suas variáveis de ambiente no arquivo `.env`:
WALLET_ADDRESS=seu_wallet_address
PRIVATE_KEY=sua_private_key

2. Instale as dependências:
pip install -r requirements.txt

3. Execute o bot:
python3 trading_bot/main.py

## Estrutura do projeto

- `trading_bot/`: código fonte do bot
- `main.py`: script principal que executa o bot
- `exchange_client.py`: interface com a exchange
- `order_manager.py`: gerenciamento de ordens
- `strategy.py`: lógica dos sinais de compra/venda