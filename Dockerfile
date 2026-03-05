FROM python:3.10

# 1. Definir o diretório de trabalho logo no início
WORKDIR /app

# 2. Instala dependências do sistema
# Adicionei 'git' e 'ca-certificates' que o CCXT às vezes usa para submódulos
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 3. Atualizar o PIP é CRUCIAL para pacotes complexos como CCXT
RUN pip install --upgrade pip setuptools wheel

# 4. Instala dependências Python (antes de copiar o código para aproveitar o cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia o código do bot
COPY . .

# Comando para iniciar o bot
CMD ["python", "main.py"]