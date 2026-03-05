# Usa a versão 3.12 para bater certo com o teu PC (e evita o -slim por agora)
FROM python:3.12

WORKDIR /app

# Instala dependências de sistema necessárias para compilar pacotes de criptografia
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Atualiza as ferramentas de instalação ANTES de instalar o CCXT
RUN pip install --upgrade pip setuptools wheel

# Copia e instala os requisitos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]