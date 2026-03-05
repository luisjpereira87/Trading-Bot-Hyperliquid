FROM python:3.12

WORKDIR /app

# 1. Dependências de sistema (essenciais para bibliotecas crypto)
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Atualizar PIP e instalar SETUPTOOLS explicitamente
# O pkg_resources vive dentro do setuptools.
RUN pip install --no-cache-dir --upgrade pip setuptools==69.5.1 wheel

# 3. Instalar requisitos do seu bot
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar o resto do código
COPY . .

# Comando de arranque
CMD ["python", "main.py"]