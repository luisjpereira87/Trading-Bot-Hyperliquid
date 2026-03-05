FROM python:3.12

WORKDIR /app

# 1. Instala dependências de sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Atualiza ferramentas de build e instala o SETUPTOOLS (vital para o pkg_resources)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. Copia e instala os requisitos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copia o código
COPY . .

# Comando para iniciar
CMD ["python", "main.py"]