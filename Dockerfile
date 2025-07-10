# Escolhe imagem com Python 3.12.6
FROM python:3.12.6-slim

# Define diretório de trabalho no container
WORKDIR /app

# Copia arquivos do projeto
COPY . /app

# Instala dependências
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Define o comando padrão (pode ajustar conforme seu entrypoint real)
CMD ["python", "main.py"]