FROM python:3.10-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primeiro para cache de camadas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o resto do código
COPY . .

# Variáveis de ambiente
ENV PORT=8000
ENV MODEL_PATH=/app/app/models/best.pt

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT