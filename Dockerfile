FROM python:3.10-slim

WORKDIR /app

# Instalar dependências do sistema necessárias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgstreamer1.0-0 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primeiro para cache de camadas
COPY requirements.txt .

# Instalar numpy primeiro para garantir compatibilidade
RUN pip install --no-cache-dir numpy==1.23.5

# Instalar outras dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o resto do código
COPY . .

# Variáveis de ambiente
ENV PORT=8000
ENV MODEL_PATH=/app/app/models/best.pt
ENV PYTHONPATH=/app

# Comando para iniciar a aplicação
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}