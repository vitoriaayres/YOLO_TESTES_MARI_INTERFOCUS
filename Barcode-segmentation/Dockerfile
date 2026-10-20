# API de identificação e decodificação de códigos de barras (FastAPI + RF-DETR).
# O modelo não está no git: antes do build, coloque o arquivo em models/rfdetr_v2.onnx.
#
#   docker build -t barcode-api .
#   docker run -p 5001:5001 barcode-api
#
# Ou: docker compose up -d (com health check e reinício automático).
FROM python:3.12-slim

# libzbar0: pyzbar; libglib2.0-0: OpenCV headless.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libzbar0 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY barcode_pipeline.py ./
COPY app/ app/
COPY models/rfdetr_v2.onnx models/rfdetr_v2.onnx

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    API_PORT=5001 \
    API_WORKERS=2

EXPOSE 5001

# Um processo uvicorn recebe os pedidos; API_WORKERS processos de leitura,
# cada um com o seu modelo (~650 MB), leem uma foto por vez de uma fila única.
# Regra prática: 1 worker para cada 2 vCPUs (ver API.md, Desempenho).
# exec: o uvicorn vira o processo principal e recebe o sinal de parada do Docker.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${API_PORT}"]
