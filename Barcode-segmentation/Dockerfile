# API de identificação e decodificação de códigos de barras (RF-DETR + decodificadores).
# O modelo não está no git: antes do build, coloque o arquivo em models/rfdetr_v2.onnx.
#
#   docker build -t barcode-api .
#   docker run -p 5001:5001 -e API_KEY=troque-esta-chave barcode-api
FROM python:3.11-slim

# libzbar0: pyzbar; libglib2.0-0: OpenCV headless.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libzbar0 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn==23.0.0

COPY api.py barcode_pipeline.py equipment_repository.py ./
COPY models/rfdetr_v2.onnx models/rfdetr_v2.onnx

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    API_PORT=5001 \
    API_WORKERS=2

EXPOSE 5001

# Cada worker é um processo com o seu próprio modelo na memória e lê uma
# foto por vez (alguns segundos cada). Para atender mais pedidos ao mesmo
# tempo, aumente API_WORKERS de acordo com os núcleos e a memória.
# exec: o gunicorn vira o processo principal e recebe o sinal de parada do Docker.
CMD ["sh", "-c", "exec gunicorn api:app --bind 0.0.0.0:${API_PORT} --workers ${API_WORKERS} --timeout 120"]
