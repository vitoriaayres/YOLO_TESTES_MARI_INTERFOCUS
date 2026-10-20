# API de códigos de barras

Recebe uma foto, encontra os códigos de barras com o RF-DETR Small (ONNX) e devolve o valor de cada um.
Feita em FastAPI (Python 3.12) e entregue como imagem Docker.

- Como subir, rotas, configuração e desempenho: [API.md](API.md)
- Coleção do Postman para testar: `barcode-api.postman_collection.json`
- Documentação interativa com a API no ar: `/docs`

O modelo (`models/rfdetr_v2.onnx`, 115 MB) não fica no git; ele precisa estar nessa pasta antes do `docker build`.

O treino do modelo, o gabarito de avaliação e os experimentos antigos ficam na branch `treino_rfdetr`.
