# API de códigos de barras

Recebe uma foto, encontra os códigos de barras (RF-DETR Small em ONNX) e devolve o valor de cada um.
Pensada para etiquetas de equipamentos (serial/MAC em Code 128/39/93). Não depende de serviço externo.

## Subir

Feita em FastAPI (Python 3.12), servida pelo uvicorn. Documentação interativa (OpenAPI) em `/docs`.

**Docker** (qualquer servidor Linux ou nuvem). O modelo não está no git: coloque `models/rfdetr_v2.onnx` antes do build.

```sh
docker compose up -d --build        # com health check e reinício automático
# ou
docker build -t barcode-api .
docker run -p 5001:5001 barcode-api
```

**Sem Docker** (desenvolvimento):

```sh
python -m venv .venv                 # Python 3.12
.venv/Scripts/activate               # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --port 5001
pytest                               # testes (não carregam o modelo)
```

### Organização do código

| Pasta | O que tem |
|---|---|
| `app/api/routes` | endpoints (só recebem, validam e chamam o serviço) |
| `app/api/dependencies` | chave `X-API-Key`, leitura do upload, acesso aos serviços |
| `app/schemas` | contratos Pydantic de resposta e de erro |
| `app/services` | regra de negócio (`DecodeService`), pool de processos de leitura, leitura da imagem e o adaptador do `barcode_pipeline.py` |
| `app/core` | configuração (variáveis `API_*`), logs e erros |
| `tests/unit`, `tests/api` | testes com um leitor falso, sem o modelo |

## Testar pelo Postman

Importe `barcode-api.postman_collection.json` (Ctrl+O no Postman). As variáveis `baseUrl` e `apiKey`
ficam na aba Variables da coleção. Em cada pedido Decode, escolha a foto em Body → `image` → Select Files.
O pedido "Decode (debug)" mostra a foto anotada na aba Visualize da resposta.

Se o Postman der `EISDIR` ou um aviso amarelo no arquivo, ligue em Settings → General → Working Directory
a opção "Allow reading files outside this directory" e escolha a foto de novo.

## Autenticação

Se a variável `API_KEY` estiver definida, as chamadas precisam do cabeçalho `X-API-Key: <chave>`.
Sem ela, a API fica aberta (útil se quem hospeda já protege num gateway). `/` e `/api/v1/health` nunca pedem chave.

## Endpoints

### `GET /api/v1/health`

```json
{"status": "ok", "detector_backend": "rfdetr"}
```

### `POST /api/v1/decode`

`multipart/form-data` com a foto (JPEG ou PNG) no campo `image`. A rotação EXIF do celular é respeitada.

```sh
curl -X POST http://localhost:5001/api/v1/decode \
  -H "X-API-Key: troque-esta-chave" \
  -F "image=@etiqueta.jpg"
```

Resposta `200`:

```json
{
  "barcodes": [
    {
      "index": 1,
      "value": "ZTEYHAAR8303110",
      "format": "CODE128",
      "confidence": 1.0,
      "bbox": [72, 452, 419, 492],
      "decoder": "pyzbar"
    }
  ],
  "image_size": {"width": 1280, "height": 960},
  "processing_ms": 4400
}
```

- `barcodes`: cada valor aparece uma vez, de cima para baixo. Lista vazia = nenhum código lido.
- `bbox`: `[x1, y1, x2, y2]` em pixels de `image_size` (a foto já rotacionada e, se tiver mais de 2000 px, reduzida).
- `confidence`: confiança do detector para a caixa (0 a 1).
- `?debug=true` acrescenta `debug.annotated_jpeg_base64` com a foto anotada.

### Erros

Sempre JSON: `{"error": {"code": "...", "message": "..."}}`.

| Status | `code` | Quando |
|---|---|---|
| 400 | `missing_image`, `empty_image`, `invalid_image` | sem campo `image`, arquivo vazio ou formato ilegível |
| 401 | `unauthorized` | `API_KEY` definida e cabeçalho ausente ou errado |
| 413 | `image_too_large` | foto acima de `API_MAX_UPLOAD_MB` |
| 422 | `validation_error` | parâmetro inválido (ex.: `?debug=talvez`) |
| 500 | `internal_error` | falha inesperada |
| 503 | `worker_restarted` | um processo de leitura caiu e foi reiniciado; reenviar a foto |

## Configuração (variáveis de ambiente)

| Variável | Padrão | O que faz |
|---|---|---|
| `API_KEY` | vazia | chave exigida em `X-API-Key` |
| `API_PORT` | `5001` | porta |
| `API_WORKERS` | `2` | processos de leitura, cada um com o seu modelo (~650 MB) e uma foto por vez; 1 para cada 2 vCPUs |
| `API_MAX_UPLOAD_MB` | `10` | tamanho máximo do upload |
| `API_MAX_IMAGE_SIDE` | `2000` | fotos maiores são reduzidas antes da leitura |
| `API_ALLOWED_FORMATS` | `CODE128,CODE39,CODE93` | formatos aceitos; vazio = todos |
| `API_VALUE_PATTERN` | `^[A-Z0-9]{6,}$` | regex que o valor precisa casar; vazio = qualquer |
| `API_SECOND_PASS_BUDGET_S` | `6` | tempo máximo (s) para insistir nas caixas difíceis |
| `API_SPLIT_UNRESOLVED_BOXES` | ligado | divide caixas com códigos empilhados |
| `API_LOG_LEVEL` | `INFO` | nível dos logs |
| `RFDETR_ONNX_PATH` | `models/rfdetr_v2.onnx` | caminho do modelo |

## Desempenho

No gabarito, numa CPU de notebook (8 núcleos): 4,4 s por foto em média, mediana 3 s, 90% abaixo de 9 s e
no máximo 12 s. Cada processo lê uma foto por vez; pedidos simultâneos esperam na fila.
Dentro do Docker Desktop no Windows fica mais lento, porque o container tem menos CPU.
No gabarito interno (71 fotos, 197 códigos) lê 182 códigos sem nenhuma leitura errada.
