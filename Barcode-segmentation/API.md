# API de códigos de barras

Recebe uma foto, encontra os códigos de barras (RF-DETR Small em ONNX) e devolve o valor de cada um.
Pensada para etiquetas de equipamentos (serial/MAC em Code 128/39/93). Não depende de serviço externo.

## Subir

**Docker** (qualquer servidor Linux ou nuvem). O modelo não está no git: coloque `models/rfdetr_v2.onnx` antes do build.

```sh
docker build -t barcode-api .
docker run -p 5001:5001 -e API_KEY=troque-esta-chave barcode-api
```

**Sem Docker** (desenvolvimento):

```sh
pip install -r requirements.txt
python api.py
```

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
  "processing_ms": 8800
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
| 500 | `internal_error` | falha inesperada |

## Configuração (variáveis de ambiente)

| Variável | Padrão | O que faz |
|---|---|---|
| `API_KEY` | vazia | chave exigida em `X-API-Key` |
| `API_PORT` | `5001` | porta |
| `API_WORKERS` | `2` | processos no Docker (cada um lê uma foto por vez) |
| `API_MAX_UPLOAD_MB` | `10` | tamanho máximo do upload |
| `API_MAX_IMAGE_SIDE` | `2000` | fotos maiores são reduzidas antes da leitura |
| `API_ALLOWED_FORMATS` | `CODE128,CODE39,CODE93` | formatos aceitos; vazio = todos |
| `API_VALUE_PATTERN` | `^[A-Z0-9]{6,}$` | regex que o valor precisa casar; vazio = qualquer |
| `API_SECOND_PASS_BUDGET_S` | `6` | tempo máximo (s) para insistir nas caixas difíceis |
| `API_SPLIT_UNRESOLVED_BOXES` | ligado | divide caixas com códigos empilhados |
| `RFDETR_ONNX_PATH` | `models/rfdetr_v2.onnx` | caminho do modelo |
| `API_EQUIPMENT_LOOKUP` | desligado | liga a consulta ao banco de exemplo (`/api/v1/equipment/<codigo>`) |

## Desempenho

Cerca de 9 s por foto em média numa CPU comum (mais em fotos difíceis), uma foto por vez por processo.
No gabarito interno (71 fotos, 197 códigos) lê 182 códigos sem nenhuma leitura errada.
