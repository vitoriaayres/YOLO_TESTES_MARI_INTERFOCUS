# API de códigos de barras

Recebe uma foto, encontra os códigos de barras (RF-DETR Small em ONNX) e devolve o valor de cada um.
Pensada para etiquetas de equipamentos (serial/MAC em Code 128/39/93). Não depende de serviço externo.

## Subir

Feita em FastAPI (Python 3.12), servida pelo uvicorn. Documentação interativa (OpenAPI) em `/docs`.

**Docker** (qualquer servidor Linux ou nuvem). O modelo não está no git: coloque `models/rfdetr_v2.onnx` antes do build.

```sh
docker compose up -d --build        # com health check e reinício automático
# ou
docker build -t barcode-api:1.1.0 -t barcode-api:latest .
docker run -p 5001:5001 barcode-api:1.1.0
```

**Servidor de teste** (acesso por SSH, ex.: MobaXterm):

```sh
git clone https://github.com/vitoriaayres/YOLO_TESTES_MARI_INTERFOCUS
cd YOLO_TESTES_MARI_INTERFOCUS/Barcode-segmentation
# 1. Envie models/rfdetr_v2.onnx para a pasta models/ (no Moba: arrastar para o painel SFTP).
# 2. Crie o .env com a chave (fica só no servidor; o .env está no .gitignore):
echo "API_KEY=$(openssl rand -hex 32)" > .env && cat .env     # guarde a chave para os clientes
# Opcional no .env: API_WORKERS=<vCPUs / 2>
# 3. Suba e espere o status "healthy" (~1 min):
docker compose up -d --build
docker compose ps
docker compose logs api | grep chave                           # "API pronta (chave exigida)"
```

Para atualizar: `git pull` e `docker compose up -d --build`. Para trocar a chave: edite o `.env` e rode
`docker compose up -d`. A API responde em HTTP; fora da rede interna ou VPN, coloque um proxy com HTTPS na frente.

**Versão**: fica em `app/__init__.py` (e igual no `pyproject.toml`; um teste confere). A imagem leva a mesma
tag, e o `/api/v1/health` mostra a versão que está no ar. Campo novo ou erro novo sobe o segundo número
(1.1 → 1.2); mudança que quebra quem já usa a API sobe o primeiro (1.x → 2.0).

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
{"status": "ok", "detector_backend": "rfdetr", "version": "1.1.0"}
```

Se o detector não carregou, responde `503` com `"status": "degraded"`. A imagem Docker usa essa rota no
`HEALTHCHECK`, então o container fica `unhealthy` nesse caso.

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
  "unread": [
    {"confidence": 0.77, "bbox": [432, 432, 833, 481]}
  ],
  "image_size": {"width": 1280, "height": 960},
  "processing_ms": 4400
}
```

- `barcodes`: cada valor aparece uma vez, de cima para baixo. Lista vazia = nenhum código lido.
- `bbox`: `[x1, y1, x2, y2]` em pixels de `image_size` (a foto já rotacionada e, se tiver mais de 2000 px, reduzida).
- `confidence`: confiança do detector para a caixa (0 a 1).
- `unread`: caixas em que o detector achou um código mas nenhum valor foi lido (lista vazia = nada
  pendente). Serve para o app pedir outra foto, mais de perto. No gabarito, com o corte padrão de 0,5,
  aparece em 11 das 12 fotos com código faltando e, por engano, em 6 das 59 fotos completas (código EAN
  ignorado de propósito, grade de ventilação do aparelho).
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
| 503 | `busy` | já há `API_MAX_PENDING` fotos sendo lidas ou na fila; reenviar em alguns segundos |
| 503 | `worker_restarted` | um processo de leitura caiu e foi reiniciado; reenviar a foto |

## Configuração (variáveis de ambiente)

| Variável | Padrão | O que faz |
|---|---|---|
| `API_KEY` | vazia | chave exigida em `X-API-Key` |
| `API_PORT` | `5001` | porta |
| `API_WORKERS` | `2` | processos de leitura, cada um com o seu modelo (~650 MB) e uma foto por vez; 1 para cada 2 vCPUs |
| `API_MAX_PENDING` | `8` | fotos sendo lidas + na fila; acima disso responde `503 busy` (~4 por worker) |
| `API_MAX_UPLOAD_MB` | `10` | tamanho máximo do upload; acima disso responde `413` pelo `Content-Length`, antes de receber o arquivo |
| `API_MAX_IMAGE_SIDE` | `2000` | fotos maiores são reduzidas antes da leitura |
| `API_ALLOWED_FORMATS` | `CODE128,CODE39,CODE93` | formatos aceitos; vazio = todos |
| `API_VALUE_PATTERN` | `^[A-Z0-9]{6,}$` | regex que o valor precisa casar; vazio = qualquer |
| `API_SECOND_PASS_BUDGET_S` | `6` | tempo máximo (s) para insistir nas caixas difíceis |
| `API_SPLIT_UNRESOLVED_BOXES` | ligado | divide caixas com códigos empilhados |
| `API_UNREAD_MIN_CONFIDENCE` | `0.5` | confiança mínima de uma caixa não lida para entrar em `unread` |
| `API_LOG_LEVEL` | `INFO` | nível dos logs |
| `RFDETR_ONNX_PATH` | `models/rfdetr_v2.onnx` | caminho do modelo |

## Produção

O container roda sem root e tem `HEALTHCHECK`. Fica com quem hospeda:

- **HTTPS**: colocar um proxy reverso com TLS (Nginx, Caddy ou o load balancer) na frente da porta 5001.
- **Limites no proxy**: upload de pelo menos 10 MB (no Nginx, `client_max_body_size 10m`) e timeout acima
  de 30 s.
- **Chave**: definir `API_KEY` se a URL ficar acessível de fora da rede interna. Sem chave, qualquer um que
  encontre a porta pode mandar fotos. O `/docs` continua aberto, mas só mostra o contrato.

## Desempenho

No gabarito, numa CPU de notebook (8 núcleos): 4,4 s por foto em média, mediana 3 s, 90% abaixo de 9 s e
no máximo 12 s. Cada processo lê uma foto por vez; pedidos simultâneos esperam na fila.
Dentro do Docker Desktop no Windows fica mais lento, porque o container tem menos CPU.
No gabarito interno (71 fotos, 197 códigos) lê 182 códigos sem nenhuma leitura errada.
