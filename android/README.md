# App Android: leitor de códigos de barras

Lê códigos de barras 1D (Code 128, Code 39, EAN etc.) de etiquetas de equipamentos, tudo no aparelho, sem servidor.

- **Câmera ao vivo:** ML Kit em cada quadro, com consenso entre quadros antes de confirmar.
- **Foto** (botão "Não está lendo? Tirar foto" ou aba Foto): leitura em duas passagens.
  1. ML Kit na imagem inteira.
  2. Regiões suspeitas passam por 5 camadas (recorte, ampliado, nitidez, contraste, Otsu), nesta ordem
     e até o teto de 10 s: caixas do **RF-DETR** (ONNX Runtime no aparelho), regiões do localizador por
     gradiente que o modelo não cobriu e, no tempo que sobrar, uma grade 3x3.

## Modelo RF-DETR (não versionado)

O modelo treinado não está no repositório. Para compilar, coloque o arquivo em:

```
android/app/src/main/assets/rfdetr_v2.onnx
```

Para gerar a partir do checkpoint do RF-DETR Small (`rfdetr` 1.10.1):

```
python ferramentas/exportar_onnx.py <checkpoint.pth> <pasta_saida> <pasta_fotos>
python ferramentas/quantizar_onnx.py <pasta_saida>/<modelo>.onnx rfdetr_v2.onnx <pasta_fotos>
```

A quantização int8 reduz o modelo de 126 MB para 45 MB sem perder caixas em relação ao fp32.

## Compilar e testar

```
./gradlew testDebugUnitTest assembleDebug
```

O teste instrumentado `LeituraMlKitTest` lê as fotos de `app/src/androidTest/assets/<pasta>/`.
A pasta não é versionada, porque as etiquetas reais têm credenciais de Wi-Fi. Argumentos:
`pasta=dificeis`, `detector=localizador` (sem RF-DETR) e `orcamento=sem` (sem teto de tempo).
