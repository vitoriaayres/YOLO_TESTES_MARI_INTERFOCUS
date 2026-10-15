"""Exporta o checkpoint do RF-DETR para ONNX e confere a saída contra o predict() do PyTorch."""
import glob
import inspect
import os
import sys

import numpy as np
from PIL import Image

PESOS, SAIDA, PASTA_FOTOS = sys.argv[1], sys.argv[2], sys.argv[3]

from rfdetr import RFDETRSmall

modelo = RFDETRSmall(pretrain_weights=PESOS)
print("export():", inspect.signature(modelo.export))
caminho = modelo.export(output_dir=SAIDA)
print("exportado:", caminho, os.listdir(SAIDA))

import onnxruntime as ort

onnx = caminho if isinstance(caminho, str) and caminho.endswith(".onnx") else glob.glob(os.path.join(SAIDA, "*.onnx"))[0]
sessao = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
entrada = sessao.get_inputs()[0]
print("entrada:", entrada.name, entrada.shape, "saídas:", [(o.name, o.shape) for o in sessao.get_outputs()])
res = int(entrada.shape[-1])
media = np.array([0.485, 0.456, 0.406], np.float32)
desvio = np.array([0.229, 0.224, 0.225], np.float32)


def detectar_onnx(img):
    x = np.asarray(img.convert("RGB").resize((res, res), Image.BILINEAR), np.float32) / 255.0
    x = ((x - media) / desvio).transpose(2, 0, 1)[None]
    caixas, logits = sessao.run(None, {entrada.name: x})
    conf = 1 / (1 + np.exp(-logits[0].max(axis=1)))
    w, h = img.size
    return sorted(
        [(round(float(conf[q]), 2), [int(caixas[0][q][0] * w), int(caixas[0][q][1] * h)]) for q in np.argsort(-conf)[:8] if conf[q] >= 0.1],
        reverse=True,
    )


for foto in sorted(glob.glob(os.path.join(PASTA_FOTOS, "*")))[:4]:
    img = Image.open(foto)
    d = modelo.predict(img, threshold=0.1)
    torch_res = sorted([(round(float(c), 2), [int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)]) for b, c in zip(d.xyxy, d.confidence)], reverse=True)[:8]
    print(os.path.basename(foto), "\n  torch:", torch_res, "\n  onnx: ", detectar_onnx(img))
