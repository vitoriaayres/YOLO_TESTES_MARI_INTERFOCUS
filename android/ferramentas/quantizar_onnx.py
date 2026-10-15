"""Gera a versão int8 (quantização dinâmica) e compara as caixas com a fp32 em todas as fotos."""
import glob
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image

FP32, INT8, PASTA_FOTOS = sys.argv[1], sys.argv[2], sys.argv[3]
quantize_dynamic(FP32, INT8, weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul", "Gemm"])
print(f"fp32 {os.path.getsize(FP32) / 1e6:.0f} MB -> int8 {os.path.getsize(INT8) / 1e6:.0f} MB")

MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], np.float32)


def carregar(caminho):
    sessao = ort.InferenceSession(caminho, providers=["CPUExecutionProvider"])
    return sessao, sessao.get_inputs()[0].name


def detectar(sessao_nome, img, minimo=0.3):
    sessao, nome = sessao_nome
    x = np.asarray(img.convert("RGB").resize((512, 512), Image.BILINEAR), np.float32) / 255.0
    x = ((x - MEDIA) / DESVIO).transpose(2, 0, 1)[None]
    inicio = time.perf_counter()
    caixas, logits = sessao.run(None, {nome: x})
    tempo = time.perf_counter() - inicio
    conf = 1 / (1 + np.exp(-logits[0].max(axis=1)))
    return [caixas[0][q] for q in np.argsort(-conf) if conf[q] >= minimo], tempo


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


fp32, int8 = carregar(FP32), carregar(INT8)
total = achadas = 0
tempos = {"fp32": [], "int8": []}
for foto in sorted(glob.glob(os.path.join(PASTA_FOTOS, "*"))):
    img = Image.open(foto)
    ref, t32 = detectar(fp32, img)
    q, t8 = detectar(int8, img, minimo=0.1)  # a int8 usa o corte do app (0,10)
    tempos["fp32"].append(t32)
    tempos["int8"].append(t8)
    total += len(ref)
    perdidas = [r for r in ref if not any(iou(r, c) >= 0.5 for c in q)]
    achadas += len(ref) - len(perdidas)
    if perdidas:
        print("perdeu caixa em", os.path.basename(foto), len(perdidas))
print(f"caixas fp32 (conf>=0,3) reencontradas na int8: {achadas}/{total}")
print({k: f"{np.mean(v) * 1000:.0f} ms" for k, v in tempos.items()})
