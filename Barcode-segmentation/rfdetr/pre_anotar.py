"""
Pré-anota imagens com o RF-DETR treinado e gera um pacote COCO para revisão.

O pacote (pasta de saída) contém as imagens renomeadas e um
_annotations.coco.json no mesmo formato do export do Roboflow, com a
classe "Barcode". Ele pode ser enviado ao Roboflow (Upload -> pasta) ou
aberto no CVAT/Label Studio para corrigir as caixas antes do treino.

Uso:
  python rfdetr/pre_anotar.py <pasta_imagens> <pasta_saida> \
      [--pesos models/checkpoint_best_total.pth] [--limiar 0.35] [--prefixo onu_google]
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import torch
from rfdetr import RFDETRSmall

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("saida")
    ap.add_argument("--pesos", default="models/checkpoint_best_total.pth")
    ap.add_argument("--limiar", type=float, default=0.35)
    ap.add_argument("--resolucao", type=int, default=640)
    ap.add_argument("--prefixo", default="img")
    args = ap.parse_args()

    torch.set_num_threads(2)
    model = RFDETRSmall(pretrain_weights=args.pesos, resolution=args.resolucao,
                        num_classes=1, device="cpu")

    entrada, saida = Path(args.entrada), Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    arquivos = sorted(p for p in entrada.iterdir() if p.suffix.lower() in EXTENSOES)

    coco = {
        "info": {"description": f"Pré-anotado pelo RF-DETR ({Path(args.pesos).name}), revisar antes de treinar"},
        "categories": [{"id": 0, "name": "Barcodes", "supercategory": "none"},
                       {"id": 1, "name": "Barcode", "supercategory": "Barcodes"}],
        "images": [], "annotations": [],
    }
    mapa = []
    for idx, origem in enumerate(arquivos, 1):
        bgr = cv2.imread(str(origem))
        if bgr is None:
            print("ignorada (não abre):", origem.name)
            continue
        h, w = bgr.shape[:2]
        nome = f"{args.prefixo}_{idx:03d}{origem.suffix.lower().replace('.jpeg', '.jpg')}"
        shutil.copy(origem, saida / nome)
        coco["images"].append({"id": idx, "file_name": nome, "width": w, "height": h})
        det = model.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), threshold=args.limiar)
        for (x1, y1, x2, y2), conf in zip(det.xyxy.tolist(), det.confidence.tolist()):
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            coco["annotations"].append({
                "id": len(coco["annotations"]) + 1, "image_id": idx, "category_id": 1,
                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                "area": round((x2 - x1) * (y2 - y1), 1), "iscrowd": 0,
                "score": round(conf, 3),
            })
        mapa.append(f"{nome};{origem.name};{w}x{h};{len(det)}")
        print(f"{nome} <- {origem.name} ({w}x{h}): {len(det)} caixas", flush=True)

    (saida / "_annotations.coco.json").write_text(json.dumps(coco, indent=1), encoding="utf-8")
    (saida / "mapa_nomes.csv").write_text("novo;original;tamanho;caixas\n" + "\n".join(mapa) + "\n",
                                          encoding="utf-8")
    print(f"\n{len(coco['images'])} imagens, {len(coco['annotations'])} caixas -> {saida}")


if __name__ == "__main__":
    main()
