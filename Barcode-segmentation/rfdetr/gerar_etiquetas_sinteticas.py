"""
Gera etiquetas sintéticas de ONU (Code 128 empilhados) com anotação COCO.

O RF-DETR treinado só com códigos de produto junta códigos empilhados numa
caixa só. Estas imagens imitam as etiquetas reais (ZTE, FiberHome, Huawei,
Zyxel, Nokia): 1 a 4 códigos um embaixo do outro, texto entre eles, QR codes
e logos como distração, colados sobre a carcaça do aparelho e fotografados
com inclinação, perspectiva, plástico, reflexo, sombra e desfoque.

Cada caixa cobre só as barras de um código, então o modelo aprende a
separá-los. Os valores seguem os formatos reais (MAC, D-SN, GPON SN, EN...).

Saída no mesmo formato do export do Roboflow (train/ e valid/ com
_annotations.coco.json e classe "Barcode"), pronta para juntar ao dataset.

Uso:
  python rfdetr/gerar_etiquetas_sinteticas.py <pasta_saida> [--n 2000] [--seed 0]
Dependências: numpy, opencv, pillow, python-barcode (MIT).
"""
import argparse
import json
import random
import string
from pathlib import Path

import cv2
import numpy as np
from barcode import Code128
from PIL import Image, ImageDraw, ImageFont

FONTES = [f for f in (Path("C:/Windows/Fonts") / n for n in
          ("arial.ttf", "arialbd.ttf", "calibri.ttf", "consola.ttf", "tahoma.ttf", "verdana.ttf", "segoeui.ttf"))
          if f.exists()]
HEX = "0123456789ABCDEF"
ALNUM = string.ascii_uppercase + string.digits


def rnd(chars, n):
    return "".join(random.choice(chars) for _ in range(n))


def fonte(tamanho):
    if FONTES:
        return ImageFont.truetype(str(random.choice(FONTES)), tamanho)
    return ImageFont.load_default()


# --- valores no formato das etiquetas reais -----------------------------------
def mac():
    v = rnd(HEX, 12)
    return v, random.choice([f"MAC: {'-'.join(v[i:i + 2] for i in range(0, 12, 2))}", f"MAC:{v}", v])


def zte_dsn():
    v = "ZTE" + random.choice(["YH86N", "EH8TP", "0QDLN", "YHAAR", "3BL4R", "KQDLM", "0QT1R", "PQT1R", "EH7PL"]) + rnd(ALNUM, 2) + rnd(string.digits, 5)
    return v, f"D-SN:{v}"


def zte_gpon():
    v = "ZTEG" + rnd(HEX, 8)
    return v, f"GPON SN:{v}"


def zte_en():
    v = random.choice(["YH8TQ", "EH8TQ", "0QDLQ", "KQDLN", "YH86M"]) + rnd(ALNUM, 2) + rnd(string.digits, 5)
    return v, f"EN:{v}"


def fiberhome():
    return random.choice([
        lambda: (lambda v: (v, f"S/N: {v}"))("FHTT" + rnd(HEX, 8)),
        lambda: (lambda v: (v, f"2134321-B7G(B7){v}"))("ZSEHBE" + rnd(string.digits, 6)),
    ])()


def huawei():
    return random.choice([
        lambda: (lambda v: (v, f"PROD ID:{v} Y EG8M8010HG01"))("2102311" + rnd(ALNUM, 13)),
        lambda: (lambda v: (v, f"(30S)PON No:{v}"))("48575443" + rnd(HEX, 8)),
    ])()


def outros():
    return random.choice([
        lambda: (lambda v: (v, f"SN: {v}"))("S2" + rnd(string.digits, 2) + "Y" + rnd(string.digits, 8)),
        lambda: (lambda v: (v, f"SN: {v}"))("ALCL" + rnd(HEX, 8)),
        lambda: (lambda v: (v, f"P/N: {v}"))("3FE" + rnd(string.digits, 5) + rnd("ABCD", 4)),
        lambda: (lambda v: (v, f"NS: {v}"))(rnd(ALNUM, 13)),
    ])()


GERADORES = [mac, zte_dsn, zte_gpon, zte_en, fiberhome, huawei, outros]


# --- desenho da etiqueta --------------------------------------------------------
def barras(valor, modulo, altura):
    """Imagem (uint8, 0=preto) com as barras do Code 128, sem zona de silêncio."""
    padrao = np.array([int(c) for c in Code128(valor).build()[0]], np.uint8)
    linha = np.repeat(1 - padrao, modulo) * 255
    return np.tile(linha, (altura, 1)).astype(np.uint8)


def qr_falso(tam):
    n = random.choice([21, 25, 29])
    m = (np.random.rand(n, n) > 0.5).astype(np.uint8)
    for (r, c) in ((0, 0), (0, n - 7), (n - 7, 0)):
        m[r:r + 7, c:c + 7] = 1
        m[r + 1:r + 6, c + 1:c + 6] = 0
        m[r + 2:r + 5, c + 2:c + 5] = 1
    return cv2.resize((1 - m) * 255, (tam, tam), interpolation=cv2.INTER_NEAREST)


def etiqueta():
    """Desenha uma etiqueta e devolve (imagem RGB, lista de caixas [x0,y0,x1,y1])."""
    n = random.choices([1, 2, 3, 4], weights=[1, 3, 4, 3])[0]
    modulo = random.choice([2, 2, 3, 3, 4])
    alt_barra = random.randint(14, 34) * modulo // 2
    fs = random.randint(9, 15) * modulo // 2
    gap = random.randint(1, 6) * modulo
    texto_embaixo = random.random() < 0.75
    margem = random.randint(10, 40)

    itens = [random.choice(GERADORES)() for _ in range(n)]
    bars = [barras(v, modulo, alt_barra) for v, _ in itens]
    larg_cod = max(b.shape[1] for b in bars)
    extra_dir = random.choice([0, 0, larg_cod // 2, int(larg_cod * 0.8)])   # espaço para QR/texto
    largura = larg_cod + 2 * margem + extra_dir
    topo = margem + random.choice([0, fs * 2, fs * 4, fs * 6])             # cabeçalho com texto
    altura = topo + n * (alt_barra + fs + 2 * gap) + margem + random.choice([0, fs * 3, fs * 6])

    fundo = random.choice([(255, 255, 255), (245, 245, 240), (232, 232, 232), (200, 205, 210)])
    img = Image.new("RGB", (largura, altura), fundo)
    d = ImageDraw.Draw(img)
    f = fonte(fs)
    for i in range(topo // max(fs + 4, 1) - 1):                             # cabeçalho
        d.text((margem, margem + i * (fs + 4)), random.choice(
            ["ZTE", "Home Gateway", "GPON ONT", "Model:ZXHN F601", "FiberHome", "HUAWEI", "Date: AUG 2025",
             "Rated Input: 12V 1.0A", "Product Code:RE905", "MADE IN CHINA"]), fill=(0, 0, 0), font=f)

    caixas, y = [], topo
    alinhar_esq = random.random() < 0.8
    for (valor, texto), b in zip(itens, bars):
        x = margem if alinhar_esq else margem + (larg_cod - b.shape[1]) // 2
        if not texto_embaixo:
            d.text((x, y), texto, fill=(0, 0, 0), font=f)
            y += fs + gap
        img.paste(Image.fromarray(b).convert("RGB"), (x, y))
        caixas.append([x, y, x + b.shape[1], y + alt_barra])
        y += alt_barra + gap
        if texto_embaixo:
            d.text((x, y), texto, fill=(0, 0, 0), font=f)
            y += fs + gap
    if extra_dir and random.random() < 0.8:                                  # QR de distração
        q = min(extra_dir - 10, altura - 2 * margem, 6 * alt_barra)
        if q > 20:
            img.paste(Image.fromarray(qr_falso(q)).convert("RGB"), (margem + larg_cod + 10, topo))
    for i in range(random.randint(0, 3)):                                     # rodapé
        if y + fs < altura - 4:
            d.text((margem, y + 4), random.choice(["Username:user", "Password:user", "SSID: ZTE_2.4G_" + rnd(ALNUM, 6),
                                                    "ZTE CORPORATION", "Login:http://192.168.1.1"]), fill=(0, 0, 0), font=f)
            y += fs + 4
    return np.array(img), caixas


# --- cena ----------------------------------------------------------------------
def carcaca(w, h):
    base = random.choice([random.randint(215, 245), random.randint(15, 55)])
    img = np.full((h, w, 3), base, np.uint8)
    img = (img.astype(np.int16) + np.random.randint(-6, 7, (1, 1, 3))).clip(0, 255).astype(np.uint8)
    furo = max(0, base - 70) if base > 128 else 0
    passo = random.randint(25, 60)
    tipo = random.choice(["circulo", "fenda", "nenhum"])
    for yy in range(0, h, passo):
        for xx in range((yy // passo % 2) * passo // 2, w, passo):
            if tipo == "circulo":
                cv2.circle(img, (xx, yy), passo // 4, (furo,) * 3, -1)
            elif tipo == "fenda":
                cv2.rectangle(img, (xx, yy), (xx + passo // 6, yy + passo // 2), (furo,) * 3, -1)
    return img


def perspectiva(etq, caixas, w, h):
    eh, ew = etq.shape[:2]
    angulo = random.choice([0, 0, 0, 90, 180, 270]) + random.uniform(-8, 8)
    girada = abs(round(angulo / 90)) % 2 == 1
    escala = random.uniform(0.3, 0.8) * (min(w / eh, h / ew) if girada else min(w / ew, h / eh))
    cx, cy = random.uniform(0.35, 0.65) * w, random.uniform(0.35, 0.65) * h
    M = cv2.getRotationMatrix2D((0, 0), angulo, escala)
    cantos = np.array([[0, 0], [ew, 0], [ew, eh], [0, eh]], np.float32)
    dst = cv2.transform(cantos[None], M)[0]
    dst += np.array([cx, cy]) - dst.mean(axis=0)
    dst += np.random.uniform(-0.04, 0.04, dst.shape) * max(ew, eh) * escala   # perspectiva
    H = cv2.getPerspectiveTransform(cantos, dst.astype(np.float32))
    novas = []
    for x0, y0, x1, y1 in caixas:
        pts = cv2.perspectiveTransform(np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], np.float32), H)[0]
        bx0, by0 = pts.min(axis=0)
        bx1, by1 = pts.max(axis=0)
        bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(w, bx1), min(h, by1)
        if bx1 - bx0 > 4 and by1 - by0 > 4:
            novas.append([float(bx0), float(by0), float(bx1), float(by1)])
    return H, novas


def fotografar(img):
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    # iluminação irregular e sombra
    gx, gy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    out *= (1 + random.uniform(-0.25, 0.25) * gx + random.uniform(-0.25, 0.25) * gy)[..., None]
    out *= random.uniform(0.7, 1.15)
    # plástico / reflexo
    if random.random() < 0.5:
        brilho = np.zeros((h, w), np.float32)
        for _ in range(random.randint(1, 6)):
            p1 = (random.randint(0, w), random.randint(0, h))
            p2 = (random.randint(0, w), random.randint(0, h))
            cv2.line(brilho, p1, p2, random.uniform(40, 140), random.randint(5, max(6, w // 25)))
        brilho = cv2.GaussianBlur(brilho, (0, 0), random.uniform(4, 15))
        out += brilho[..., None]
    out = out.clip(0, 255).astype(np.uint8)
    # desfoque
    if random.random() < 0.6:
        out = cv2.GaussianBlur(out, (0, 0), random.uniform(0.3, 1.6))
    if random.random() < 0.2:
        k = random.randint(3, 9)
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1 / k
        M = cv2.getRotationMatrix2D((k / 2, k / 2), random.uniform(0, 180), 1)
        out = cv2.filter2D(out, -1, cv2.warpAffine(ker, M, (k, k)))
    # ruído, baixa resolução e JPEG
    out = (out.astype(np.int16) + np.random.normal(0, random.uniform(0, 8), out.shape)).clip(0, 255).astype(np.uint8)
    if random.random() < 0.3:
        s = random.uniform(0.35, 0.7)
        out = cv2.resize(cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (w, h))
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, random.randint(45, 92)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def cena():
    w, h = random.choice([(1280, 960), (960, 1280), (1280, 720), (1024, 1024), (1600, 1200)])
    fundo = carcaca(w, h)
    ocupado = np.zeros((h, w), np.uint8)
    caixas_total = []
    for k in range(random.choices([1, 2], weights=[4, 1])[0]):   # às vezes um adesivo extra (EN)
        etq, caixas = etiqueta()
        if k == 1 and random.random() < 0.7:                        # adesivo pequeno de 1 código
            etq, caixas = etiqueta_simples()
        # O adesivo precisa caber inteiro na imagem e não cobrir outro: senão
        # ficariam caixas marcadas em códigos cortados ou tapados.
        for _ in range(15):
            H, novas = perspectiva(etq, caixas, w, h)
            cantos = cv2.perspectiveTransform(
                np.array([[[0, 0], [etq.shape[1], 0], [etq.shape[1], etq.shape[0]], [0, etq.shape[0]]]],
                         np.float32), H)[0]
            dentro = (cantos[:, 0].min() >= 2 and cantos[:, 1].min() >= 2
                      and cantos[:, 0].max() <= w - 2 and cantos[:, 1].max() <= h - 2)
            mask = cv2.warpPerspective(np.full(etq.shape[:2], 255, np.uint8), H, (w, h))
            if dentro and not (ocupado & mask).any() and len(novas) == len(caixas):
                break
        else:
            continue
        warp = cv2.warpPerspective(etq, H, (w, h), borderValue=(0, 0, 0))
        fundo[mask > 0] = cv2.cvtColor(warp, cv2.COLOR_RGB2BGR)[mask > 0]
        ocupado |= cv2.dilate(mask, np.ones((15, 15), np.uint8))
        caixas_total += novas
    return fotografar(fundo), caixas_total


def etiqueta_simples():
    valor, texto = zte_en()
    modulo = random.choice([2, 3])
    b = barras(valor, modulo, random.randint(14, 30) * modulo // 2)
    fs = random.randint(9, 14) * modulo // 2
    m = random.randint(8, 20)
    img = Image.new("RGB", (b.shape[1] + 2 * m, b.shape[0] + fs + 3 * m), (245, 245, 240))
    img.paste(Image.fromarray(b).convert("RGB"), (m, m))
    ImageDraw.Draw(img).text((m, m + b.shape[0] + 4), texto, fill=(0, 0, 0), font=fonte(fs))
    return np.array(img), [[m, m, m + b.shape[1], m + b.shape[0]]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("saida")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac_valid", type=float, default=0.1)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    raiz = Path(args.saida)
    n_valid = int(args.n * args.frac_valid)
    for split, inicio, fim in (("valid", 0, n_valid), ("train", n_valid, args.n)):
        pasta = raiz / split
        pasta.mkdir(parents=True, exist_ok=True)
        coco = {"categories": [{"id": 0, "name": "Barcodes", "supercategory": "none"},
                               {"id": 1, "name": "Barcode", "supercategory": "Barcodes"}],
                "images": [], "annotations": []}
        for i in range(inicio, fim):
            img, caixas = cena()
            nome = f"sintetico_{i:05d}.jpg"
            cv2.imwrite(str(pasta / nome), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            coco["images"].append({"id": i, "file_name": nome, "width": img.shape[1], "height": img.shape[0]})
            for x0, y0, x1, y1 in caixas:
                coco["annotations"].append({
                    "id": len(coco["annotations"]) + 1, "image_id": i, "category_id": 1,
                    "bbox": [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)],
                    "area": round((x1 - x0) * (y1 - y0), 1), "iscrowd": 0})
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{args.n}", flush=True)
        (pasta / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
        print(f"{split}: {len(coco['images'])} imagens, {len(coco['annotations'])} caixas")


if __name__ == "__main__":
    main()
