import cv2
import numpy as np
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import decode, Rect
from google.colab.patches import cv2_imshow # Importa cv2_imshow para exibição no Colab


def decodificar(img):
    return [codigo for codigo in decode(img) if codigo.type != "QRCODE"]


def rects_sobrepoem(r1, r2, margem=10):
    """Verifica se dois retângulos (x, y, w, h) se sobrepõem, com margem de tolerância."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (x1 + w1 + margem < x2 or x2 + w2 + margem < x1 or
                y1 + h1 + margem < y2 or y2 + h2 + margem < y1)


def localizar_regioes_candidatas(img_cinza, area_min_pct=0.0015):
    """Encontra regiões com padrão de código de barras via gradiente Sobel, testando as duas orientações."""
    candidatos_totais = []

    altura, largura = img_cinza.shape
    area_total = altura * largura
    area_minima = area_total * area_min_pct

    for orientacao in ["normal", "rotacionada"]:
        gradX = cv2.Sobel(img_cinza, cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(img_cinza, cv2.CV_32F, dx=0, dy=1, ksize=-1)

        if orientacao == "normal":
            gradiente = cv2.subtract(gradX, gradY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        else:
            gradiente = cv2.subtract(gradY, gradX)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 21))

        gradiente = cv2.convertScaleAbs(gradiente)
        blur = cv2.blur(gradiente, (9, 9))
        _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

        fechado = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        fechado = cv2.erode(fechado, None, iterations=2)
        fechado = cv2.dilate(fechado, None, iterations=2)

        contornos, _ = cv2.findContours(fechado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contornos:
            area = cv2.contourArea(c)
            if area < area_minima:
                continue

            x, y, w, h = cv2.boundingRect(c)
            proporcao = max(w, h) / min(w, h)

            if proporcao < 1.8:
                continue

            candidatos_totais.append((x, y, w, h))

    candidatos_unicos = []
    for cand in candidatos_totais:
        if not any(rects_sobrepoem(cand, existente, margem=5) for existente in candidatos_unicos):
            candidatos_unicos.append(cand)

    return candidatos_unicos


def tentar_forcado(recorte_cinza):
    """Bateria de tentativas mais agressivas numa região recortada."""
    tentativas = []

    # 1. upscale forte
    up = cv2.resize(recorte_cinza, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    tentativas.append(up)

    # 2. upscale + threshold adaptativo (bom para iluminação irregular/reflexo)
    thresh_adapt = cv2.adaptiveThreshold(
        up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    tentativas.append(thresh_adapt)

    # 3. upscale + sharpen (ajuda em blur leve)
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(up, -1, kernel_sharp)
    tentativas.append(sharp)

    # 4. rotações leves (±5°, ±10°) — útil se a etiqueta não está perfeitamente alinhada
    (h, w) = up.shape
    centro = (w // 2, h // 2)
    for angulo in [-10, -5, 5, 10]:
        M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
        rotacionada = cv2.warpAffine(up, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        tentativas.append(rotacionada)

    for tentativa in tentativas:
        resultado = decodificar(tentativa)
        if resultado:
            return resultado

    return []


def processar_imagem(caminho):
    img = cv2.imread(caminho)
    if img is None:
        raise FileNotFoundError(f"Não consegui abrir: {caminho}")

    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. decodifica o que for fácil
    codigos_ok = decodificar(img)
    print(f"Decodificados de primeira: {len(codigos_ok)}")

    rects_ok = [c.rect for c in codigos_ok]  # cada rect: Rect(left, top, width, height)

    # 2. localiza todas as regiões candidatas a código de barras
    candidatas = localizar_regioes_candidatas(cinza)

    # 3. filtra só as candidatas que NÃO batem com nenhum código já decodificado
    faltando = []
    for cand in candidatas:
        ja_decodificado = any(
            rects_sobrepoem(cand, (r.left, r.top, r.width, r.height))
            for r in rects_ok
        )
        if not ja_decodificado:
            faltando.append(cand)

    print(f"Regiões candidatas sem decodificação: {len(faltando)}")

    # 4. ataca cada região faltante com técnicas mais fortes
    resultados_extra = []
    for (x, y, w, h) in faltando:
        margem = 15
        y1 = max(0, y - margem)
        y2 = min(cinza.shape[0], y + h + margem)
        x1 = max(0, x - margem)
        x2 = min(cinza.shape[1], x + w + margem)
        recorte = cinza[y1:y2, x1:x2]

        if recorte.size == 0:
            continue

        extra = tentar_forcado(recorte)
        if extra:
            print(f"  -> Região ({x},{y},{w},{h}) recuperada com técnica forçada!")

            # remapeia o rect de cada código pro retângulo do recorte original,
            # já em coordenadas da imagem grande
            novo_rect = Rect(left=x1, top=y1, width=(x2 - x1), height=(y2 - y1))
            extra = [codigo._replace(rect=novo_rect) for codigo in extra]

            resultados_extra.extend(extra)
            filename = f"debug_recuperado_{x}_{y}.jpg"
            cv2.imwrite(filename, recorte)
            print(f"    Recorte de debug salvo: {filename}")
        else:
            print(f"  -> Região ({x},{y},{w},{h}) continua ilegível. Nenhum recorte de debug salvo para esta região.")

    todos = codigos_ok + resultados_extra

    vistos = set()
    todos_dedup = []
    for c in todos:
        if c.data not in vistos:
            vistos.add(c.data)
            todos_dedup.append(c)
    todos = todos_dedup

    # Desenha todos os códigos de barras decodificados na imagem original
    for codigo in todos:
        (x_rect, y_rect, w_rect, h_rect) = codigo.rect
        cv2.rectangle(img, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 255, 0), 2)

        dado = codigo.data.decode("utf-8")
        cv2.putText(img, dado, (x_rect, y_rect - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return todos, img


if __name__ == "__main__":
    resultados, img_final = processar_imagem("/content/imagens_novo/imagens/imagem_dezesseis.jpeg") # Caminho da imagem corrigido
    print(f"\nTotal final: {len(resultados)} códigos")
    for c in resultados:
        dado = c.data.decode("utf-8")
        print(f"{c.type}: {dado}")

    # Exibe a imagem final com os códigos de barras desenhados
    cv2_imshow(img_final)
