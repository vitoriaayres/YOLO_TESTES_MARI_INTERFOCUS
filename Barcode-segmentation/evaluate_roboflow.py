"""
Compara o pipeline COM e SEM o detector Roboflow numa pasta de imagens.

As imagens passam pelo mesmo pré-processamento da API (rotação EXIF,
redução para API_MAX_IMAGE_SIDE e filtros de formato/valor), então o resultado reflete o que o app
mobile vai receber.

Uso:
  python evaluate_roboflow.py <pasta_imagens> [--expected gabarito.csv] [--output resultado.csv]

Gabarito (opcional), CSV com cabeçalho:
  arquivo,codigo_esperado
  imagem_um.jpg,7891234567890
  imagem_dois.jpg,                <- vazio = imagem sem código legível

Sem gabarito, o relatório mostra quem leu, quem não leu e as divergências.
"""
import argparse
import contextlib
import csv
import io
import os
import sys
from time import perf_counter

import barcode_pipeline as bp
from app.core.config import get_settings
from app.core.exceptions import ApiError
from app.services.barcode_reader import configure_pipeline
from app.services.image_codec import decode_upload

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_expected(path):
    if not path:
        return None
    expected = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            codes = {c.strip() for c in row["codigo_esperado"].split(";") if c.strip()}
            expected[row["arquivo"].strip()] = codes
    return expected


def run_pipeline(image, name, roboflow_on):
    # Liga/desliga o detector trocando os globais lidos por
    # roboflow_barcode_candidates(); o restante do pipeline fica igual.
    saved = bp.ROBOFLOW_MODEL, bp.ROBOFLOW_CLIENT
    if not roboflow_on:
        bp.ROBOFLOW_MODEL, bp.ROBOFLOW_CLIENT = None, None
    try:
        start = perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            regions, *_ = bp.analyze_barcode_image(
                image, image_name=name, show_debug=False, show_output=False
            )
        elapsed = perf_counter() - start
    finally:
        bp.ROBOFLOW_MODEL, bp.ROBOFLOW_CLIENT = saved
    return {str(r["value"]) for r in regions}, elapsed


def classify(read, expected):
    if expected is None:
        return "leu" if read else "nao_leu"
    if not expected:
        return "falso_positivo" if read else "ok_vazio"
    if expected <= read:
        return "acerto" if read == expected else "acerto_com_extra"
    if expected & read:
        return "parcial"
    return "erro" if read else "nao_leu"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("images")
    parser.add_argument("--expected")
    parser.add_argument("--output", default="avaliacao_roboflow.csv")
    args = parser.parse_args()

    if bp.ROBOFLOW_BACKEND not in {"rfdetr", "local", "hosted"}:
        sys.exit(f"Detector não está ativo (backend={bp.ROBOFLOW_BACKEND}); "
                 "confira o modelo ONNX ou a ROBOFLOW_API_KEY no .env.")

    settings = get_settings()
    configure_pipeline(bp, settings)
    expected_map = load_expected(args.expected)
    files = sorted(
        f for f in os.listdir(args.images)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )
    if not files:
        sys.exit("Nenhuma imagem encontrada.")

    # Aquece o modelo para o tempo da primeira imagem não distorcer a média.
    with open(os.path.join(args.images, files[0]), "rb") as handle:
        run_pipeline(decode_upload(handle.read(), settings.max_image_side), "warmup", True)

    rows = []
    for index, name in enumerate(files, 1):
        with open(os.path.join(args.images, name), "rb") as handle:
            try:
                image = decode_upload(handle.read(), settings.max_image_side)
            except ApiError as error:
                print(f"[{index}/{len(files)}] {name}: ignorada ({error.message})")
                continue

        expected = expected_map.get(name) if expected_map is not None else None
        with_rf, time_with = run_pipeline(image, name, True)
        without_rf, time_without = run_pipeline(image, name, False)

        row = {
            "arquivo": name,
            "esperado": ";".join(sorted(expected)) if expected is not None else "",
            "com_roboflow": ";".join(sorted(with_rf)),
            "sem_roboflow": ";".join(sorted(without_rf)),
            "status_com": classify(with_rf, expected),
            "status_sem": classify(without_rf, expected),
            "concordam": with_rf == without_rf,
            "tempo_com_s": round(time_with, 2),
            "tempo_sem_s": round(time_without, 2),
        }
        rows.append(row)
        print(
            f"[{index}/{len(files)}] {name}: "
            f"com={row['com_roboflow'] or '-'} ({time_with:.1f}s) | "
            f"sem={row['sem_roboflow'] or '-'} ({time_without:.1f}s)"
        )

    with open(args.output, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)

    def count(key, value):
        return sum(1 for r in rows if r[key] == value)

    def mean(key):
        return sum(r[key] for r in rows) / total

    print("\n================ RESUMO ================")
    print(f"Imagens avaliadas: {total}")
    if expected_map is None:
        read_with = sum(1 for r in rows if r["com_roboflow"])
        read_without = sum(1 for r in rows if r["sem_roboflow"])
        print(f"Leram algum código  COM Roboflow: {read_with}/{total}")
        print(f"Leram algum código  SEM Roboflow: {read_without}/{total}")
        print(f"Só COM Roboflow leu: "
              f"{sum(1 for r in rows if r['com_roboflow'] and not r['sem_roboflow'])}")
        print(f"Só SEM Roboflow leu: "
              f"{sum(1 for r in rows if r['sem_roboflow'] and not r['com_roboflow'])}")
        print(f"Ambos leram, valores diferentes: "
              f"{sum(1 for r in rows if r['com_roboflow'] and r['sem_roboflow'] and not r['concordam'])}")
    else:
        for status in ("acerto", "acerto_com_extra", "parcial", "erro",
                       "nao_leu", "falso_positivo", "ok_vazio"):
            print(f"{status:>18}: COM={count('status_com', status):>3} "
                  f"SEM={count('status_sem', status):>3}")
    print(f"Tempo médio COM Roboflow: {mean('tempo_com_s'):.2f}s")
    print(f"Tempo médio SEM Roboflow: {mean('tempo_sem_s'):.2f}s")
    print(f"Detalhes por imagem: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
