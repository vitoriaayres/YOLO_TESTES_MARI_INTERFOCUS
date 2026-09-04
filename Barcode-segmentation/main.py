import os
import traceback

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect

from werkzeug.utils import secure_filename

try:
    from dotenv import load_dotenv
    load_dotenv()  # lê o arquivo .env na mesma pasta e define as variáveis de ambiente
except ImportError:
    pass  # sem python-dotenv instalado, só funciona se a variável já estiver setada no terminal

import barcode_pipeline as bp

# Initialize Flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Backend Roboflow ativo: {bp.ROBOFLOW_BACKEND}")
if bp.ROBOFLOW_BACKEND == "desativado":
    print(
        "AVISO: ROBOFLOW_API_KEY não foi definida (ou o modelo falhou ao "
        "carregar). O pipeline vai rodar só com os fallbacks locais, que "
        "estão desligados por padrão no CONFIG (enable_opencv_fallback=False)."
    )


def _placeholder_image(path, message):
    """Salva uma imagem simples com um aviso, para quando não há nada a mostrar."""
    blank = np.full((200, 900, 3), 245, dtype=np.uint8)
    cv2.putText(
        blank, message, (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA,
    )
    cv2.imwrite(path, blank)


def process_image(image_path, output_folder):
    """
    Roda o pipeline Roboflow + OpenCV (analyze_barcode_image) numa imagem
    e devolve os mesmos 4 valores que main_no_sam2.py devolvia, para que
    o template index.html continue funcionando sem alterações:
        barcode_results, yolo_bbox_path, mask_image_path, binary_image_path
    """
    yolo_bbox_path = os.path.join(output_folder, "yolo_bbox.png")
    mask_image_path = os.path.join(output_folder, "mask_image.png")
    binary_image_path = os.path.join(output_folder, "binary_image.png")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_name = os.path.basename(image_path)

    regions, annotated, table, visual_stack = bp.analyze_barcode_image(
        image, image_name=image_name, show_debug=True, show_output=False
    )

    # "yolo_bbox" = imagem original com as caixas dos códigos confirmados.
    cv2.imwrite(yolo_bbox_path, annotated)

    # "mask_image" antes mostrava a máscara verde do YOLO; aqui reaproveitamos
    # a mesma imagem anotada (não existe mais uma etapa de máscara separada).
    cv2.imwrite(mask_image_path, annotated)

    # "binary_image" agora é a pilha visual: mostra recorte -> super-resolução
    # -> grayscale -> contraste -> threshold, camada por camada, por candidato.
    if visual_stack is not None:
        cv2.imwrite(binary_image_path, visual_stack)
    else:
        _placeholder_image(
            binary_image_path, "Nenhuma pilha visual gerada para esta imagem."
        )

    barcode_results = []
    for region in regions:
        x, y, w, h = region["bbox"]
        signature = region.get("signature", {})
        confidence = region.get("roboflow_confidence")
        if confidence is None:
            confidence = signature.get("score")

        source_parts = [region.get("decoder", "")]
        if region.get("variant"):
            source_parts.append(region["variant"])
        if region.get("crop"):
            source_parts.append(region["crop"])
        source = " | ".join(p for p in source_parts if p)

        barcode_results.append({
            "index": region.get("barcode_id", len(barcode_results) + 1),
            "values": [region["value"]],
            "confidence": f"{confidence:.2f}" if confidence is not None else None,
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "source": source,
        })

    return barcode_results, yolo_bbox_path, mask_image_path, binary_image_path


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                barcode_results, yolo_bbox_path, mask_image_path, binary_image_path = process_image(
                    file_path,
                    app.config['OUTPUT_FOLDER']
                )
            except Exception:
                traceback.print_exc()
                return render_template(
                    "index.html",
                    original_image=file_path,
                    error="Falha ao processar a imagem. Veja o log do servidor.",
                )

            return render_template(
                "index.html",
                original_image=file_path,
                yolo_bbox=yolo_bbox_path,
                mask_image=mask_image_path,
                binary_image=binary_image_path,
                barcode_results=barcode_results,
            )

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)