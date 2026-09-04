import os
import cv2
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO


# Initialize Flask app
app = Flask(__name__)

# YOLO Model for Barcode Detection
model = YOLO('YOLO/YoloV8s30-best.pt')
print("YOLO Model loaded successfully!")
# Upload folder
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

LINEAR_BARCODE_SYMBOLS = [
    ZBarSymbol.CODABAR,
    ZBarSymbol.CODE39,
    ZBarSymbol.CODE93,
    ZBarSymbol.CODE128,
    ZBarSymbol.EAN8,
    ZBarSymbol.EAN13,
    ZBarSymbol.I25,
    ZBarSymbol.UPCA,
    ZBarSymbol.UPCE,
]


def get_box_center(box):
  center_x = int((box[0] + box[2]) / 2)
  center_y = int((box[1] + box[3]) / 2)
  return np.array([[center_x, center_y]])


def clamp_box(box, image_shape, padding=0):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width - 1, x2 + padding)
    y2 = min(height - 1, y2 + padding)
    return x1, y1, x2, y2


def unique_values(values):
    unique = []
    for value in values:
        if value and value not in unique:
            unique.append(value)
    return unique


def to_grayscale(image):
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image, scale):
    if scale == 1:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


# Function to preprocess the image for OCR
def preprocess_image(image):
    gray = to_grayscale(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def barcode_decode_variants(image):
    if image is None or image.size == 0:
        return []

    variants = []
    seen_shapes = set()

    for scale in (1, 2, 3):
        resized = resize_image(image, scale)
        gray = to_grayscale(resized)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        sharpened = cv2.addWeighted(gray, 1.8, cv2.GaussianBlur(gray, (0, 0), 1), -0.8, 0)

        for variant in (resized, gray, otsu, adaptive, sharpened):
            key = (variant.shape, variant.dtype.str, variant.tobytes()[:256])
            if key in seen_shapes:
                continue
            variants.append(variant)
            seen_shapes.add(key)

    return variants


# Function to read barcode
def read_barcodes(cropped_image):
    # Decode the barcode using pyzbar
    return unique_values([
        symbol["value"]
        for symbol in read_barcode_symbols(cropped_image)
        if is_linear_barcode(symbol)
    ])


def read_barcode_symbols(image):
    symbols = []
    if image is None or image.size == 0:
        return symbols

    for barcode in decode(image, symbols=LINEAR_BARCODE_SYMBOLS):
        rect = barcode.rect
        value = barcode.data.decode('utf-8')
        if not value:
            continue
        symbols.append({
            "type": barcode.type,
            "value": value,
            "bbox": [rect.left, rect.top, rect.left + rect.width, rect.top + rect.height],
        })
    return symbols


def is_linear_barcode(symbol):
    return symbol["type"] != "QRCODE"


def decode_barcode_candidates(*images):
    values = []
    for candidate in images:
        if candidate is None or candidate.size == 0:
            continue
        values.extend(read_barcodes(candidate))
        values.extend(read_barcodes(preprocess_image(candidate)))
    if values:
        return unique_values(values)

    for candidate in images:
        for variant in barcode_decode_variants(candidate):
            values.extend(read_barcodes(variant))
    return unique_values(values)


def decode_symbols_candidates(*images):
    symbols = []
    seen_values = set()
    for candidate in images:
        if candidate is None or candidate.size == 0:
            continue
        for symbol in read_barcode_symbols(candidate):
            if symbol["value"] not in seen_values:
                symbols.append(symbol)
                seen_values.add(symbol["value"])
        for symbol in read_barcode_symbols(preprocess_image(candidate)):
            if symbol["value"] not in seen_values:
                symbols.append(symbol)
                seen_values.add(symbol["value"])
    return symbols


def save_binary_contact_sheet(processed_images, output_path):
    if not processed_images:
        return

    tiles = []
    max_width = 320
    for index, image in enumerate(processed_images, start=1):
        tile = image
        if len(tile.shape) == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)

        height, width = tile.shape[:2]
        if width > max_width:
            scale = max_width / width
            tile = cv2.resize(tile, (max_width, max(1, int(height * scale))))

        tile = cv2.copyMakeBorder(tile, 32, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, f"Barcode {index}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        tiles.append(tile)

    max_height = max(tile.shape[0] for tile in tiles)
    padded_tiles = []
    for tile in tiles:
        bottom = max_height - tile.shape[0]
        padded_tiles.append(cv2.copyMakeBorder(tile, 0, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)))

    contact_sheet = cv2.hconcat(padded_tiles)
    cv2.imwrite(output_path, contact_sheet)

# Function to process the image using YOLO and SAM
def process_image(image_path, output_folder):

    # File paths for output images
    yolo_bbox_path = os.path.join(output_folder, "yolo_bbox.png")
    mask_image_path = os.path.join(output_folder, "mask_image.png")
    binary_image_path = os.path.join(output_folder, "binary_image.png")

    # Detect barcode with YOLO
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = model(image)
    print("YOLO Results:", results)

    yolo_bbox = image.copy()
    mask_overlay = image.copy()
    processed_images = []
    barcode_results = []

    if not results or len(results[0].boxes) == 0:
        cv2.imwrite(yolo_bbox_path, yolo_bbox)
        cv2.imwrite(mask_image_path, mask_overlay)
        cv2.imwrite(binary_image_path, preprocess_image(image))
        return barcode_results, yolo_bbox_path, mask_image_path, binary_image_path

    boxes = results[0].boxes.xyxy.detach().cpu().numpy()
    confidences = results[0].boxes.conf.detach().cpu().numpy()
    ordered_indexes = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))

    seen_decoded_values = set()
    decode_paddings = (8, 16, 25, 40)

    for detection_number, box_index in enumerate(ordered_indexes, start=1):
        x1, y1, x2, y2 = clamp_box(boxes[box_index], image.shape)
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(yolo_bbox, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.rectangle(mask_overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(
            yolo_bbox,
            f"D{detection_number}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

        decoded_by_value = []
        preview_crop = None
        for padding in decode_paddings:
            crop_x1, crop_y1, crop_x2, crop_y2 = clamp_box(boxes[box_index], image.shape, padding=padding)
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            yolo_crop = image[crop_y1:crop_y2 + 1, crop_x1:crop_x2 + 1]
            if preview_crop is None or padding == 25:
                preview_crop = yolo_crop

            for value in decode_barcode_candidates(yolo_crop):
                if value not in [item[0] for item in decoded_by_value]:
                    decoded_by_value.append((value, padding))

        if preview_crop is not None:
            processed_images.append(preprocess_image(preview_crop))

        added_values = []
        for value, padding in decoded_by_value:
            if value in seen_decoded_values:
                continue
            seen_decoded_values.add(value)
            added_values.append(value)
            barcode_results.append({
                "index": len(barcode_results) + 1,
                "values": [value],
                "confidence": f"{confidences[box_index]:.2f}",
                "bbox": [x1, y1, x2, y2],
                "source": f"YOLO crop D{detection_number} padding {padding}",
            })

        print(f"Detection D{detection_number}:", added_values or "not decoded")

    seen_values = {
        value
        for result in barcode_results
        for value in result["values"]
    }
    direct_symbols = sorted(
        [symbol for symbol in decode_symbols_candidates(image) if is_linear_barcode(symbol)],
        key=lambda symbol: (symbol["bbox"][1], symbol["bbox"][0]),
    )

    for symbol in direct_symbols:
        if symbol["value"] in seen_values:
            continue

        result_number = len(barcode_results) + 1
        x1, y1, x2, y2 = clamp_box(symbol["bbox"], image.shape)
        cv2.rectangle(yolo_bbox, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.rectangle(mask_overlay, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.putText(
            yolo_bbox,
            f"#{result_number}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )

        barcode_results.append({
            "index": result_number,
            "values": [symbol["value"]],
            "confidence": None,
            "bbox": [x1, y1, x2, y2],
            "source": f"Direct {symbol['type']}",
        })
        seen_values.add(symbol["value"])

    cv2.imwrite(yolo_bbox_path, yolo_bbox)
    cv2.imwrite(mask_image_path, mask_overlay)
    save_binary_contact_sheet(processed_images, binary_image_path)

    return barcode_results, yolo_bbox_path, mask_image_path, binary_image_path


# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded image
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Generate outputs
            barcode_results, yolo_bbox_path, mask_image_path, binary_image_path = process_image(
                file_path,
                app.config['OUTPUT_FOLDER']
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
