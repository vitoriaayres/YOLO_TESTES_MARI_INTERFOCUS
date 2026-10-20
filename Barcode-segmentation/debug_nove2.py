import cv2
import barcode_pipeline as bp
import numpy as np

bp.CONFIG['roboflow_enabled'] = False
bp.CONFIG['enable_opencv_fallback'] = True

image = cv2.imread('static/uploads/imagem_nove.jpg')

candidates, steps, scale, proposal_count = bp.locate_bar_regions(image, bp.CONFIG, return_steps=True)

# Testa rotação 90° nos tight_rois
for i, c in enumerate(candidates):
    sig = c.get('signature', {})
    if not sig.get('accepted', False):
        continue
    
    tight_roi = c.get('tight_roi')
    if tight_roi is None or tight_roi.size == 0:
        continue
    
    print(f"\n=== Candidato {i+1}: bbox={c['bbox']} ===")
    
    # Testa original
    result = bp.decode_zxing(tight_roi)
    if result:
        print(f"  Original: {result['type']} = {result['value']}")
        continue
    
    # Testa rotacionado 90°
    rotated = cv2.rotate(tight_roi, cv2.ROTATE_90_CLOCKWISE)
    result = bp.decode_zxing(rotated)
    if result:
        print(f"  Rotacionado 90: {result['type']} = {result['value']}")
        continue
    
    # Testa rotacionado 90 anti-horário
    rotated = cv2.rotate(tight_roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
    result = bp.decode_zxing(rotated)
    if result:
        print(f"  Rotacionado -90: {result['type']} = {result['value']}")
        continue
    
    # Testa variantes rotacionadas
    for name, variant in [("orig", tight_roi), ("rot90", cv2.rotate(tight_roi, cv2.ROTATE_90_CLOCKWISE))]:
        h, w = variant.shape[:2]
        pad_y = max(12, int(round(h * 0.22)))
        pad_x = max(24, int(round(w * 0.08)))
        padded = cv2.copyMakeBorder(variant, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(255,255,255))
        
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        scale = float(np.clip(180.0 / max(h, 1), 2.0, 5.0))
        enlarged = cv2.resize(clahe, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        for vname, v in [("gray", gray), ("clahe", clahe), ("otsu", otsu), ("enlarged", enlarged)]:
            zxing_result = bp.decode_zxing(v)
            if zxing_result:
                print(f"  {name}+{vname}: {zxing_result['type']} = {zxing_result['value']}")
                break
        else:
            continue
        break
    else:
        print(f"  NENHUM decoder leu (original nem rotacionado)")