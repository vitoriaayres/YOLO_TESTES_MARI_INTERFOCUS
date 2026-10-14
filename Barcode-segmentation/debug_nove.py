import cv2
import barcode_pipeline as bp
import numpy as np

bp.CONFIG['roboflow_enabled'] = False
bp.CONFIG['enable_opencv_fallback'] = True

image = cv2.imread('static/uploads/imagem_nove.jpg')
h, w = image.shape[:2]
print(f"Dimensoes: {w}x{h}")

# Analisa os candidatos
candidates, steps, scale, proposal_count = bp.locate_bar_regions(image, bp.CONFIG, return_steps=True)
print(f"Candidatos pos-NMS: {len(candidates)}")

# Testa decodificação direta em cada candidato
for i, c in enumerate(candidates):
    sig = c.get('signature', {})
    if not sig.get('accepted', False):
        continue
    
    tight_roi = c.get('tight_roi')
    decode_roi = c.get('decode_roi')
    
    print(f"\n=== Candidato {i+1}: bbox={c['bbox']} ===")
    print(f"  signature: score={sig.get('score', 0):.3f}, edge_runs={sig.get('edge_runs')}, direction_ratio={sig.get('direction_ratio', 0):.3f}")
    print(f"  tight_roi: {tight_roi.shape if tight_roi is not None else None}")
    print(f"  decode_roi: {decode_roi.shape if decode_roi is not None else None}")
    
    if tight_roi is not None and tight_roi.size > 0:
        # Testa ZXing direto no tight_roi
        result = bp.decode_zxing(tight_roi)
        print(f"  ZXing tight: {result}")
        
        # Testa variantes do decode_validated_roi
        h, w = tight_roi.shape[:2]
        pad_y = max(12, int(round(h * 0.22)))
        pad_x = max(24, int(round(w * 0.08)))
        padded = cv2.copyMakeBorder(tight_roi, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(255,255,255))
        
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        scale = float(np.clip(180.0 / max(h, 1), 2.0, 5.0))
        enlarged_gray = cv2.resize(clahe, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        for name, variant in [("gray", gray), ("clahe", clahe), ("otsu", otsu), ("enlarged", enlarged_gray)]:
            zxing_result = bp.decode_zxing(variant)
            if zxing_result:
                print(f"    ZXing {name}: {zxing_result['type']} = {zxing_result['value']}")
                break
            zbar_result = bp.decode_pyzbar(variant)
            if zbar_result:
                print(f"    ZBar {name}: {zbar_result['type']} = {zbar_result['value']}")
                break
        else:
            print(f"    NENHUM decoder leu as variantes")

# Salva tight_rois para inspeção visual
for i, c in enumerate(candidates):
    if c.get('signature', {}).get('accepted', False):
        tight = c.get('tight_roi')
        if tight is not None and tight.size > 0:
            cv2.imwrite(f'static/outputs/debug_nove_tight_{i+1}.png', tight)

print("\nTight ROIs salvos em static/outputs/debug_nove_tight_*.png")