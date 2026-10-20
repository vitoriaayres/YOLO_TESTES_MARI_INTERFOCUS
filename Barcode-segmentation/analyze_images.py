import cv2
import barcode_pipeline as bp

bp.CONFIG['roboflow_enabled'] = False
bp.CONFIG['enable_opencv_fallback'] = True

images = [
    'imagem_vinte_e_oito.jpeg',
    'imagem_vinte_e_quatro.jpeg', 
    'imagem_quarenta.jpeg',
    'imagem_onze.jpg',
    'imagem_nove.jpg',
    'imagem_dez.jpg'
]

for img_name in images:
    print(f"\n{'='*60}")
    print(f"Analisando: {img_name}")
    print(f"{'='*60}")
    try:
        image = cv2.imread(f'static/uploads/{img_name}')
        if image is None:
            print(f'  NAO ENCONTRADO')
            continue
        
        h, w = image.shape[:2]
        print(f"  Dimensoes: {w}x{h}")
        
        # Debug locate_bar_regions
        candidates, steps, scale, proposal_count = bp.locate_bar_regions(image, bp.CONFIG, return_steps=True)
        print(f"  Propostas brutas: {proposal_count}")
        print(f"  Candidatos pos-NMS: {len(candidates)}")
        
        bp.decode_regions(candidates)
        
        decoded_count = sum(1 for c in candidates if c.get('decoded'))
        print(f"  Decodificados: {decoded_count}")
        
        for i, c in enumerate(candidates):
            if c.get('decoded'):
                print(f"    Candidato {i+1}: {c.get('type')} = {c.get('value')} (decoder: {c.get('decoder')}, symbol_width_ratio: {c.get('symbol_width_ratio', 0):.3f})")
                sig = c.get('signature', {})
                print(f"      signature: accepted={sig.get('accepted')}, score={sig.get('score', 0):.3f}, edge_runs={sig.get('edge_runs')}, direction_ratio={sig.get('direction_ratio', 0):.3f}")
            else:
                sig = c.get('signature', {})
                if sig.get('accepted', False):
                    print(f"    Candidato {i+1}: NAO DECODIFICADO (accepted={sig.get('accepted')}, score={sig.get('score', 0):.3f}, edge_runs={sig.get('edge_runs')})")
        
        # Pipeline completo
        regions, _, _, _ = bp.analyze_barcode_image(image, image_name=img_name, show_debug=False, show_output=False)
        print(f"  Regioes finais: {len(regions)}")
        for r in regions:
            print(f"    Barcode {r['barcode_id']}: {r['type']} = {r['value']} (decoder: {r['decoder']})")
            
    except Exception as e:
        print(f'  ERRO: {e}')
        import traceback
        traceback.print_exc()