# -*- coding: utf-8 -*-
"""Processa imagens da pasta queue/ com PaddleOCR e gera JSONs em results/."""
import json
from pathlib import Path
from paddleocr import PaddleOCR

QUEUE   = Path("queue")
RESULTS = Path("results")
QUEUE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

print("[OCR] Carregando modelo PaddleOCR...")
# ★ Ajustes para detectar texto pequeno (attrs SPD/STR/POW/SKL/RES):
#    - det_limit_side_len=1600 → não reduz tanto a resolução
#    - det_db_thresh=0.2       → threshold mais baixo para considerar pixel como texto
#    - det_db_box_thresh=0.3   → aceita box de texto com menor confiança
ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en',
    show_log=False,
    det_limit_side_len=1600,
    det_db_thresh=0.2,
    det_db_box_thresh=0.3,
    det_db_unclip_ratio=1.8,
)

imgs = sorted(list(QUEUE.glob("*.png")) + list(QUEUE.glob("*.jpg")))
print(f"[OCR] {len(imgs)} imagem(ns) na fila")

for img in imgs:
    try:
        print(f"[OCR] {img.name}")
        res = ocr.ocr(str(img), cls=False)

        tokens = []
        if res and res[0]:
            for line in res[0]:
                bbox, (text, conf) = line
                x = (bbox[0][0] + bbox[2][0]) / 2.0
                y = (bbox[0][1] + bbox[2][1]) / 2.0
                tokens.append({
                    "text": text,
                    "conf": round(float(conf), 4),
                    "x":    round(float(x), 2),
                    "y":    round(float(y), 2),
                    "bbox": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in bbox]
                })

        (RESULTS / f"{img.stem}.json").write_text(
            json.dumps({"file": img.name, "tokens": tokens}, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"[OCR]   -> {len(tokens)} token(s)")
        img.unlink()

    except Exception as e:
        print(f"[OCR] ERRO em {img.name}: {e}")
        (RESULTS / f"{img.stem}.json").write_text(
            json.dumps({"file": img.name, "error": str(e)}, ensure_ascii=False),
            encoding="utf-8"
        )

print("[OCR] Fim")
