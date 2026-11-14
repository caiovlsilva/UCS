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
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

imgs = sorted(list(QUEUE.glob("*.png")) + list(QUEUE.glob("*.jpg")))
print(f"[OCR] {len(imgs)} imagem(ns) na fila")

for img in imgs:
    try:
        print(f"[OCR] {img.name}")
        res = ocr.ocr(str(img), cls=True)

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
        img.unlink()  # ← só deleta o PNG, meta fica para o GAS consumir

    except Exception as e:
        print(f"[OCR] ERRO em {img.name}: {e}")
        (RESULTS / f"{img.stem}.json").write_text(
            json.dumps({"file": img.name, "error": str(e)}, ensure_ascii=False),
            encoding="utf-8"
        )

# ★ NÃO deletar metas! O GAS precisa deles para consumir os results
print("[OCR] Fim")
