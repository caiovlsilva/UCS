# -*- coding: utf-8 -*-
"""Processa imagens da pasta queue/ com PaddleOCR e gera JSONs em results/."""
import json
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
from paddleocr import PaddleOCR

QUEUE   = Path("queue")
RESULTS = Path("results")
QUEUE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

print("[OCR] Carregando modelo PaddleOCR...")
ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en',
    show_log=False,
    det_limit_side_len=2400,
    det_limit_type='max',
    det_db_thresh=0.1,
    det_db_box_thresh=0.2,
    det_db_unclip_ratio=2.2,
    det_db_score_mode='slow',
    drop_score=0.1,
)

imgs = sorted(list(QUEUE.glob("*.png")) + list(QUEUE.glob("*.jpg")))
print(f"[OCR] {len(imgs)} imagem(ns) na fila")

for img in imgs:
    try:
        print(f"[OCR] {img.name}")

        pil = Image.open(img).convert("RGB")
        w_orig, h_orig = pil.size

        gray = ImageOps.grayscale(pil)
        contrast = ImageEnhance.Contrast(gray).enhance(1.5)

        if max(w_orig, h_orig) < 1600:
            scale = 1600 / max(w_orig, h_orig)
            new_size = (int(w_orig * scale), int(h_orig * scale))
        else:
            new_size = (w_orig, h_orig)
            scale = 1.0

        gray_up = contrast.resize(new_size, Image.LANCZOS)
        tmp_path = Path("/tmp") / f"{img.stem}_prep.png"
        gray_up.save(tmp_path)

        res = ocr.ocr(str(tmp_path), cls=False)

        tokens = []
        if res and res[0]:
            for line in res[0]:
                bbox, (text, conf) = line
                x = ((bbox[0][0] + bbox[2][0]) / 2.0) / scale
                y = ((bbox[0][1] + bbox[2][1]) / 2.0) / scale
                tokens.append({
                    "text": text,
                    "conf": round(float(conf), 4),
                    "x":    round(float(x), 2),
                    "y":    round(float(y), 2),
                    "bbox": [[round(float(p[0]) / scale, 2), round(float(p[1]) / scale, 2)] for p in bbox]
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
