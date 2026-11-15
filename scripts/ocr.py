# -*- coding: utf-8 -*-
"""Processa imagens da pasta queue/ com PaddleOCR e gera JSONs em results/."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from paddleocr import PaddleOCR

QUEUE = Path("queue")
RESULTS = Path("results")
QUEUE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

MIN_LADO = 2000          # upscale mínimo (nunca reduz)
MAX_LADO = 3200          # se a imagem já é grande, não mexe
IOU_DUP = 0.55           # caixas sobrepostas = mesma detecção
DROP_SCORE = 0.5


def criar_ocr():
    """Compatível com PaddleOCR 2.x e 3.x."""
    tentativas = [
        dict(lang="en", use_angle_cls=False, show_log=False,
             det_limit_side_len=2560, det_limit_type="max",
             det_db_thresh=0.3, det_db_box_thresh=0.6,
             det_db_unclip_ratio=1.6, det_db_score_mode="slow",
             drop_score=DROP_SCORE),
        dict(lang="en", use_angle_cls=False),
        dict(lang="en"),
    ]
    for kw in tentativas:
        try:
            return PaddleOCR(**kw)
        except (TypeError, ValueError) as e:
            print(f"[OCR] kwargs rejeitados ({e}) — tentando versão mais simples")
    raise RuntimeError("Não foi possível instanciar o PaddleOCR")


def rodar_ocr(ocr, caminho):
    try:
        return ocr.ocr(str(caminho), cls=False)
    except TypeError:
        return ocr.ocr(str(caminho))


def extrair_linhas(saida):
    """Normaliza a saída para [(poly, texto, score)] em qualquer versão."""
    linhas = []
    if not saida:
        return linhas

    # 3.x: lista de OCRResult (dict-like)
    if isinstance(saida[0], dict) or hasattr(saida[0], "get"):
        for pg in saida:
            if pg is None:
                continue
            textos = pg.get("rec_texts") or []
            scores = pg.get("rec_scores") or []
            polys = pg.get("rec_polys")
            if polys is None:
                polys = pg.get("dt_polys")
            if polys is None:
                polys = pg.get("rec_boxes")
            for i, (txt, sc) in enumerate(zip(textos, scores)):
                poly = polys[i] if polys is not None and i < len(polys) else None
                linhas.append((poly, txt, float(sc)))
        return linhas

    # 2.x: [[ [poly, (txt, score)], ... ]]
    for pg in saida:
        if not pg:
            continue
        for item in pg:
            poly, (txt, score) = item[0], item[1]
            linhas.append((poly, txt, float(score)))
    return linhas


def bbox_xyxy(poly):
    if poly is None:
        return None
    a = np.array(poly, dtype="float32").reshape(-1, 2)
    x1, y1 = a.min(axis=0)
    x2, y2 = a.max(axis=0)
    return [float(x1), float(y1), float(x2), float(y2)]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    uniao = area_a + area_b - inter
    return inter / uniao if uniao > 0 else 0.0


def deduplicar(tokens):
    """Remove detecções repetidas da mesma região (causa dos números duplicados)."""
    tokens.sort(key=lambda t: t["conf"], reverse=True)
    mantidos = []
    for t in tokens:
        dup = False
        for m in mantidos:
            i = iou(t["_box"], m["_box"])
            if i > IOU_DUP or (i > 0.3 and t["text"].strip().upper() == m["text"].strip().upper()):
                dup = True
                break
        if not dup:
            mantidos.append(t)
    return mantidos


print("[OCR] Carregando modelo PaddleOCR...")
ocr = criar_ocr()

imgs = sorted(list(QUEUE.glob("*.png")) + list(QUEUE.glob("*.jpg")))
print(f"[OCR] {len(imgs)} imagem(ns) na fila")

for img in imgs:
    try:
        print(f"[OCR] {img.name}")

        pil = Image.open(img).convert("RGB")
        w, h = pil.size

        maior = max(w, h)
        if maior < MIN_LADO:
            escala = MIN_LADO / maior
        elif maior > MAX_LADO:
            escala = MAX_LADO / maior
        else:
            escala = 1.0

        if escala != 1.0:
            novo = (max(1, int(round(w * escala))), max(1, int(round(h * escala))))
            preparada = ImageOps.autocontrast(
                pil.convert("L").resize(novo, Image.LANCZOS), cutoff=1
            ).convert("RGB")
        else:
            preparada = pil

        tmp = Path("/tmp") / f"{img.stem}_prep.png"
        preparada.save(tmp)

        linhas = extrair_linhas(rodar_ocr(ocr, tmp))

        tokens = []
        for poly, texto, conf in linhas:
            texto = (texto or "").strip()
            if not texto or conf < DROP_SCORE:
                continue
            box = bbox_xyxy(poly)
            if box is None:
                continue
            if (box[2] - box[0]) < 2 or (box[3] - box[1]) < 2:
                continue
            tokens.append({
                "_box": box,
                "text": texto,
                "conf": round(conf, 4),
                "bbox": [
                    [round(min(box[0], box[2]) / escala, 1), round(min(box[1], box[3]) / escala, 1)],
                    [round(max(box[0], box[2]) / escala, 1), round(min(box[1], box[3]) / escala, 1)],
                    [round(max(box[0], box[2]) / escala, 1), round(max(box[1], box[3]) / escala, 1)],
                    [round(min(box[0], box[2]) / escala, 1), round(max(box[1], box[3]) / escala, 1)],
                ],
            })

        tokens = deduplicar(tokens)

        for t in tokens:
            b = t.pop("_box")
            t["x"] = round((b[0] + b[2]) / 2.0 / escala, 1)
            t["y"] = round((b[1] + b[3]) / 2.0 / escala, 1)

        tokens.sort(key=lambda t: (round(t["y"] / 10), t["x"]))

        (RESULTS / f"{img.stem}.json").write_text(
            json.dumps({"file": img.name, "tokens": tokens}, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[OCR]   -> {len(tokens)} token(s)")
        img.unlink()

    except Exception as e:
        print(f"[OCR] ERRO em {img.name}: {e}")
        (RESULTS / f"{img.stem}.json").write_text(
            json.dumps({"file": img.name, "error": str(e)}, ensure_ascii=False),
            encoding="utf-8",
        )

print("[OCR] Fim")
