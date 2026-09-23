"""Local OCR engine module with Korean PP-OCRv3 ONNX support."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

_OCR_ENGINE = None


def get_korean_ocr():
    """Initializes and caches a RapidOCR instance with Korean recognition model if available."""
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE

    try:
        import rapidocr_onnxruntime
        from rapidocr_onnxruntime import RapidOCR
        from rapidocr_onnxruntime.utils import read_yaml, concat_model_path
        from rapidocr_onnxruntime.ch_ppocr_v3_rec.text_recognize import TextRecognizer
        from rapidocr_onnxruntime.ch_ppocr_v3_rec.utils import CTCLabelDecode

        # use_angle_cls=False prevents confusing rotated boxes on tabular / numeric lines
        ocr = RapidOCR(use_angle_cls=False)

        # 모델 경로 탐색
        candidates = [
            Path(r"C:\Purchasing Tool\models"),
            Path(__file__).resolve().parent.parent / "models",
        ]
        model_dir = None
        for cand in candidates:
            if (cand / "korean_PP-OCRv3_rec_infer.onnx").exists() and (cand / "korean_dict.txt").exists():
                model_dir = cand
                break

        if model_dir:
            pkg_dir = Path(rapidocr_onnxruntime.__file__).parent
            config_path = pkg_dir / "config.yaml"
            if config_path.exists():
                config = read_yaml(str(config_path))
                config = concat_model_path(config)
                config['Rec']['model_path'] = str(model_dir / "korean_PP-OCRv3_rec_infer.onnx")
                config['Rec']['keys_path'] = str(model_dir / "korean_dict.txt")

                with open(config['Rec']['keys_path'], "r", encoding="utf-8") as f:
                    chars = [line.strip("\r\n") for line in f]

                recognizer = TextRecognizer(config['Rec'])
                recognizer.postprocess_op = CTCLabelDecode(chars)
                ocr.text_recognizer = recognizer

        _OCR_ENGINE = ocr
        return _OCR_ENGINE
    except Exception as e:
        print(f"[RapidOCR Init Warning] {e}")
        return None


def ocr_image(img_input: Union[Image.Image, np.ndarray, str, Path]) -> str:
    """Performs OCR on an image using RapidOCR Korean model or pytesseract fallback."""
    # 1. Convert input to cv2 BGR image
    if isinstance(img_input, (str, Path)):
        img_cv = cv2.imread(str(img_input))
    elif isinstance(img_input, Image.Image):
        img_cv = cv2.cvtColor(np.array(img_input.convert("RGB")), cv2.COLOR_RGB2BGR)
    elif isinstance(img_input, np.ndarray):
        img_cv = img_input
    else:
        return ""

    if img_cv is None or img_cv.size == 0:
        return ""

    # 2. Try RapidOCR (Korean + Default Numeric with Spatial Row Clustering)
    try:
        import re
        from rapidocr_onnxruntime import RapidOCR
        ocr_kor = get_korean_ocr()
        ocr_def = RapidOCR(use_angle_cls=False)

        h, w = img_cv.shape[:2]
        if max(h, w) > 2500:
            scale = 2000 / max(h, w)
            img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        elif max(h, w) < 1200:
            scale = min(2.0, 1500 / max(h, w))
            img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        res_k, _ = ocr_kor(img_cv) if ocr_kor is not None else (None, None)
        res_d, _ = ocr_def(img_cv) if ocr_def is not None else (None, None)

        def _box_center(b):
            return (float(b[0][0] + b[2][0]) / 2.0, float(b[0][1] + b[2][1]) / 2.0)

        def _box_height(b):
            return abs(float(b[2][1] - b[0][1]))

        items = []
        used_d = set()
        for rk in (res_k or []):
            bk, tk, ck = rk[0], rk[1], float(rk[2])
            cx_k, cy_k = _box_center(bk)
            hk = _box_height(bk)

            best_d = None
            best_dist = 999999
            best_idx = -1
            for idx, rd in enumerate(res_d or []):
                if idx in used_d:
                    continue
                cx_d, cy_d = _box_center(rd[0])
                dist = np.hypot(cx_k - cx_d, cy_k - cy_d)
                if dist < max(hk, 20) and dist < best_dist:
                    best_dist = dist
                    best_d = rd
                    best_idx = idx

            chosen_text = tk
            chosen_box = bk
            if best_d is not None and best_dist < max(hk, 20):
                used_d.add(best_idx)
                td = best_d[1]
                cd = float(best_d[2])
                has_digits_d = bool(re.search(r'\d{3,}', td))
                has_digits_k = bool(re.search(r'\d{3,}', tk))

                brand_words = ('imac', 'macbook', 'ipad', 'applecare', 'pencil', 'keyboard', 'retina', 'nano-texture')
                has_brand_d = any(bw in td.lower() for bw in brand_words)
                has_brand_k = any(bw in tk.lower() for bw in brand_words)

                if has_brand_d and not has_brand_k:
                    if re.search(r'[\uac00-\ud7a3]', tk):
                        chosen_text = f'{tk} {td}'
                    else:
                        chosen_text = td
                elif has_digits_d and not has_digits_k:
                    chosen_text = td
                elif has_digits_d and has_digits_k:
                    if ',' in td and ',' not in tk:
                        chosen_text = td
                    elif cd > ck:
                        chosen_text = td
                elif cd > ck + 0.2 and not re.search(r'[\uac00-\ud7a3]', tk):
                    chosen_text = td

            items.append((chosen_box, chosen_text))

        for idx, rd in enumerate(res_d or []):
            if idx not in used_d:
                cd = float(rd[2])
                td = rd[1]
                # Filter out low-confidence detections from default model to eliminate background watermark tokens
                if cd >= 0.70 or (cd >= 0.60 and re.search(r'\d{3,}', td)):
                    items.append((rd[0], td))

        if items:
            items.sort(key=lambda it: _box_center(it[0])[1])
            rows = []
            for box, text in items:
                cx, cy = _box_center(box)
                h = max(_box_height(box), 12)
                matched_row = None
                for r in rows:
                    if abs(r['y'] - cy) < h * 0.6:
                        matched_row = r
                        break
                if matched_row is not None:
                    matched_row['items'].append((cx, text))
                    matched_row['y'] = (matched_row['y'] * (len(matched_row['items']) - 1) + cy) / len(matched_row['items'])
                else:
                    rows.append({'y': cy, 'items': [(cx, text)]})

            rows.sort(key=lambda r: r['y'])
            lines = []
            for r in rows:
                r['items'].sort(key=lambda it: it[0])
                row_txt = '   '.join(it[1] for it in r['items']).strip()
                if row_txt:
                    lines.append(row_txt)

            if lines:
                return "\n".join(lines)
    except Exception as e:
        print(f"[RapidOCR Run Error] {e}")

    # 3. Pytesseract Fallback
    try:
        import pytesseract
        pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return pytesseract.image_to_string(pil_img, lang="kor+eng").strip()
    except Exception as e:
        print(f"[Pytesseract Fallback Notice] {e}")
        return ""


def extract_text_from_pdf_data(data: bytes) -> str:
    """Extracts text from PDF data, automatically falling back to RapidOCR for scanned pages,
    and supplementing header/footer image OCR when PDF contains images (logos, stamps)."""
    import pdfplumber

    page_texts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for idx, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            # 텍스트 레이어가 50자 이상이면 기본 텍스트 사용 + 헤더/푸터 이미지 보충
            if len(text) >= 50:
                if getattr(page, "images", None) and len(page.images) > 0:
                    try:
                        img = page.to_image(resolution=200).original
                        img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
                        h, w = img_cv.shape[:2]
                        header_ocr = ocr_image(img_cv[: int(h * 0.20), :])
                        footer_ocr = ocr_image(img_cv[int(h * 0.78) :, :])
                        parts = [text]
                        if header_ocr:
                            parts.append(header_ocr)
                        if footer_ocr:
                            parts.append(footer_ocr)
                        page_texts.append("\n".join(parts))
                        continue
                    except Exception as e:
                        print(f"[Header/Footer OCR Warning] {e}")
                        pass
                page_texts.append(text)
            else:
                # 스캔 이미지로 렌더링 후 OCR (scale=2.5 및 pypdfium2 우선 적용)
                try:
                    import pypdfium2 as pdfium
                    pdf_doc = pdfium.PdfDocument(io.BytesIO(data))
                    p_img = pdf_doc[idx].render(scale=2.5).to_pil()
                    img_cv = cv2.cvtColor(np.array(p_img.convert("RGB")), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"[pypdfium2 Render Warning, falling back to pdfplumber] {e}")
                    img = page.to_image(resolution=200).original
                    img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
                ocr_res = ocr_image(img_cv)
                
                # 통장사본 도트 폰트 보조 스캔 (상단 14.5%~19.5% 영역 예금주 라인)
                try:
                    h, w = img_cv.shape[:2]
                    y1, y2 = int(h * 0.145), int(h * 0.195)
                    crop = img_cv[y1:y2, :]
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    dark_y, dark_x = np.where(gray < 150)
                    if len(dark_x) > 100:
                        sub = crop[dark_y.min() : dark_y.max() + 1, dark_x.min() : dark_x.max() + 1]
                        ocr = get_korean_ocr()
                        if ocr and hasattr(ocr, "text_recognizer"):
                            rec_res, _ = ocr.text_recognizer([sub])
                            if rec_res and rec_res[0][0]:
                                band_txt = rec_res[0][0]
                                if "님" in band_txt or any(c in band_txt for c in ("주", "회사", "상호")):
                                    ocr_res = f"{band_txt}\n{ocr_res}"
                except Exception as e:
                    print(f"[Passbook Band Scan Warning] {e}")
                    pass

                page_texts.append(ocr_res)

    return "\n\n".join(page_texts)
