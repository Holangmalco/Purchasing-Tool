"""Local OCR engine module with Korean PP-OCRv3 ONNX support and high-precision fusing."""
from __future__ import annotations

import io
import os
import re
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


def deskew_image(img_cv: np.ndarray, max_angle: float = 30.0) -> np.ndarray:
    """텍스트 행의 기울기(Skew)를 감지하여 수평(0도)으로 자동 회전 보정합니다."""
    if img_cv is None or img_cv.size == 0:
        return img_cv

    h, w = img_cv.shape[:2]
    calc_scale = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
    calc_img = (
        cv2.resize(img_cv, (int(w * calc_scale), int(h * calc_scale)), interpolation=cv2.INTER_AREA)
        if calc_scale < 1.0
        else img_cv
    )

    gray = cv2.cvtColor(calc_img, cv2.COLOR_BGR2GRAY) if len(calc_img.shape) == 3 else calc_img
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    min_line_len = max(int(w * calc_scale * 0.15), 50)
    lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, threshold=100, minLineLength=min_line_len, maxLineGap=20)
    if lines is None or len(lines) == 0:
        return img_cv

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= max_angle:
            angles.append(angle)
        elif abs(angle - 180) <= max_angle:
            angles.append(angle - 180)
        elif abs(angle + 180) <= max_angle:
            angles.append(angle + 180)

    if not angles or len(angles) < 3:
        return img_cv

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return img_cv

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def remove_red_stamp(img_cv: np.ndarray) -> tuple[np.ndarray, bool]:
    """붉은색 인감/직인 도장을 감지 및 투과 완화하여 도장 뒤의 글씨 획을 보존합니다.

    Returns:
        tuple[np.ndarray, bool]: (도장 완화 이미지, 도장 감지 여부)
    """
    if img_cv is None or len(img_cv.shape) != 3 or img_cv.shape[2] != 3:
        return img_cv, False

    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 45, 50])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([168, 45, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    h, w = img_cv.shape[:2]
    total_pixels = h * w
    red_count = np.sum(red_mask > 0)
    red_ratio = red_count / total_pixels

    if 0.0003 <= red_ratio <= 0.20:
        destamped = img_cv.copy()
        # 어두운 검은 글씨 획(True Black text: B, G, R < 85)은 붉은색 마스크에서 제외하여 글씨 보존
        dark_text = (img_cv[:, :, 0] < 85) & (img_cv[:, :, 1] < 85) & (img_cv[:, :, 2] < 85)
        red_ink_only = red_mask & (~dark_text)
        destamped[red_ink_only > 0] = [255, 255, 255]
        return destamped, True

    return img_cv, False


def clean_watermark_and_enhance(img_cv: np.ndarray) -> np.ndarray:
    """국세청 마크 등 배경의 옅은 회색 워터마크를 제거하고 글씨 대비를 향상시킵니다."""
    if img_cv is None or img_cv.size == 0:
        return img_cv

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
    # 옅은 회색(밝기 165 이상)을 255(흰색)로 날리고 0~150 범위를 0~255로 스트레칭
    table = np.array([min(255, int(i * (255 / 165))) if i < 165 else 255 for i in range(256)]).astype('uint8')
    cleaned_gray = cv2.LUT(gray, table)
    if len(img_cv.shape) == 3:
        return cv2.cvtColor(cleaned_gray, cv2.COLOR_GRAY2BGR)
    return cleaned_gray


def ocr_image(img_input: Union[Image.Image, np.ndarray, str, Path]) -> str:
    """Performs high-precision OCR combining Korean model with filtered English/Numeric model."""
    # 1. Convert input to cv2 BGR image
    if isinstance(img_input, (str, Path)):
        img_cv = cv2.imread(str(img_input))
    elif isinstance(img_input, Image.Image):
        img_cv = cv2.cvtColor(np.array(img_input.convert("RGB")), cv2.COLOR_RGB2BGR)
    elif isinstance(img_input, np.ndarray):
        img_cv = img_cv_input = img_input
        if len(img_cv.shape) == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
    else:
        return ""

    if img_cv is None or img_cv.size == 0:
        return ""

    # 1-1. 수평 보정 (Deskew)
    try:
        img_cv = deskew_image(img_cv)
    except Exception:
        pass

    # 1-2. 붉은색 인감/도장 투과 필터 (검은 글씨 획 보존)
    target_img = img_cv
    has_stamp = False
    try:
        destamped_img, has_stamp = remove_red_stamp(img_cv)
        if has_stamp:
            target_img = destamped_img
    except Exception:
        pass

    # 2. RapidOCR Dual Engine (Korean Primary + Clean Numeric Fallback)
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr_kor = get_korean_ocr()
        ocr_def = RapidOCR(use_angle_cls=False)

        h, w = target_img.shape[:2]
        # 과도하게 큰 이미지만 비율 조정 (해상도 3600px까지는 100% 원본 해상도 보존하여 작은 글씨 및 도트 폰트 보존)
        if max(h, w) > 3600:
            scale = 3200 / max(h, w)
            target_img = cv2.resize(target_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        elif max(h, w) < 1200:
            scale = min(2.5, 1800 / max(h, w))
            target_img = cv2.resize(target_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        res_k, _ = ocr_kor(target_img) if ocr_kor is not None else (None, None)
        res_d, _ = ocr_def(target_img) if ocr_def is not None else (None, None)

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
                if dist < max(hk, 25) and dist < best_dist:
                    best_dist = dist
                    best_d = rd
                    best_idx = idx

            chosen_text = tk
            chosen_box = bk
            if best_d is not None and best_dist < max(hk, 25):
                used_d.add(best_idx)
                td = best_d[1].strip()
                cd = float(best_d[2])

                # [중요] 기본 모델(ch)의 중국어 한자 오염 원천 차단
                has_cjk_d = bool(re.search(r'[\u4e00-\u9fff]', td))
                if has_cjk_d:
                    # 한자가 섞인 td는 폐기하고 한국어 모델 tk 유지
                    pass
                else:
                    # 1. 사업자번호(10자리), 계좌번호, 전화번호, 금액 등 정형화된 숫자/코드 패턴은 기본 모델 td가 압도적으로 정확
                    is_biz_d = bool(re.search(r'^\d{3}-\d{2}-\d{5}$', td) or re.search(r'\b\d{3}-\d{2}-\d{5}\b', td))
                    is_phone_d = bool(re.search(r'\b(?:010|02|031|032|033|041|042|043|051|052|053|054|055|061|062|063|064|070)-\d{3,4}-\d{4}\b', td))
                    is_acc_d = bool(re.search(r'\b\d{3,6}-\d{2,6}-\d{3,8}\b', td))
                    is_clean_number = bool(re.search(r'^\d{1,3}(?:,\d{3})+$', td) or re.search(r'^\d{4,12}$', td))
                    is_email_d = bool(re.search(r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}', td))

                    brand_words = ('imac', 'macbook', 'ipad', 'applecare', 'pencil', 'keyboard', 'retina', 'nano-texture', 'samsung')
                    has_brand_d = any(bw in td.lower() for bw in brand_words)
                    has_brand_k = any(bw in tk.lower() for bw in brand_words)

                    if (is_biz_d or is_phone_d or is_acc_d or is_email_d) and cd >= 0.60:
                        chosen_text = td
                    elif is_clean_number and cd >= 0.70:
                        chosen_text = td
                    elif has_brand_d and not has_brand_k:
                        if re.search(r'[\uac00-\ud7a3]', tk):
                            chosen_text = f'{tk} {td}'
                        else:
                            chosen_text = td
                    elif bool(re.search(r'\d{3,}', td)) and not bool(re.search(r'\d{3,}', tk)) and cd >= 0.70:
                        chosen_text = td
                    elif bool(re.search(r'\d{3,}', td)) and bool(re.search(r'\d{3,}', tk)):
                        if (',' in td or '-' in td) and (',' not in tk and '-' not in tk):
                            chosen_text = td
                        elif cd > ck + 0.15:
                            chosen_text = td

            items.append((chosen_box, chosen_text))

        # 기본 모델에서 한국어 모델이 감지하지 못한 독립적인 박스 병합
        for idx, rd in enumerate(res_d or []):
            if idx not in used_d:
                cd = float(rd[2])
                td = rd[1].strip()
                # [필터링] 중국어 한자 배제 및 영문/숫자/기호 패턴만 수용 (배경 워터마크 오염 방지)
                if not re.search(r'[\u4e00-\u9fff]', td):
                    if re.search(r'^[0-9a-zA-Z\s.,\-:()/#@%*+~]+$', td) and (cd >= 0.65 or (cd >= 0.50 and re.search(r'\d{3,}', td))):
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

    return ""


def extract_text_from_pdf_data(data: bytes) -> str:
    """Extracts text from PDF data with 300 DPI rendering, watermark reduction,
    and automatic full-page/header/footer OCR supplementation."""
    import pdfplumber

    page_texts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for idx, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            # 폰트 CID 글리프 깨짐 또는 유니코드 외계어 깨짐 감지 시 즉시 로컬 OCR로 전환
            is_corrupted = text.count("(cid:") >= 3 or ("î" in text and "ð" in text)

            # 텍스트 레이어가 30자 이상이고 깨지지 않았으면 기본 텍스트 사용 + 헤더/푸터/도장 이미지 보충
            if len(text) >= 30 and not is_corrupted:
                if getattr(page, "images", None) and len(page.images) > 0:
                    try:
                        img = page.to_image(resolution=250).original
                        img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
                        h, w = img_cv.shape[:2]
                        header_ocr = ocr_image(img_cv[: int(h * 0.20), :])
                        footer_ocr = ocr_image(img_cv[int(h * 0.78) :, :])
                        parts = [text]
                        if header_ocr:
                            parts.append(header_ocr)
                        if footer_ocr:
                            parts.append(footer_ocr)

                        # 본문 영역(18%~80%)에 포함된 표/산출내역 이미지 보충 스캔
                        p_h = float(getattr(page, "height", 1000) or 1000)
                        has_body_image = False
                        for p_img in getattr(page, "images", []):
                            img_top = float(p_img.get("top", 0) or 0)
                            img_h = float(p_img.get("height", 0) or 0)
                            if (img_top > p_h * 0.15 and img_top < p_h * 0.85) or (img_h > p_h * 0.20):
                                has_body_image = True
                                break

                        if has_body_image:
                            body_ocr = ocr_image(img_cv[int(h * 0.18) : int(h * 0.80), :])
                            if body_ocr:
                                parts.append(body_ocr)

                        page_texts.append("\n".join(parts))
                        continue
                    except Exception as e:
                        print(f"[Header/Footer/Body OCR Warning] {e}")
                        pass
                page_texts.append(text)
            else:
                # 스캔 이미지로 렌더링 후 OCR (scale=3.0 약 250~300 DPI 적용)
                try:
                    import pypdfium2 as pdfium
                    pdf_doc = pdfium.PdfDocument(io.BytesIO(data))
                    p_img = pdf_doc[idx].render(scale=3.0).to_pil()
                    img_cv = cv2.cvtColor(np.array(p_img.convert("RGB")), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"[pypdfium2 Render Warning, falling back to pdfplumber] {e}")
                    img = page.to_image(resolution=250).original
                    img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

                # 1. 원본 이미지 기반 OCR
                ocr_res = ocr_image(img_cv)

                # 2. 사업자등록증/통장사본 등 워터마크 또는 연한 텍스트 보강
                # (텍스트에 등록번호나 계좌번호가 누락된 경우 워터마크 제거 버전으로 추가 보강)
                if not re.search(r'\d{3}-\d{2}-\d{5}', ocr_res) or "사업자" in ocr_res or "등록증" in ocr_res:
                    try:
                        enhanced_cv = clean_watermark_and_enhance(img_cv)
                        enh_res = ocr_image(enhanced_cv)
                        if enh_res and len(enh_res) > 20:
                            # 향상된 버전에서 등록번호나 상호가 더 잘 잡혔으면 병합
                            m_biz_enh = re.search(r'\d{3}-\d{2}-\d{5}', enh_res)
                            m_biz_org = re.search(r'\d{3}-\d{2}-\d{5}', ocr_res)
                            if m_biz_enh and not m_biz_org:
                                ocr_res = f"{enh_res}\n{ocr_res}"
                    except Exception as e:
                        print(f"[Watermark clean scan warning] {e}")

                # 3. 통장사본 상단 예금주 밴드 정밀 보조 스캔
                try:
                    h, w = img_cv.shape[:2]
                    # 통장 상단 8%~25% 영역
                    band_crop = img_cv[int(h * 0.08) : int(h * 0.25), : int(w * 0.75)]
                    band_txt = ocr_image(band_crop)
                    if band_txt and any(k in band_txt for k in ("님", "공조", "상호", "회사", "성명")):
                        ocr_res = f"{band_txt}\n{ocr_res}"
                except Exception as e:
                    print(f"[Passbook Band Scan Warning] {e}")

                page_texts.append(ocr_res)

    return "\n\n".join(page_texts)
