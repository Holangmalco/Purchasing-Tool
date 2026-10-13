"""Tests for OCR enhancements: Deskewing, Red Stamp Removal, and PDF Extraction."""
import numpy as np
import cv2
from purchase_verifier.ocr import deskew_image, remove_red_stamp, ocr_image


def test_remove_red_stamp_detects_and_removes():
    # 흰색 배경에 빨간색 도장 원 생성
    h, w = 200, 200
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # 중앙에 빨간색 원(인감도장 모사)
    cv2.circle(img, (100, 100), 25, (0, 0, 230), -1)  # BGR에서 R=230

    destamped, has_stamp = remove_red_stamp(img)
    assert has_stamp is True
    # 빨간색이었던 중심부 픽셀이 흰색(255, 255, 255)으로 치환되었는지 확인
    assert np.all(destamped[100, 100] == [255, 255, 255])


def test_remove_red_stamp_preserves_clean_image():
    # 도장이 없는 일반 검은 텍스트 문서
    h, w = 100, 100
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.putText(img, "TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    destamped, has_stamp = remove_red_stamp(img)
    assert has_stamp is False
    assert np.array_equal(destamped, img)


def test_deskew_image_safe_on_flat_and_empty():
    # 비어있는 이미지 또는 평평한 이미지 안전 검증
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert deskew_image(empty).size == 0

    flat = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = deskew_image(flat)
    assert res.shape == (100, 100, 3)
