"""Streamlit UI for local OCR purchase verification."""
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pdfplumber
import pytesseract
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes

from purchase_verifier.core import (
    DocumentType, PurchaseDocument, classify_document, extract_document,
    find_lowest_quote, mask_account_number, verify_documents,
)

st.set_page_config(page_title="구매 문서 검증", layout="wide")

def preprocess_scan_image(pil_img: Image.Image) -> np.ndarray:
    image = np.array(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    red = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    red |= cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    image[red > 0] = [255, 255, 255]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)

def extract_local_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()
    if suffix == ".xlsx":
        frame = pd.read_excel(io.BytesIO(data))
        return frame.to_string(index=False)
    if suffix == ".pdf":
        text_parts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        text = "\n".join(text_parts)
        if len(text.strip()) >= 50:
            return text
        pages = convert_from_bytes(data)
        return "\n".join(pytesseract.image_to_string(preprocess_scan_image(page), lang="kor+eng", config="--psm 6") for page in pages)
    if suffix in {".png", ".jpg", ".jpeg"}:
        image = Image.open(io.BytesIO(data))
        return pytesseract.image_to_string(preprocess_scan_image(image), lang="kor+eng", config="--psm 6")
    return ""

def status_label(status: str) -> str:
    return status

def render_document(document: PurchaseDocument) -> None:
    st.subheader(document.filename)
    st.caption(f"분류: {document.document_type.value}")
    cols = st.columns(4)
    cols[0].metric("업체명", document.vendor_name.raw or "미검출")
    cols[1].metric("사업자번호", document.business_number.raw or "미검출")
    cols[2].metric("합계", f"{document.total.value:,}원" if document.total.value is not None else "미확정")
    account = mask_account_number(document.account_number.raw) if document.account_number.raw else "미검출"
    cols[3].metric("계좌", account)
    if document.total.evidence.inferred:
        st.warning("합계는 공급가액과 부가세를 산술 추론한 값입니다.")
    if document.warnings:
        for warning in document.warnings:
            st.warning(warning)
    if document.items:
        st.markdown("**품목표**")
        st.dataframe(pd.DataFrame([{"품명": i.name, "수량": i.quantity, "단가": i.unit_price, "금액": i.amount} for i in document.items]), use_container_width=True, hide_index=True)
    with st.expander("원문 보기"):
        st.text(document.raw_text)

def main() -> None:
    st.title("구매 문서 검증")
    purchase_type = st.radio("구매 유형", ["물품", "용역", "공사"], horizontal=True)
    files = st.file_uploader("PDF, PNG, JPG, XLSX 업로드", type=["pdf", "png", "jpg", "jpeg", "xlsx"], accept_multiple_files=True)
    if not files:
        st.info("문서를 업로드하면 로컬 OCR과 규칙 기반 분석을 시작합니다.")
        return
    if st.button("로컬 분석 실행", type="primary"):
        documents = []
        for uploaded in files:
            try:
                raw_text = extract_local_text(uploaded)
                documents.append(extract_document(raw_text, uploaded.name))
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f"{uploaded.name}: 분석 실패 - {exc}")
        st.session_state["purchase_documents"] = documents
    documents = st.session_state.get("purchase_documents", [])
    if not documents:
        return
    st.markdown("## 문서별 요약")
    for document in documents:
        render_document(document)
    st.markdown("## 최저가 후보")
    lowest = find_lowest_quote(documents)
    if lowest.candidate_filename:
        (st.success if lowest.confirmed else st.warning)(f"{lowest.candidate_filename}: {lowest.amount:,}원 - {lowest.reason}")
    else:
        st.warning(lowest.reason)
    st.markdown("## 문서 교차검증")
    for result in verify_documents(documents, purchase_type):
        st.write(f"`{status_label(result.status.value)}` **{result.name}** — {result.detail}")

main()
