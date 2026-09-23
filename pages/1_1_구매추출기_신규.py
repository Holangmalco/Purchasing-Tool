"""Integrated Purchase Verification Dashboard (ERP Text + Local OCR)."""
import datetime
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pdfplumber
import pytesseract
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes

# Windows 기본 Tesseract 경로 자동 탐색 등록
for t_path in [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]:
    if os.path.exists(t_path):
        pytesseract.pytesseract.tesseract_cmd = t_path
        break

from purchase_verifier.core import (
    CheckResult, DocumentType, PurchaseDocument, VerificationStatus,
    classify_document, detect_ocr_anomaly, extract_document,
    find_lowest_quote, mask_account_number, verify_documents,
)
from purchase_verifier.vendor_cache import (
    auto_cache_verified_documents, get_vendor_by_biz_num, save_vendor,
)
from purchase_verifier.erp_parser import (
    build_smart_matrix, calculate_business_days, check_nts_business,
    extract_base_info, generate_opinion, get_accurate_total, parse_goods_erp,
    parse_service_erp, save_to_cache, smart_item_matching,
)

st.set_page_config(page_title="구매추출기 통합 대시보드", page_icon="💰", layout="wide")

SESSION_REQ_KEY = "req_analysis_done"
SESSION_MANUAL_JSON_KEY = "manual_json_data"
SESSION_DOCS_KEY = "purchase_documents"
SESSION_AUTO_CACHED_KEY = "has_auto_cached_current_analysis"


# ---------------------------------------------------------
# OCR 및 문서 텍스트 추출 엔진
from purchase_verifier.ocr import ocr_image


# ---------------------------------------------------------
# OCR 및 문서 텍스트 추출 엔진
# ---------------------------------------------------------
def ocr_image_robust(pil_img: Image.Image) -> str:
    """Multi-pass OCR for scanned invoices, bankbooks, and business licenses."""
    # 1. RapidOCR Korean ONNX 엔진 우선 실행 (고성능 로컬 모델)
    try:
        rapid_res = ocr_image(pil_img)
        if rapid_res and len(rapid_res.strip()) >= 30:
            return rapid_res
    except Exception:
        pass

    try:
        img_arr = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

        # Pass 1: CLAHE 대비 강화 + 자동 레이아웃(PSM 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        text1 = pytesseract.image_to_string(enhanced, lang="kor+eng", config="--psm 3").strip()
        if len(text1) >= 30:
            return text1

        # Pass 2: 가우시안 블러 + 적응형 이진화 + 단일 블록(PSM 6)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        text2 = pytesseract.image_to_string(thresh, lang="kor+eng", config="--psm 6").strip()
        if len(text2) >= 30:
            return text2

        # Pass 3: Otsu 이진화 + Sparse 텍스트(PSM 11)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text3 = pytesseract.image_to_string(otsu, lang="kor+eng", config="--psm 11").strip()

        return max([text1, text2, text3], key=len)
    except pytesseract.TesseractNotFoundError:
        return "[스캔 이미지 감지: 로컬 OCR(Tesseract) 미설치로 텍스트 자동 판독 불가]"
    except Exception as e:
        return f"[스캔 이미지 OCR 실패: {e}]"


def extract_local_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()

    # 1. 엑셀 파일 (.xlsx)
    if suffix == ".xlsx":
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
            parts = [f"[시트: {s_name}]\n{df.to_string(index=False)}" for s_name, df in sheets.items()]
            return "\n\n".join(parts)
        except Exception:
            frame = pd.read_excel(io.BytesIO(data))
            return frame.to_string(index=False)

    # 2. PDF 파일 (.pdf)
    if suffix == ".pdf":
        page_texts = []
        pages_needing_ocr = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for idx, page in enumerate(pdf.pages):
                p_text = (page.extract_text() or "").strip()
                if len(p_text) >= 30:
                    page_texts.append((idx, p_text))
                else:
                    pages_needing_ocr.append(idx)

        if pages_needing_ocr:
            # poppler 없이 자체 구동되는 pypdfium2 우선 사용
            try:
                import pypdfium2 as pdfium
                pdf_doc = pdfium.PdfDocument(io.BytesIO(data))
                for idx in pages_needing_ocr:
                    pil_img = pdf_doc[idx].render(scale=2).to_pil()
                    ocr_res = ocr_image_robust(pil_img)
                    page_texts.append((idx, ocr_res))
            except Exception:
                # pdf2image fallback
                try:
                    for idx in pages_needing_ocr:
                        page_imgs = convert_from_bytes(data, first_page=idx + 1, last_page=idx + 1, dpi=300)
                        ocr_result = "\n".join(ocr_image_robust(img) for img in page_imgs)
                        page_texts.append((idx, ocr_result))
                except Exception as e:
                    for idx in pages_needing_ocr:
                        page_texts.append((idx, f"[스캔 이미지 감지: 로컬 OCR(Tesseract) 미설치 ({e})]"))

        page_texts.sort(key=lambda x: x[0])
        return "\n\n".join(text for _, text in page_texts)

    # 3. 이미지 파일 (.png, .jpg, .jpeg)
    if suffix in {".png", ".jpg", ".jpeg"}:
        try:
            image = Image.open(io.BytesIO(data))
            return ocr_image_robust(image)
        except Exception as e:
            return f"[이미지 읽기 실패: {e}]"

    return ""


def status_label(status: str) -> str:
    mapping = {
        "PASS": "✅ 적합 (PASS)",
        "REVIEW": "⚠️ 확인 필요 (REVIEW)",
        "CONFLICT": "❌ 불일치 (CONFLICT)",
        "MISSING": "❓ 누락 (MISSING)",
        "BLOCKED": "🚫 반려 대상 (BLOCKED)",
        "WARNING": "⚠️ 주의 (WARNING)",
    }
    return mapping.get(status, status)


def display_status_value(label: str, val: Any) -> None:
    st.write(f"**{label}**")
    if not val or val == "미검출":
        st.info("🚨 입력 필요")
    elif "확인 불가" in str(val):
        st.warning("⚠️ 자동 추출 실패 (직접 입력)")
    else:
        st.code(str(val), language=None)


# ---------------------------------------------------------
# 메인 대시보드 애플리케이션
# ---------------------------------------------------------
def main() -> None:
    st.title("💰 구매추출기 통합 업무 대시보드")
    st.caption("ERP 구매요청 텍스트와 첨부 서류(PDF/이미지/엑셀)를 원클릭으로 통합 검증하고, 필요한 정보를 손쉽게 복사합니다.")

    # 1. 구매/계약 기본 정보 선택
    st.markdown("#### 📂 1. 구매/계약 기본 정보 선택")
    c_type1, c_type2 = st.columns(2)
    with c_type1:
        req_type = st.radio("구매 유형", ["📦 물품", "🛠️ 용역", "🏗️ 공사"], horizontal=True, key="p_req_type")
    with c_type2:
        con_type = st.radio("계약 방식", ["비교견적", "수의계약"], horizontal=True, key="p_con_type")

    st.write("### 📥 [STEP 1] 정보 입력 및 서류 업로드")
    col_in1, col_in2 = st.columns(2)

    with col_in1:
        with st.container(border=True):
            st.markdown("#### 📝 2. ERP 텍스트 복사")
            raw_req = st.text_area("1️⃣ 구매요청서 전체", height=120, key="erp_raw_req", placeholder="구매요구번호, 과제번호, 연구책임자정보 등이 포함된 본문 전체")
            raw_fin = st.text_area("2️⃣ 재원내역 행", height=60, key="erp_raw_fin", placeholder="재원내역 테이블 행 (공급가, 부가세 등)")
            raw_itm = st.text_area("3️⃣ 물품/용역내역 행", height=80, key="erp_raw_itm", placeholder="물품 및 용역 내역 행 (품명, 규격, 수량, 금액 등)")

    with col_in2:
        with st.container(border=True):
            st.markdown("#### 📎 3. 로컬 서류 업로드")
            st.caption("견적서, 비교견적서, 사업자등록증, 통장사본, 공사확인서 (PDF, PNG, JPG, XLSX)")
            files = st.file_uploader(
                "파일 업로드",
                type=["pdf", "png", "jpg", "jpeg", "xlsx"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="uploaded_docs_input",
            )
            st.info(
                "**📋 필수 서류 검증 체크리스트**\n\n"
                "- ✅ **명의 확인:** 모든 견적서의 수신자가 '세종대학교 산학협력단'인지 전수 확인\n"
                "- ✅ **유효 기간:** 만료일까지 **5영업일(평일) 이상** 잔여 여부 자동 계산\n"
                "- ✅ **사업자 일치:** 최저가 견적서와 사업자등록증 정보 일치 대조\n"
                "- ✅ **계좌 검증:** 상호 누락된 대표자 개인명의 단독 통장 사본 자동 반려\n"
                "- ✅ **공사 서류:** '공사' 건일 경우 시설공사/장비설치 확인서 첨부 유무 확인\n"
            )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_analysis = st.button("🚀 전체 데이터 통합 분석 시작", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🔄 전체 초기화", use_container_width=True):
            for k in ["erp_raw_req", "erp_raw_fin", "erp_raw_itm", SESSION_REQ_KEY, SESSION_DOCS_KEY, SESSION_MANUAL_JSON_KEY, SESSION_AUTO_CACHED_KEY]:
                st.session_state.pop(k, None)
            st.rerun()

    # 분석 실행 트리거
    if start_analysis:
        if not raw_req and not files:
            st.warning("📌 ERP 텍스트를 입력하거나 검증할 서류를 1개 이상 업로드해주세요.")
            return

        documents: list[PurchaseDocument] = []
        if files:
            with st.spinner("📄 로컬 OCR 엔진으로 서류를 분석 중입니다..."):
                for uploaded in files:
                    try:
                        raw_text = extract_local_text(uploaded)
                        doc = extract_document(raw_text, uploaded.name)
                        documents.append(doc)
                    except Exception as exc:
                        st.error(f"'{uploaded.name}' 분석 실패: {exc}")

        st.session_state[SESSION_DOCS_KEY] = documents
        st.session_state[SESSION_REQ_KEY] = True
        st.session_state[SESSION_AUTO_CACHED_KEY] = False

    # ---------------------------------------------------------
    # [STEP 2] 분석 결과 대시보드 표출
    # ---------------------------------------------------------
    if st.session_state.get(SESSION_REQ_KEY):
        st.divider()
        st.write("### 📤 [STEP 2] 통합 분석 대시보드")

        documents: list[PurchaseDocument] = st.session_state.get(SESSION_DOCS_KEY, [])
        manual_ai = st.session_state.get(SESSION_MANUAL_JSON_KEY, {})

        # ---------------------------------------------------------
        # OCR 품질 이상 감지 및 제미나이 3.8 Flash 우회 가이드 (사용자 요구사항 2번)
        # ---------------------------------------------------------
        ocr_anomalies = []
        for doc in documents:
            is_anomaly, reason = detect_ocr_anomaly(doc.raw_text)
            if is_anomaly:
                ocr_anomalies.append((doc.filename, reason))

        if ocr_anomalies and not manual_ai:
            st.warning("⚠️ **[로컬 OCR 글자 깨짐/품질 저하 감지]**\n일부 문서의 스캔 화질이 낮아 글자나 숫자가 부정확할 수 있습니다. **제미나이 3.8 Flash** 등 외부 AI로 정밀 분석하는 것을 적극 권장합니다.")
            
            today_str = datetime.date.today().strftime("%Y-%m-%d")
            extra_const = "🚨 이 건은 '공사' 건입니다. 첨부 서류 중 '시설공사 확인서' 또는 '장비설치 확인서'가 반드시 포함되어야 하므로 이를 꼼꼼히 확인하세요.\n" if "공사" in req_type else ""

            fallback_prompt = (
                f"🚨 [기준일] 오늘 날짜는 {today_str}입니다. 만료일자 계산 시 참고하세요.\n"
                f"{extra_const}"
                "분석 후 반드시 아래 형태의 JSON 형식으로만 응답해. 절대 마크다운(```json)이나 다른 텍스트를 넣지 마.\n"
                "{\n"
                '  "비교견적리스트": [ {"업체명": "A업체", "사업자번호": "123-45-67890", "제출금액": 10000}, {"업체명": "B업체", "사업자번호": "987-65-43210", "제출금액": 12000} ],\n'
                '  "최저가판단과정": "위 비교견적리스트를 바탕으로 가장 금액이 낮은 업체를 최종 선정하는 과정을 짧게 작성",\n'
                '  "수신자명확인": "첨부된 모든 견적서(비교견적서 포함)의 수신자가 모두 \'세종대학교 산학협력단\' 명의가 맞으면 \'예\', 하나라도 누락되거나 다르면 \'아니오(사유)\' (단, 애플/삼성 등 제조사 공식스토어 장바구니 캡처본은 공식몰 특성상 수신자 표기 생략 인정)",\n'
                '  "만료일자": "유효기간 만료일자를 YYYY-MM-DD 형식으로 기재",\n'
                '  "동일규격확인": "복수 견적서 간 동일 물품이면 \'예\', 다르면 \'아니오(사유)\', 1장뿐이면 \'단일견적\'",\n'
                '  "사업자일치여부": "최저가(최종 선정) 견적서와 사업자등록증 상의 정보가 일치하면 \'예\', 다르면 \'아니오(사유)\'",\n'
                '  "업체명": "첨부된 사업자등록증 상의 \'법인명(단체명)\' 또는 \'상호\'를 최우선으로 추출 (사업자등록증이 없는 경우 견적서의 상호명 추출)",\n'
                '  "사업자번호": "최저가 업체의 사업자등록번호 (반드시 000-00-00000 형식)",\n'
                '  "대표자명": "사업자등록증 상의 대표자 성명",\n'
                '  "담당자": "최저가 업체의 견적 담당자명",\n'
                '  "연락처": "최저가 업체 담당자의 010 전화번호 (없으면 대표번호)",\n'
                '  "이메일": "최저가 업체의 이메일",\n'
                '  "예금주": "통장사본의 예금주명",\n'
                '  "계좌번호": "통장사본의 계좌번호",\n'
                '  "총금액": "최종 선정된 업체의 견적 총액 (반드시 부가가치세/VAT 포함된 최종 합계 금액을 숫자로만 기재)",\n'
                '  "공사확인서유무": "첨부 파일 중 \'시설공사\' 또는 \'장비설치 확인서\' 문서가 있으면 \'예\', 없으면 \'아니오\', 비슷한 문서가 있으면 \'확인요망\'",\n'
                '  "품목리스트": [ {"품명": "품명 텍스트", "수량": 1, "단가": 1000, "금액": 1000} ]\n'
                "}"
            )

            with st.expander("💡 [외부 AI 우회 가이드] 제미나이 웹용 프롬프트 복사 & JSON 적용 폼", expanded=True):
                st.info("1. 아래 프롬프트를 복사하여 제미나이(Gemini 3.8 Flash 등) 웹페이지에 서류 파일과 함께 붙여넣고 전송합니다.\n2. 결과 JSON 텍스트를 복사하여 아래에 붙여넣고 [수동 데이터 시스템 적용]을 누르면 대시보드에 즉시 반영됩니다.")
                st.code(fallback_prompt, language="text")

                manual_json_text = st.text_area("제미나이 결과 JSON 붙여넣기", height=120, placeholder="{\n  \"업체명\": \"...\",\n  \"사업자번호\": \"...\"\n}")
                c_m1, c_m2 = st.columns(2)
                with c_m1:
                    if st.button("🚀 수동 데이터 시스템 적용", type="primary", use_container_width=True):
                        try:
                            clean_json = re.search(r"\{.*\}", manual_json_text, re.DOTALL)
                            if clean_json:
                                parsed = json.loads(clean_json.group())
                                st.session_state[SESSION_MANUAL_JSON_KEY] = parsed
                                st.success("✅ 제미나이 분석 결과가 완벽하게 적용되었습니다!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("🚨 중괄호 { } 형태의 JSON 문자열을 정확히 붙여넣어주세요.")
                        except Exception as err:
                            st.error(f"🚨 JSON 파싱 오류: {err}")
                with c_m2:
                    if st.button("계속 로컬 결과로 진행", use_container_width=True):
                        st.session_state[SESSION_MANUAL_JSON_KEY] = {}
                        st.rerun()

        # ---------------------------------------------------------
        # ERP 텍스트 파싱
        # ---------------------------------------------------------
        base_data = extract_base_info(raw_req)
        total_erp_amt = get_accurate_total(raw_req, "합계액")

        if "물품" in req_type:
            item_list, is_valid_erp = parse_goods_erp(raw_itm)
        else:
            item_list, is_valid_erp = parse_service_erp(raw_itm, force_category="용역" if "용역" in req_type else "공사, 용역")

        # ---------------------------------------------------------
        # 연차연구기간 종료일 모니터링 배너 (사용자 요구사항 4번)
        # ---------------------------------------------------------
        prj_end_str = base_data.get("prj_end", "")
        if prj_end_str:
            try:
                prj_end_date = datetime.datetime.strptime(prj_end_str, "%Y-%m-%d").date()
                today = datetime.date.today()
                days_left = (prj_end_date - today).days

                if 0 <= days_left <= 30:
                    st.warning(f"⏰ **[과제 종료 임박]** 과제 연차 종료일({prj_end_str})이 **{days_left}일** 남았습니다! 기한 내 대금 청구가 완료되도록 주의하세요.")
                elif days_left < 0:
                    st.error(f"🚨 **[과제 종료됨]** 과제 연차 종료일({prj_end_str})이 이미 **{abs(days_left)}일 지났습니다!** 과제 이월 또는 취소 여부를 즉시 확인하세요.")
                else:
                    st.info(f"📅 **[과제 연차 종료일]** {prj_end_str} (잔여 {days_left}일)")
            except Exception:
                pass

        # ---------------------------------------------------------
        # 견적서 선택 및 최저가 분석
        # ---------------------------------------------------------
        quotes = [d for d in documents if d.document_type in (DocumentType.QUOTE, DocumentType.ONLINE_STORE_CART)]
        lowest = find_lowest_quote(documents)
        selected_filename = None

        if quotes:
            col_q1, col_q2 = st.columns([2, 1])
            with col_q1:
                quote_opts = [q.filename for q in quotes]
                def format_quote_opt(fname: str) -> str:
                    d = next((x for x in quotes if x.filename == fname), None)
                    if d and d.document_type == DocumentType.ONLINE_STORE_CART:
                        return f"🛒 {fname} (공식스토어 장바구니)"
                    return f"📄 {fname} (견적서)"

                def_idx = quote_opts.index(lowest.candidate_filename) if lowest.candidate_filename in quote_opts else 0
                selected_filename = st.selectbox(
                    "선택(낙찰/계약 기준) 견적서",
                    quote_opts,
                    format_func=format_quote_opt,
                    index=def_idx,
                    help="사업자등록증, 통장사본 등 지원 서류와 대조할 기준 견적서입니다 (기본값: 최저가 견적).",
                )
            with col_q2:
                if lowest.candidate_filename:
                    low_doc = next((d for d in documents if d.filename == lowest.candidate_filename), None)
                    is_cart = low_doc and low_doc.document_type == DocumentType.ONLINE_STORE_CART
                    if is_cart:
                        badge = "🛒 공식스토어 최저가 확인" if lowest.confirmed else "🛒 공식스토어 최저가 후보 (품목 확인 필요)"
                    else:
                        badge = "🏆 최저가 견적 확인" if lowest.confirmed else "💡 최저가 후보 (품목 확인 필요)"
                    amt_str = f" ({lowest.amount:,}원)" if lowest.amount is not None else " (금액 미확정)"
                    st.info(f"{badge}\n\n**{lowest.candidate_filename}**{amt_str}")

        # ---------------------------------------------------------
        # 서류 교차 검증 및 국세청 조회
        # ---------------------------------------------------------
        results = verify_documents(documents, purchase_type="공사" if "공사" in req_type else "물품", selected_filename=selected_filename)
        auto_reject_reasons = []

        # 검증 결과에서 반려(BLOCKED) 항목 수집
        for r in results:
            if r.status == VerificationStatus.BLOCKED:
                auto_reject_reasons.append(r.detail.split("\n")[0].replace("반려 대상: ", ""))

        # 국세청 사업자 실시간 휴폐업 조회
        target_doc = next((d for d in documents if d.filename == selected_filename), None) or (quotes[0] if quotes else None)
        biz_numbers_to_check = []

        if target_doc and target_doc.business_number.raw:
            biz_numbers_to_check.append((target_doc.vendor_name.raw or "최종선정업체", target_doc.business_number.raw))

        for q in quotes:
            if q.business_number.raw and not any(b[1] == q.business_number.raw for b in biz_numbers_to_check):
                biz_numbers_to_check.append((q.vendor_name.raw or q.filename, q.business_number.raw))

        nts_report_items = []
        if biz_numbers_to_check:
            with st.spinner(f"🔍 국세청에 사업자 휴·폐업 상태를 실시간 조회 중입니다... ({len(biz_numbers_to_check)}건)"):
                for v_name, b_no in biz_numbers_to_check:
                    stt = check_nts_business(b_no)
                    nts_report_items.append(f"{v_name}: {stt}")
                    if "휴업자" in stt:
                        auto_reject_reasons.append(f"국세청 조회 결과 '휴업' 상태 사업자 포함 - {v_name} ({b_no})")
                    elif "폐업자" in stt:
                        auto_reject_reasons.append(f"국세청 조회 결과 '폐업' 상태 사업자 포함 - {v_name} ({b_no})")

        nts_summary_str = " / ".join(nts_report_items) if nts_report_items else "⚠️ 사업자번호 미검출 (조회 불가)"

        # ---------------------------------------------------------
        # 🚦 서류 검증 종합 신호등 리포트 요약
        # ---------------------------------------------------------
        st.markdown("#### 🚦 서류 검증 종합 신호등 리포트")
        
        # 주요 상태 요약 라인 구성
        rec_status = next((r.detail for r in results if r.name == "수신자 산학협력단"), "미확인")
        exp_status = next((r.detail for r in results if r.name == "견적서 유효기간"), "미확인")
        same_items_status = next((r.detail for r in results if r.name == "품목 구성"), "단일견적 또는 미확인")
        license_match_status = next((r.detail for r in results if "사업자등록증 업체명" in r.name or "사업자등록증 사업자번호" in r.name), "대조 생략(서류 없음)")
        bank_match_status = next((r.detail for r in results if "통장사본 예금주" in r.name), "대조 생략(서류 없음)")
        const_doc_status = next((r.detail for r in results if r.name == "공사 확인서"), "해당없음(물품/용역)")

        rep_lines = [
            f"- **산단명의:** {rec_status}",
            f"- **국세청조회:** {nts_summary_str}",
            f"- **유효기간:** {exp_status}",
            f"- **동일규격:** {same_items_status}",
            f"- **사업자일치:** {license_match_status}",
            f"- **예금주검증:** {bank_match_status}",
            f"- **재원구분:** {'⚠️ 간접비 집행 건 (확인필요)' if '간접비' in raw_fin else ('일반 연구비(직접비)' if raw_fin.strip() else '재원 미입력')}",
        ]
        if "공사" in req_type:
            rep_lines.append(f"- **공사서류:** {const_doc_status}")

        report_summary_text = "\n".join(rep_lines)

        # ---------------------------------------------------------
        # 🚫 [반려 사유 자동 생성] 원클릭 복사창
        # ---------------------------------------------------------
        if auto_reject_reasons:
            st.warning(f"**⚠️ 서류 검증 경고 및 반려 사유 발생**\n\n{report_summary_text}")
            rej_text = "안녕하세요 교수님, 산학협력단 구매담당자입니다.\n요청해주신 건에 대해 아래 사유로 보완 및 재발행이 필요하여 안내드립니다.\n\n"
            for idx, r_reason in enumerate(auto_reject_reasons, start=1):
                rej_text += f"{idx}. {r_reason}\n"
            rej_text += "\n서류 보완 후 재요청 부탁드립니다. 감사합니다."
            st.code(rej_text, language="text")
        else:
            st.success(f"**🎉 모든 서류 교차 검증 통과 (적합)**\n\n{report_summary_text}")

        # ---------------------------------------------------------
        # 📋 상세 교차 검증 테이블
        # ---------------------------------------------------------
        with st.expander("🔍 상세 교차 검증 항목별 판정 내역 보기", expanded=False):
            v_rows = []
            for r in results:
                icon_lbl = status_label(r.status.value)
                v_rows.append({
                    "검증 항목": r.name,
                    "판정 결과": icon_lbl,
                    "상세 검토 내용 및 실제 추출값": r.detail,
                })
            st.dataframe(pd.DataFrame(v_rows), use_container_width=True, hide_index=True)

        # ---------------------------------------------------------
        # 📊 스마트 견적 대조표
        # ---------------------------------------------------------
        with st.expander("📊 스마트 견적 대조표 (품목 및 수량·단가 매칭)", expanded=False):
            smart_df = build_smart_matrix(documents, item_list)
            st.dataframe(smart_df, use_container_width=True, hide_index=True)

        # ---------------------------------------------------------
        # 거래처 마스터 DB 자동 캐싱 및 보완 안내 (분석 1회당 1번만 갱신)
        # ---------------------------------------------------------
        if not auto_reject_reasons and documents and not st.session_state.get(SESSION_AUTO_CACHED_KEY):
            try:
                cached_res = auto_cache_verified_documents(documents)
                st.session_state[SESSION_AUTO_CACHED_KEY] = True
                if cached_res:
                    st.toast(f"🏢 거래처 마스터 DB 등록/갱신: {cached_res['vendor_name']} ({cached_res['business_number']})", icon="💾")
            except Exception:
                pass

        has_master_inferred = any(
            "[거래처 마스터 DB 참조]" in (d.vendor_name.evidence or "") or
            "[거래처 마스터 DB 참조]" in (d.business_number.evidence or "") or
            "[거래처 마스터 DB 참조]" in (d.representative.evidence or "") or
            "[거래처 마스터 DB 참조]" in (d.account_number.evidence or "")
            for d in documents
        )
        if has_master_inferred:
            st.info("💡 **[거래처 마스터 DB 연동 완료]** 일부 서류의 OCR 인식률 저하 항목을 기존에 검증된 거래처 마스터 DB에서 안전하게 참조하여 보완했습니다.")

        # ---------------------------------------------------------
        # ⚖️ 금액 교차 검증 (ERP 금액 vs 견적서 금액)
        # ---------------------------------------------------------
        st.divider()
        st.markdown("#### ⚖️ 금액 교차 검증")
        doc_total = (target_doc.total.value or 0) if (target_doc and target_doc.total and target_doc.total.value is not None) else 0
        total_erp = total_erp_amt or 0

        if total_erp > 0 and doc_total > 0:
            if total_erp == doc_total:
                st.success(f"⚖️ **[금액 완벽 일치]** ERP 요청액과 견적서 총액(VAT포함)이 정확히 일치합니다. ({total_erp:,}원)")
            elif round(doc_total * 1.1) == total_erp:
                st.success(f"⚖️ **[부가세(10%) 보정 일치]** 견적서 공급가액에 부가세를 합산하면 ERP 금액({total_erp:,}원)과 일치합니다. ➔ `[VAT_CORRECTED]`")
            else:
                diff = doc_total - total_erp
                st.error(f"🚨 **[금액 불일치]** ERP 요청액: **{total_erp:,}원** / 견적서 추출액: **{doc_total:,}원** / 차액: **{diff:+,}원**")
        elif total_erp > 0:
            st.info(f"ERP 구매요청액: **{total_erp:,}원** (견적서 총액 미확정)")
        elif doc_total > 0:
            st.info(f"견적서 추출 총액: **{doc_total:,}원** (ERP 요청서 텍스트 미입력)")
        else:
            st.caption("ERP 요청액 또는 견적서 총액이 입력/추출되지 않아 금액 교차 검증을 대기합니다.")

        # ---------------------------------------------------------
        # 🔍 품목 스마트 매칭 테이블
        # ---------------------------------------------------------
        st.divider()
        st.markdown("#### 🔍 품목 교차 검증 (ERP ↔ 견적서 스마트 매칭)")
        doc_items = target_doc.items if target_doc else []

        if item_list and doc_items:
            matches = smart_item_matching(item_list, doc_items)
            failed = [m for m in matches if "실패" in m.get("최종 판정", "") or "다름" in m.get("수량 검증", "") or "차액" in m.get("금액 검증", "")]
            pending = [m for m in matches if "확인필요" in m.get("최종 판정", "")]
            doc_name_str = f"({target_doc.filename})" if target_doc else ""

            if failed:
                st.warning(f"⚠️ ERP 품목과 견적서 간 불일치(품명·수량·금액) 주의 항목이 {len(failed)}건 있습니다. 아래 대조표를 확인하세요.")
            elif pending:
                st.info(f"🔍 총액은 일치하지만 개별 품명이 상이하여 담당자 확인이 필요한 항목이 {len(pending)}건 있습니다 (보류/확인요망). 아래 대조표를 확인하세요.")
            else:
                st.success(f"✅ ERP 입력 품목 {len(matches)}건 전체가 견적서{doc_name_str} 품목·수량·금액과 정상 매칭되었습니다.")
            matching_df = pd.DataFrame(matches)
            st.dataframe(matching_df, use_container_width=True, hide_index=True)
        elif not item_list:
            st.caption("ERP 물품/용역 내역 텍스트가 입력되지 않아 품목 스마트 매칭을 생략합니다.")
        else:
            st.caption("견적서에서 추출된 세부 품목이 없어 품목 스마트 매칭을 생략합니다 (총액 견적서 등).")

        # ---------------------------------------------------------
        # 📝 ERP 기본 정보 & 📦 품목 내역 뷰어
        # ---------------------------------------------------------
        st.divider()
        st.markdown("#### 📝 ERP 기본 정보")
        c_e1, c_e2, c_e3 = st.columns(3)
        with c_e1: display_status_value("🔢 구매요구번호", base_data.get("req_no"))
        with c_e2: display_status_value("📌 과제번호", base_data.get("p_no"))
        with c_e3: display_status_value("👤 연구책임자", base_data.get("pi_name"))

        c_e4, c_e5, c_e6, c_e7 = st.columns(4)
        with c_e4: display_status_value("🏫 학과/소속", base_data.get("pi_dept"))
        with c_e5: display_status_value("💰 구매요청액", f"{total_erp_amt:,}원" if total_erp_amt else "미검출")
        with c_e6: display_status_value("📧 연락처/이메일", base_data.get("pi_formatted"))
        with c_e7: display_status_value("💳 재원 구분", "⚠️ 간접비 (확인필요)" if "간접비" in raw_fin else ("직접비" if raw_fin.strip() else "미입력"))

        if item_list:
            st.markdown("#### 📦 ERP 품목 내역")
            t_data = []
            for idx, itm in enumerate(item_list, start=1):
                t_data.append({
                    "No.": idx,
                    "품명": itm["name"],
                    "단위": itm["unit"],
                    "등록여부": itm["reg_status"],
                    "물품담당자": itm["contact"],
                    "단가": f"{itm['unit_price']:,}원" if itm["unit_price"] else "-",
                    "금액": f"{itm['price']:,}원" if itm["price"] else "-",
                })
            st.dataframe(pd.DataFrame(t_data), use_container_width=True, hide_index=True)

        # ---------------------------------------------------------
        # 📁 업체 정보 추출값 (의견 생성 및 원클릭 보드용)
        # ---------------------------------------------------------
        license_doc = next((d for d in documents if d.document_type == DocumentType.BUSINESS_LICENSE), None)
        cand_ven = (target_doc.vendor_name.raw if target_doc and target_doc.vendor_name.raw else "") or (license_doc.vendor_name.raw if license_doc and license_doc.vendor_name.raw else "")
        def_ven = manual_ai.get("업체명") or cand_ven
        cand_biz = (target_doc.business_number.raw if target_doc and target_doc.business_number.raw else "") or (license_doc.business_number.raw if license_doc and license_doc.business_number.raw else "")
        def_biz = manual_ai.get("사업자번호") or cand_biz
        cand_rep = (target_doc.representative.raw if target_doc and target_doc.representative.raw else "") or (license_doc.representative.raw if license_doc and license_doc.representative.raw else "")
        def_rep = manual_ai.get("대표자명") or cand_rep
        def_contact = manual_ai.get("담당자") or (target_doc.contact_person.raw if target_doc else "")
        def_phn = manual_ai.get("연락처") or (target_doc.phone.raw if target_doc else "")
        def_eml = manual_ai.get("이메일") or (target_doc.email.raw if target_doc else "")

        # ---------------------------------------------------------
        # ✍️ 결재 의견 자동 생성
        # ---------------------------------------------------------
        st.divider()
        st.markdown("#### ✍️ 결재 의견 (자동 생성)")
        st.caption("💡 오른쪽 📋 아이콘을 눌러 결재 상신용 의견 텍스트를 바로 복사하세요.")
        opinion_text = generate_opinion(
            req_type,
            con_type,
            total_erp_amt or doc_total,
            item_list,
            raw_fin,
            vendor_contact=def_contact or def_phn or def_eml,
        )
        st.code(opinion_text, language="text")

        # ---------------------------------------------------------
        # 📁 업체 정보 통합 & ✨ 원클릭 복사 보드
        # ---------------------------------------------------------
        st.divider()
        st.markdown("#### 📁 업체 정보 통합 및 ✨ 원클릭 복사 보드")

        c_v1, c_v2, c_v3, c_v4, c_v5, c_v6 = st.columns(6)
        with c_v1: t_ven = st.text_input("업체명", value=def_ven, key="in_vendor_name")
        with c_v2: t_biz = st.text_input("사업자번호", value=def_biz, key="in_biz_num")
        with c_v3: t_rep = st.text_input("대표자명(CEO)", value=def_rep, key="in_representative")
        with c_v4: t_contact = st.text_input("담당자명(실무)", value=def_contact, key="in_contact_person")
        with c_v5: t_phn = st.text_input("연락처", value=def_phn, key="in_phone")
        with c_v6: t_eml = st.text_input("이메일", value=def_eml, key="in_email")

        st.caption("✨ **원클릭 복사 보드 (우측 📋 아이콘 클릭)**")
        c_cp1, c_cp2, c_cp3, c_cp4 = st.columns([1.2, 1.2, 1.0, 2.2])
        with c_cp1: st.code(t_ven or "업체명 없음", language=None)
        with c_cp2: st.code(t_biz or "사업자번호 없음", language=None)
        with c_cp3: st.code(t_rep or "대표자 없음", language=None)
        with c_cp4: st.code(f"{t_contact or '담당자'} / {t_phn or '연락처'} / {t_eml or '이메일'}", language=None)

        # 추천 폴더명 자동 생성
        now = datetime.date.today()
        t_dept = base_data.get("pi_dept", "학과")
        t_pi = base_data.get("pi_name", "연구책임자")
        fold_nm = f"({now.strftime('%Y.%m.%d')}) {t_dept} {t_pi} - {t_ven}"
        st.caption("📁 **추천 폴더명 (바탕화면 서류 보관용)**")
        st.code(fold_nm, language="text")

        # ---------------------------------------------------------
        # 🏦 통장사본 계좌정보
        # ---------------------------------------------------------
        st.divider()
        st.markdown("#### 🏦 통장사본 계좌정보")
        bank_doc = next((d for d in documents if d.document_type == DocumentType.BANK_COPY), None)
        def_holder = manual_ai.get("예금주") or (bank_doc.account_holder.raw if bank_doc else "")
        def_acc = manual_ai.get("계좌번호") or (bank_doc.account_number.raw if bank_doc else "")

        c_b1, c_b2 = st.columns(2)
        with c_b1:
            t_acc_holder = st.text_input("예금주 (수정 가능)", value=def_holder, key="in_acc_holder")
            st.code(t_acc_holder or "예금주 없음", language=None)
        with c_b2:
            t_acc_num = st.text_input("계좌번호 (수정 가능)", value=def_acc, key="in_acc_num")
            st.code(t_acc_num or "계좌번호 없음", language=None)

        # ---------------------------------------------------------
        # 💾 로컬 캐시 DB 저장
        # ---------------------------------------------------------
        st.divider()
        if st.button("💾 현재 분석 내역을 시스템(DB)에 저장하기", type="primary", use_container_width=True):
            req_no_val = base_data.get("req_no") or now.strftime("%Y%m%d%H%M%S")
            meta_save = {
                "요청분석일자": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "구매요구번호": req_no_val,
                "과제번호": base_data.get("p_no", ""),
                "구매요청액": f"{total_erp_amt:,}원" if total_erp_amt else f"{doc_total:,}원",
                "연구책임자정보": base_data.get("pi_formatted", ""),
                "물품담당자": item_list[0]["contact"] if item_list else "미기재",
                "업체명": t_ven,
                "대표자명": t_rep,
                "업체담당자정보": f"{t_contact} / {t_phn} / {t_eml}",
                "사업자번호": t_biz,
                "계좌번호": t_acc_num,
                "예금주": t_acc_holder,
            }
            raw_save = {
                "req_input": raw_req,
                "fin_input": raw_fin,
                "itm_input": raw_itm,
                "req_type": req_type,
                "con_type": con_type,
                "manual_ai": manual_ai,
            }
            save_to_cache(req_no_val, meta_save, raw_save)
            if t_biz and t_ven:
                try:
                    save_vendor(
                        biz_num=t_biz,
                        vendor_name=t_ven,
                        representative=t_rep,
                        account_num=t_acc_num,
                        account_holder=t_acc_holder,
                        phone=t_phn,
                        email=t_eml,
                    )
                except Exception:
                    pass
            st.success(f"✅ 구매요구번호 [{req_no_val}] 건의 데이터 및 거래처 마스터 정보가 성공적으로 저장되었습니다! ('과거 작업 내역 불러오기'에서 확인 가능)")


if __name__ == "__main__":
    main()
