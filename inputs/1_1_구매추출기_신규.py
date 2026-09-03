import streamlit as st
import re
import requests
import os
import io
import pandas as pd
from dotenv import load_dotenv

import pdfplumber 
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import cv2
import numpy as np

st.set_page_config(page_title="구매추출기 (단순추출)", page_icon="⚡", layout="wide")

# 1. 환경변수 및 국세청 API 세팅
load_dotenv()
NTS_API_KEY = os.getenv("NTS_API_KEY")

def check_nts_business(biz_no_str):
    clean_b = re.sub(r'[^\d]', '', str(biz_no_str))
    if len(clean_b) != 10: 
        return "⚠️ 번호오류"
    
    url = f"https://api.odcloud.kr/api/nts-businessman/v1/status?serviceKey={NTS_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {"b_no": [clean_b]}
    
    try:
        res = requests.post(url, headers=headers, json=data, timeout=5)
        if res.status_code == 200:
            r_data = res.json().get('data', [])
            if r_data:
                stt_cd = r_data[0].get('b_stt_cd', '')
                if stt_cd == '01': return "✅ 계속사업자"
                elif stt_cd == '02': return "🚨 휴업자"
                elif stt_cd == '03': return "🚨 폐업자"
    except Exception as e:
        return f"⚠️ 조회실패(서버오류: {e})"
    return "⚠️ 조회실패"

# 💡 [업데이트] 줄 단위 문맥 추출 및 OCR 자간 압축
def find_with_context(text, pattern):
    results = []
    # 한글 낱글자 사이의 1~2칸 공백을 제거 (예: 계 약 일 자 -> 계약일자)
    clean_text = re.sub(r'(?<=[가-힣])\s{1,2}(?=[가-힣])', '', text)
    lines = clean_text.split('\n')
    
    for line in lines:
        matches = list(re.finditer(pattern, line))
        if matches:
            hl_line = line
            for match in reversed(matches):
                hl_line = hl_line[:match.start()] + f" 🎯[{match.group()}] " + hl_line[match.end():]
            results.append(hl_line.strip())
            
    return list(set(results))

# 💡 [신규] 금액 2단계 추출 및 수학적 검증(추론) 로직
def extract_financial_amounts(extracted_text):
    raw_amounts = re.findall(r'\b\d{1,3}(?:,\d{3})+\b', extracted_text)
    amounts = sorted(list(set([int(a.replace(',', '')) for a in raw_amounts])), reverse=True)
    
    result = {"합계": None, "공급가액": None, "부가세": None}
    if not amounts:
        return result

    total_pattern = r'(?:합계|총계|총액|공급대가|결제금액|Total)[^\d\n]*(\d{1,3}(?:,\d{3})+)'
    vat_pattern = r'(?:부가세|부가가치세|세액|V\.?A\.?T)[^\d\n]*(\d{1,3}(?:,\d{3})+)'
    supply_pattern = r'(?:공급가|공급가액|단가|Subtotal)[^\d\n]*(\d{1,3}(?:,\d{3})+)'
    
    total_match = re.search(total_pattern, extracted_text, re.IGNORECASE)
    vat_match = re.search(vat_pattern, extracted_text, re.IGNORECASE)
    supply_match = re.search(supply_pattern, extracted_text, re.IGNORECASE)
    
    if total_match: result["합계"] = f"{total_match.group(1)} 원"
    if supply_match: result["공급가액"] = f"{supply_match.group(1)} 원"
    if vat_match: result["부가세"] = f"{vat_match.group(1)} 원"
    
    if len(amounts) >= 2 and (not result["합계"] or not result["부가세"]):
        for i in range(len(amounts)):
            for j in range(i+1, len(amounts)):
                total_candidate = amounts[i]
                supply_candidate = amounts[j]
                expected_vat = int(supply_candidate * 0.1) 
                
                if abs(total_candidate - (supply_candidate + expected_vat)) <= 1:
                    if not result["합계"]: result["합계"] = f"{total_candidate:,} 원 🤖(추론)"
                    if not result["공급가액"]: result["공급가액"] = f"{supply_candidate:,} 원 🤖(추론)"
                    if not result["부가세"]: result["부가세"] = f"{expected_vat:,} 원 🤖(추론)"
                    break
                    
    if not result["합계"] and amounts:
        result["합계"] = f"{amounts[0]:,} 원 ⚠️(최대값)"

    return result

# 💡 [업데이트] 붉은색 직인 제거 및 이미지 전처리 파이프라인
def preprocess_scan_image(pil_img):
    img_cv = np.array(pil_img)
    
    if len(img_cv.shape) == 3:
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        img_cv[mask > 0] = [255, 255, 255] # 붉은색을 흰색으로 변경
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_cv

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
            
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.erode(binary, kernel, iterations=1)
    
    return processed

st.title("⚡ 구매추출기 (선택형 추출 모드)")
st.info("다양한 형태의 견적서(PDF, 이미지, 엑셀)를 로컬에서 즉시 텍스트로 변환하고 핵심 정보를 뽑아냅니다.")

# 1. 구매/계약 기본 정보 선택
st.markdown("#### 📂 1. 구매/계약 기본 정보 선택")
c_type1, c_type2 = st.columns(2)
with c_type1:
    req_type = st.radio("구매 유형", ["📦 물품", "🛠️ 용역", "🏗️ 공사"], horizontal=True, key="new_req_type")
with c_type2:
    con_type = st.radio("계약 방식", ["비교견적", "수의계약"], horizontal=True, key="new_con_type")

st.divider()

# 2. ERP 텍스트 복사 및 업로드란 (2단 분할)
col_in1, col_in2 = st.columns(2)

with col_in1:
    with st.container(border=True):
        st.markdown("#### 📝 2. ERP 텍스트 복사")
        if "p_req" not in st.session_state: st.session_state["p_req"] = ""
        if "p_fin" not in st.session_state: st.session_state["p_fin"] = ""
        if "p_itm" not in st.session_state: st.session_state["p_itm"] = ""
        
        raw_req = st.text_area("1️⃣ 구매요청서 전체", height=120, value=st.session_state["p_req"])
        st.session_state["p_req"] = raw_req
        
        raw_fin = st.text_area("2️⃣ 재원내역 행", height=60, value=st.session_state["p_fin"])
        st.session_state["p_fin"] = raw_fin
        
        raw_itm = st.text_area("3️⃣ 물품/용역내역 행", height=80, value=st.session_state["p_itm"])
        st.session_state["p_itm"] = raw_itm

with col_in2:
    with st.container(border=True):
        st.markdown("#### 📎 3. 만능 견적서 업로드 & 국세청 조회")
        
        raw_biz_no = st.text_input("🔍 사업자등록번호 수동 입력 (실시간 조회)", placeholder="예: 1234567890")
        if raw_biz_no:
            with st.spinner("국세청 상태 조회 중..."):
                status = check_nts_business(raw_biz_no)
                if "계속사업자" in status:
                    st.success(f"결과: [{raw_biz_no}] {status}")
                else:
                    st.error(f"결과: [{raw_biz_no}] {status}")
                    
        st.markdown("---")
        uploaded_files = st.file_uploader("PDF, 이미지, 엑셀(.xlsx) 모두 지원", accept_multiple_files=True, type=['pdf', 'png', 'jpg', 'jpeg', 'xlsx'])

st.divider()

# 3. 추출 실행 버튼부
if uploaded_files:
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        btn_local = st.button("⚡ 빠르고 단순하게 텍스트만 추출 (로컬)", use_container_width=True)
    with c_btn2:
        btn_ai = st.button("🧠 제미나이 AI 정밀 분석 (추후 연결)", type="primary", use_container_width=True)

    if btn_local:
        with st.spinner("로컬 환경에서 파일을 분석 중입니다..."):
            for f in uploaded_files:
                st.markdown(f"#### 📄 파일명: {f.name}")
                extracted_text = ""
                
                try:
                    if f.name.lower().endswith('.xlsx'):
                        df = pd.read_excel(f)
                        extracted_text = df.to_string(index=False)
                        st.success("✅ 엑셀 데이터 추출 완료")

                    elif f.name.lower().endswith('.pdf'):
                        with pdfplumber.open(f) as pdf:
                            for page in pdf.pages:
                                text = page.extract_text()
                                if text: extracted_text += text + "\n"
                        
                        if len(extracted_text.strip()) < 50:
                            st.warning("⚠️ 스캔본 PDF 감지됨 ➔ 이미지 전처리 및 분석 모드로 우회합니다.")
                            f.seek(0) 
                            images = convert_from_bytes(f.read())
                            extracted_text = ""
                            for img in images:
                                processed_img = preprocess_scan_image(img)
                                extracted_text += pytesseract.image_to_string(processed_img, lang='kor+eng', config='--psm 6') + "\n"
                        else:
                            st.success("✅ 일반 PDF 추출 완료")

                    elif f.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img = Image.open(f)
                        processed_img = preprocess_scan_image(img)
                        extracted_text = pytesseract.image_to_string(processed_img, lang='kor+eng', config='--psm 6')
                        st.success("✅ 이미지 OCR 추출 완료")
                        
                        pdf_buffer = io.BytesIO()
                        img.convert('RGB').save(pdf_buffer, format="PDF")
                        st.download_button(
                            label="📥 이 이미지를 PDF로 변환하여 다운로드",
                            data=pdf_buffer.getvalue(),
                            file_name=f"{f.name.split('.')[0]}_변환.pdf",
                            mime="application/pdf",
                            key=f"dl_{f.name}"
                        )
                    
                    if extracted_text.strip():
                        biz_pattern = r'\d{3}-\d{2}-\d{5}'
                        biz_results = find_with_context(extracted_text, biz_pattern)
                        
                        date_pattern = r'\d{4}[-./년]\s?\d{1,2}[-./월]\s?\d{1,2}[일]?'
                        date_results = find_with_context(extracted_text, date_pattern)
                        
                        financial_data = extract_financial_amounts(extracted_text)
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown("**🏢 사업자번호 탐지**")
                            if biz_results:
                                for r in biz_results: st.info(r)
                            else: st.write("감지 안 됨")
                                
                        with col_b:
                            st.markdown("**📅 날짜 탐지**")
                            if date_results:
                                for r in date_results: st.info(r)
                            else: st.write("감지 안 됨")
                                
                        with col_c:
                            st.markdown("**💰 금액 탐지**")
                            if financial_data["합계"]:
                                st.success(f"합계: {financial_data['합계']}")
                                st.write(f"공급가: {financial_data.get('공급가액', '감지 안 됨')}")
                                st.write(f"부가세: {financial_data.get('부가세', '감지 안 됨')}")
                            else: 
                                st.write("금액 감지 안 됨")
                        
                        st.markdown("---")
                        st.text_area("📄 추출된 전체 텍스트 원본", value=extracted_text, height=150, key=f"{f.name}_txt")
                        
                    else:
                        st.error("텍스트를 추출하지 못했습니다.")
                
                except Exception as e:
                    st.error(f"추출 중 오류 발생: {e}")

    if btn_ai:
        st.info("이곳에 기존처럼 제미나이 API를 호출하여 JSON 형태로 뽑아주는 로직이 들어갈 예정입니다.")