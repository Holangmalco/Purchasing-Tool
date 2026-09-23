import streamlit as st

st.set_page_config(page_title="로컬 구매 문서 검증", layout="wide")

def home() -> None:
    st.title("구매 문서 검증")
    st.write("Gemini나 외부 API 없이 PDF, 이미지, XLSX를 로컬 OCR과 규칙으로 분석합니다.")
    st.markdown("### 제공 기능")
    st.write("견적서·사업자등록증·통장사본·공사 확인서를 분류하고, 금액·품목·업체 정보를 문서 간 비교합니다.")
    st.info("OCR 결과는 검토를 돕는 참고자료이며 법적 진위, 계좌 소유권, 자동 승인을 확정하지 않습니다.")

p1 = st.Page("pages/1_1_구매추출기_신규.py", title="구매 분석 대시보드", icon="💰", default=True)
p2 = st.Page("pages/2_분석내용_불러오기.py", title="과거 내역 대시보드", icon="🕒")

pg = st.navigation([p1, p2], position="top")
pg.run()
