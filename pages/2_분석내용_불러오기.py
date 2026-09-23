"""Past analysis history dashboard, data recovery, and vendor master DB management."""
import json
import time
import pandas as pd
import streamlit as st
from purchase_verifier.erp_parser import CACHE_FILE, get_cache_list, load_from_cache
from purchase_verifier.vendor_cache import (
    delete_vendor, get_all_vendors, normalize_biz_num, save_vendor,
)

st.set_page_config(page_title="과거 작업 내역 및 거래처 DB", page_icon="🕒", layout="wide")

st.title("🕒 업무 이력 및 거래처 마스터 관리 대시보드")
st.caption("과거 분석 내역을 복구하거나, OCR 자동 추출 보완에 활용되는 거래처 마스터 데이터베이스(DB)를 관리합니다.")

tab1, tab2 = st.tabs(["🕒 과거 구매요청 작업 내역", "🏢 거래처 마스터 DB (캐시 관리)"])

with tab1:
    st.markdown("### 📊 과거 작업 내역 및 데이터 복구")
    st.caption("이전에 작업하고 저장해둔 분석 내역을 엑셀 대장처럼 확인하고, 언제든 다시 불러와서 이어서 작업할 수 있습니다.")

    cache_list = get_cache_list()
    if not cache_list:
        st.info("아직 저장된 작업 내역이 없습니다. '구매 문서 검증' 화면에서 작업을 마치고 [💾 현재 분석 내역 저장] 버튼을 눌러주세요.")
    else:
        st.markdown("#### 📊 내장 데이터베이스 (목록 직접 수정 가능 ✏️)")
        st.info("💡 표 안의 글씨를 **더블클릭**해서 엑셀처럼 바로 수정할 수 있습니다. (고유번호인 구매요구번호는 수정 불가)\n\n수정을 마친 후 반드시 아래의 **[저장]** 버튼을 눌러주세요!")

        edited_df = st.data_editor(
            pd.DataFrame(cache_list),
            use_container_width=True,
            hide_index=True,
            disabled=["구매요구번호"],
        )

        if st.button("💾 대시보드 표 수정사항을 DB에 즉시 반영하기", type="secondary"):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    cache_db = json.load(f)

                for _, row in edited_df.iterrows():
                    req_no = str(row["구매요구번호"])
                    if req_no in cache_db:
                        cache_db[req_no]["meta"]["과제번호"] = row["과제번호"]
                        cache_db[req_no]["meta"]["구매요청액"] = row["구매요청액"]
                        cache_db[req_no]["meta"]["연구책임자정보"] = row["연구책임자정보"]
                        cache_db[req_no]["meta"]["물품담당자"] = row["물품담당자"]
                        cache_db[req_no]["meta"]["업체명"] = row["업체명"]
                        cache_db[req_no]["meta"]["업체담당자정보"] = row["업체담당자정보"]

                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(cache_db, f, ensure_ascii=False, indent=2)

                st.success("✅ 대시보드 수정사항이 DB에 완벽하게 덮어쓰기 되었습니다!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"🚨 DB 업데이트 중 오류 발생: {e}")

        st.divider()
        st.markdown("#### 📂 데이터 복구하기")

        c_load1, c_load2 = st.columns([3, 1])
        with c_load1:
            req_val = st.text_input("불러올 구매요구번호를 입력(또는 붙여넣기)하세요", placeholder="예: 202609090001", key="target_req_no")

        with c_load2:
            st.write("")
            st.write("")
            do_load = st.button("🚀 데이터 불러오기", type="primary", use_container_width=True)

        if do_load:
            if not req_val or not req_val.strip():
                st.error("🚨 구매요구번호를 입력해주세요.")
            else:
                loaded_data = load_from_cache(req_val.strip())
                if loaded_data:
                    st.session_state["erp_raw_req"] = loaded_data.get("req_input", "")
                    st.session_state["erp_raw_fin"] = loaded_data.get("fin_input", "")
                    st.session_state["erp_raw_itm"] = loaded_data.get("itm_input", "")
                    st.session_state["p_req_type"] = loaded_data.get("req_type", "📦 물품")
                    st.session_state["p_con_type"] = loaded_data.get("con_type", "비교견적")
                    st.session_state["manual_json_data"] = loaded_data.get("manual_ai", {})
                    st.session_state["req_analysis_done"] = True

                    st.success(f"✅ 구매요구번호 [{req_val}] 건의 데이터가 복구되었습니다! 상단 '구매 문서 검증' 메뉴로 이동하시면 바로 확인하실 수 있습니다.")
                else:
                    st.error(f"🚨 입력하신 번호 [{req_val}]에 해당하는 저장 데이터가 없습니다.")

with tab2:
    st.markdown("### 🏢 거래처 마스터 데이터베이스 (Vendor Master DB)")
    st.caption("영구 불변인 **사업자등록번호(10자리)**를 기준으로 구축된 로컬 SQLite 캐시입니다. 흐릿한 스캔본 등 OCR 인식률이 떨어졌을 때 자동으로 안전하게 보완(Fallback)합니다.")

    vendors = get_all_vendors()

    # 지표 요약
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("📋 등록된 거래처 수", f"{len(vendors)} 개사")
    with m2:
        verified_total = sum(v.get("verified_count", 1) for v in vendors)
        st.metric("🔄 누적 검증 통과 횟수", f"{verified_total} 회")
    with m3:
        latest_trade = max((v.get("last_seen", "-") for v in vendors), default="-")
        st.metric("📅 최근 거래 갱신일", latest_trade)

    st.divider()

    # 1. 거래처 목록 테이블
    st.markdown("#### 📑 등록 거래처 현황")
    if not vendors:
        st.info("현재 등록된 거래처 마스터 정보가 없습니다. 서류 검증 시 통과된 업체가 자동으로 등록되거나, 아래에서 직접 등록할 수 있습니다.")
    else:
        table_rows = []
        for v in vendors:
            # 계좌 포맷
            accs = v.get("accounts", [])
            acc_str = ", ".join(f"{a.get('bank', '')} {a.get('account', '')}".strip() for a in accs) if accs else "-"
            # 별칭
            aliases = v.get("aliases", [])
            alias_str = ", ".join(aliases) if aliases else v["vendor_name"]
            table_rows.append({
                "사업자등록번호": v["business_number"],
                "정식 상호명": v["vendor_name"],
                "대표자": v.get("representative") or "-",
                "등록 계좌": acc_str,
                "연락처": v.get("phone") or "-",
                "이메일": v.get("email") or "-",
                "검증 횟수": f"{v.get('verified_count', 1)}회",
                "최근 거래일": v.get("last_seen", "-"),
                "인식 별칭목록": alias_str,
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.divider()

    # 2. 거래처 관리 (직접 등록 / 삭제)
    col_v1, col_v2 = st.columns([3, 2])

    with col_v1:
        st.markdown("#### ➕ 신규 거래처 직접 등록 / 정보 갱신")
        with st.form("form_register_vendor", clear_on_submit=True):
            f_bn = st.text_input("사업자등록번호 (10자리 또는 하이픈 포함)*", placeholder="예: 123-45-67890")
            f_vn = st.text_input("상호명(법인명)*", placeholder="예: (주)테스트링크")
            f_rep = st.text_input("대표자명", placeholder="예: 홍길동")
            c_bk1, c_bk2 = st.columns(2)
            with c_bk1:
                f_bank = st.text_input("주거래 은행", placeholder="예: 국민은행")
            with c_bk2:
                f_acc = st.text_input("계좌번호", placeholder="예: 123-456-789012")
            f_holder = st.text_input("예금주명", placeholder="예: (주)테스트링크")
            c_ct1, c_ct2 = st.columns(2)
            with c_ct1:
                f_ph = st.text_input("연락처", placeholder="예: 02-1234-5678")
            with c_ct2:
                f_em = st.text_input("이메일", placeholder="예: sales@vendor.co.kr")

            submit_vendor = st.form_submit_button("💾 거래처 마스터 DB에 저장", type="primary", use_container_width=True)
            if submit_vendor:
                clean_bn = normalize_biz_num(f_bn)
                if not clean_bn or len(clean_bn) != 12:
                    st.error("🚨 사업자등록번호 형식이 올바르지 않습니다 (10자리 필요).")
                elif not f_vn.strip():
                    st.error("🚨 상호명을 입력해주세요.")
                else:
                    ok = save_vendor(
                        biz_num=clean_bn,
                        vendor_name=f_vn.strip(),
                        representative=f_rep.strip(),
                        bank=f_bank.strip(),
                        account_num=f_acc.strip(),
                        account_holder=f_holder.strip(),
                        phone=f_ph.strip(),
                        email=f_em.strip(),
                    )
                    if ok:
                        st.success(f"✅ 거래처 [{f_vn}] ({clean_bn}) 정보가 마스터 DB에 성공적으로 저장되었습니다!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("🚨 저장 중 오류가 발생했습니다.")

    with col_v2:
        st.markdown("#### 🗑️ 거래처 삭제")
        st.caption("폐업했거나 오인식되어 잘못 캐시된 거래처를 마스터 DB에서 영구 삭제합니다.")
        if vendors:
            v_options = {f"{v['vendor_name']} ({v['business_number']})": v["business_number"] for v in vendors}
            selected_to_delete = st.selectbox("삭제할 거래처 선택", list(v_options.keys()))
            if st.button("🚨 선택한 거래처 DB에서 삭제", type="secondary", use_container_width=True):
                target_bn = v_options[selected_to_delete]
                if delete_vendor(target_bn):
                    st.success(f"✅ 거래처 [{selected_to_delete}] 삭제가 완료되었습니다.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("🚨 삭제 실패: 대상을 찾을 수 없습니다.")
        else:
            st.caption("삭제할 거래처가 없습니다.")
