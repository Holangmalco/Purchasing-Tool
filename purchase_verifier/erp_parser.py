"""ERP text parsing, smart item matching, NTS API check, and workflow utilities."""
from __future__ import annotations

import datetime
import difflib
import json
import os
import re
from typing import Any, Optional
import requests
from dotenv import load_dotenv

load_dotenv()
NTS_API_KEY = os.getenv("NTS_API_KEY")
CACHE_FILE = "local_cache.json"


def safe_int(val: Any) -> int:
    try:
        return int(re.sub(r"[^\d]", "", str(val)))
    except Exception:
        return 0


def format_phone(p_str: str) -> str:
    p = re.sub(r"[^\d]", "", str(p_str))
    if len(p) == 11:
        return f"{p[:3]}-{p[3:7]}-{p[7:]}"
    elif len(p) == 10:
        if p.startswith("02"):
            return f"{p[:2]}-{p[2:6]}-{p[6:]}"
        return f"{p[:3]}-{p[3:6]}-{p[6:]}"
    return str(p_str)


def calculate_business_days(start_date: datetime.date, end_date: datetime.date) -> int:
    """주말(토/일)을 제외한 평일(영업일) 수를 계산합니다."""
    if end_date < start_date:
        # 이미 지난 경우 음수 영업일 계산
        cur = end_date
        count = 0
        while cur < start_date:
            if cur.weekday() < 5:  # 월~금
                count += 1
            cur += datetime.timedelta(days=1)
        return -count

    cur = start_date
    count = 0
    while cur < end_date:
        if cur.weekday() < 5:  # 월~금
            count += 1
        cur += datetime.timedelta(days=1)
    return count


def check_nts_business(biz_no_str: str) -> str:
    """국세청 사업자 등록 상태 실시간 조회 (NTS_API_KEY 필요, 없으면 안전 안내)."""
    clean_b = re.sub(r"[^\d]", "", str(biz_no_str))
    if len(clean_b) != 10:
        return "⚠️ 번호오류(10자리 아님)"

    if not NTS_API_KEY:
        return "ℹ️ 미조회 (국세청 API 키 미설정)"

    url = f"https://api.odcloud.kr/api/nts-businessman/v1/status?serviceKey={NTS_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {"b_no": [clean_b]}

    try:
        res = requests.post(url, headers=headers, json=data, timeout=4)
        if res.status_code == 200:
            r_data = res.json().get("data", [])
            if r_data:
                stt_cd = r_data[0].get("b_stt_cd", "")
                if stt_cd == "01":
                    return "✅ 계속사업자"
                elif stt_cd == "02":
                    return "🚨 휴업자"
                elif stt_cd == "03":
                    return "🚨 폐업자"
    except Exception as e:
        return f"⚠️ 조회실패(서버오류: {e})"

    return "⚠️ 조회실패"


def parse_contact(token_str: str) -> str:
    if not token_str or str(token_str).strip() in ["미검출", "없음", "확인불가", "N/A", "확인 불가", ""]:
        return "⚠️ 연락처 없음"
    raw = str(token_str).strip()

    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", raw)
    email = email_match.group() if email_match else None

    pure_digits = re.sub(r"[^\d]", "", raw)
    phone = None

    internal_match = re.search(r"(3408|6935)\d{4}", pure_digits)
    if internal_match:
        raw_phone = internal_match.group()
        phone = f"02-{raw_phone[:4]}-{raw_phone[4:]}"
    elif len(pure_digits) in [9, 10, 11]:
        phone = format_phone(pure_digits)
    elif len(pure_digits) == 4:
        if pure_digits.startswith(("34", "69")):
            phone = f"02-3408-{pure_digits}"
        else:
            phone = f"내선 {pure_digits}"

    clean_name = raw
    if email:
        clean_name = clean_name.replace(email, "")
    phone_digits_match = re.search(r"[\d\-\s]{4,}", clean_name)
    if phone_digits_match:
        clean_name = clean_name.replace(phone_digits_match.group(), "")

    clean_name = re.sub(r"[\(\)\[\]\-\:\/\_\=\+]", "", clean_name).strip()

    if clean_name and phone:
        return f"{clean_name} / {phone}"
    elif clean_name and email:
        return f"{clean_name} / {email}"
    elif email:
        return f"담당자 / {email}"
    elif phone:
        return f"담당자 / {phone}"
    elif clean_name and len(clean_name) >= 2:
        return f"{clean_name} / ⚠️ 연락처 미기재"

    return raw.strip() if raw.strip() else "⚠️ 연락처 없음"


def extract_base_info(text: str) -> dict[str, str]:
    """구매요청서 전체 텍스트에서 기본 정보를 추출합니다. (연차연구기간 종료일 포함)"""
    info = {
        "p_no": "⚠️ 확인 불가",
        "req_no": "⚠️ 확인 불가",
        "pi_name": "",
        "pi_dept": "",
        "pi_formatted": "",
        "prj_end": "",
        "pi_email": "",
    }
    if not text:
        return info

    p_no_match = re.search(r"(\d{4}-\d{4}-\d{2})", text)
    if p_no_match:
        info["p_no"] = p_no_match.group(1)

    req_no_match = re.search(r"구매요구번호[^\d]*(\d{10,14})", text)
    if req_no_match:
        info["req_no"] = req_no_match.group(1)

    # 연차연구기간의 기간 종료일만 정밀 추출 (사용자 요구사항 4번)
    prj_match = re.search(r"연차연구기간\s*(\d{4}[-./년]\s*\d{1,2}[-./월]\s*\d{1,2})\s*[~-]\s*(\d{4}[-./년]\s*\d{1,2}[-./월]\s*\d{1,2})", text)
    if prj_match:
        raw_end = prj_match.group(2)
        end_clean = re.sub(r"[년월일\s]", "-", raw_end).strip("-")
        parts = [p for p in end_clean.split("-") if p]
        if len(parts) == 3:
            info["prj_end"] = f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
    else:
        # 단일 종료일 표기 패턴
        single_end = re.search(r"(?:연구종료일|종료일자|종료일)[^\d]*(20\d{2}[-./년]\s*\d{1,2}[-./월]\s*\d{1,2})", text)
        if single_end:
            raw_end = single_end.group(1)
            end_clean = re.sub(r"[년월일\s]", "-", raw_end).strip("-")
            parts = [p for p in end_clean.split("-") if p]
            if len(parts) == 3:
                info["prj_end"] = f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"

    pi_match = re.search(r"연구책임자정보\s*(.*?)\s*연차연구기간", text, re.DOTALL)
    if pi_match:
        parts = [p.strip() for p in pi_match.group(1).strip().split("/")]
        if len(parts) >= 4:
            info["pi_name"] = parts[0]
            info["pi_dept"] = parts[2] if len(parts) >= 5 else parts[1]
            info["pi_email"] = parts[-1]
            info["pi_formatted"] = f"{parts[0]} / {format_phone(parts[-2])} / {parts[-1]}"

            clean_text = text.replace(" ", "")
            pi_name = info["pi_name"]
            prj_context_match = re.search(r"과제성격(.*?)(연구책임자정보|연차연구기간)", clean_text, flags=re.DOTALL)
            prj_context = prj_context_match.group(1) if prj_context_match else ""

            if pi_name == "박재우" and ("지역혁신중심" in prj_context or "RISE" in prj_context.upper()):
                info["pi_dept"] += "(RISE사업단)"
            elif pi_name == "김재호" and "사물인터넷" in prj_context:
                info["pi_dept"] += "(사물인터넷혁신융합대학사업단)"
            elif pi_name == "송오영" and "SW중심대학" in prj_context:
                info["pi_dept"] += "(SW중심대학사업단)"
        else:
            info["pi_formatted"] = pi_match.group(1).strip()

    return info


def get_accurate_total(text: str, keyword: str = "합계액") -> int:
    """ERP 본문 내에서 합계액(구매요청액)을 정확히 추출합니다."""
    if not text:
        return 0
    search_area = text.split(keyword)[-1] if keyword and keyword in text else text
    search_area = re.sub(r"(?<!\d)\d+(?:-\d+)+(?!\d)", "", search_area)
    search_area = re.sub(r"(?<!\d)\d[\d,]*\s*(?:mm|cm|m|kg|g|mg|ml|l|v|hz|w|ea|u|%)(?!\w)", "", search_area, flags=re.IGNORECASE)
    search_area = re.sub(r"[a-zA-Z]+\d+|\d+[a-zA-Z]+", "", search_area)
    # 한글 '원' 앞에서도 금액(2,500,000 등)이 잘리지 않도록 정규식 적용
    all_numbers = re.findall(r"(?<!\d)(?:\d{1,3}(?:,\d{3})+|\d+)(?!\d)", search_area)

    clean_numbers = []
    for n in all_numbers:
        num_str = n.replace(",", "")
        val = safe_int(num_str)
        if num_str.startswith("0") and len(num_str) > 1:
            continue
        if len(num_str) >= 11:
            continue
        if len(num_str) == 8 and (num_str.startswith("19") or num_str.startswith("20")):
            month = safe_int(num_str[4:6])
            day = safe_int(num_str[6:8])
            if 1 <= month <= 12 and 1 <= day <= 31:
                continue
        if val >= 1000 or val == 0:
            clean_numbers.append(val)
    return max(clean_numbers) if clean_numbers else 0


def parse_goods_erp(text: str) -> tuple[list[dict[str, Any]], bool]:
    """물품 내역 행을 탭 구분자로 파싱합니다."""
    items = []
    valid_format = False
    if not text or not text.strip():
        return items, valid_format

    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        tokens = line.split("\t")
        try:
            anchor_idx = next((i for i, t in enumerate(tokens) if t.strip() in ["예", "아니오"]), -1)
            if anchor_idx != -1:
                valid_format = True
                reg_status = tokens[anchor_idx].strip()

                if anchor_idx >= 11:
                    cat_idx, con_idx = anchor_idx - 5, anchor_idx + 1
                    contact_raw = tokens[con_idx].strip() if con_idx < len(tokens) else ""
                    price = safe_int(tokens[anchor_idx - 6])
                    qty = safe_int(tokens[anchor_idx - 11])
                    unit = tokens[anchor_idx - 10].strip()
                    unit_price = safe_int(tokens[anchor_idx - 9])
                    raw_cat = tokens[cat_idx].strip() if cat_idx >= 0 and len(tokens) > cat_idx else ""
                else:
                    # 축약형/단축형 테이블 대비: 숫자 토큰 자동 탐지 (전화번호 형태 제외)
                    raw_nums = []
                    for idx, tok in enumerate(tokens):
                        if idx == anchor_idx:
                            continue
                        clean_digits = re.sub(r"\D", "", tok)
                        # 9자리 이상(010, 02 전화번호 등)은 단가/금액에서 제외
                        if clean_digits.startswith(("010", "011", "02")) or len(clean_digits) >= 9:
                            continue
                        clean_num = safe_int(tok)
                        if clean_num > 0:
                            raw_nums.append((idx, clean_num, tok))
                    
                    price = raw_nums[-1][1] if raw_nums else 0
                    unit_price = raw_nums[-2][1] if len(raw_nums) >= 2 else price
                    qty = raw_nums[0][1] if len(raw_nums) >= 3 else 1
                    unit = "ea"
                    raw_cat = "비품" if "예" in reg_status else "소모품"
                    contact_raw = tokens[-1].strip() if tokens else ""

                name_str = tokens[1].strip() if len(tokens) > 1 else "무명"
                if len(tokens) > 2 and tokens[2].strip() and tokens[2].strip() != tokens[1].strip():
                    name_str = f"{name_str} ({tokens[2].strip()})"

                clean_n = name_str.replace(" ", "").lower()
                is_sw = is_software_item(clean_n, raw_cat)
                is_fix = is_fixture_item(clean_n, raw_cat)

                # 세종대학교 물품등록여부 사전 판별
                if is_sw:
                    is_short = any(kw in clean_n for kw in ["월간", "1개월", "3개월", "6개월", "단기"])
                    if (price < 2000000 or is_short) and reg_status == "예":
                        reg_status = "예/확인필요"
                    elif (price >= 2000000 and not is_short) and reg_status == "아니오":
                        reg_status = "아니오/확인필요"
                elif is_fix:
                    if unit_price >= 150000 and reg_status == "아니오":
                        reg_status = "아니오/확인필요"
                    elif unit_price > 0 and unit_price < 150000 and reg_status == "예":
                        reg_status = "예/확인필요"
                else:
                    if unit_price >= 300000 and reg_status == "아니오":
                        reg_status = "아니오/확인필요"
                    elif unit_price > 0 and unit_price < 300000 and reg_status == "예":
                        reg_status = "예/확인필요"

                items.append({
                    "name": name_str,
                    "category": raw_cat if raw_cat else "공란",
                    "reg_status": reg_status,
                    "contact": parse_contact(contact_raw),
                    "qty": qty,
                    "unit": unit,
                    "price": price,
                    "unit_price": unit_price,
                })
        except Exception:
            continue
    return items, valid_format


def parse_service_erp(text: str, force_category: str = "용역") -> tuple[list[dict[str, Any]], bool]:
    """용역/공사 내역 행을 탭 구분자로 파싱합니다."""
    items = []
    valid_format = False
    if not text or not text.strip():
        return items, valid_format

    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        tokens = line.split("\t")
        try:
            if len(tokens) >= 15:
                valid_format = True
                name_str = tokens[1].strip() if len(tokens) > 1 else "무명"
                if len(tokens) > 2 and tokens[2].strip():
                    name_str = f"{name_str} ({tokens[2].strip()})"

                items.append({
                    "name": name_str,
                    "category": tokens[9].strip() if tokens[9].strip() else force_category,
                    "reg_status": f"N/A({force_category})",
                    "contact": parse_contact(tokens[14].strip()),
                    "qty": safe_int(tokens[3]),
                    "unit": tokens[4].strip(),
                    "price": safe_int(tokens[8]),
                    "unit_price": safe_int(tokens[5]),
                })
        except Exception:
            continue
    return items, valid_format


def is_fixture_item(clean_name: str, raw_cat: str = "") -> bool:
    """집기 품목 여부 판별 (책상, 의자, 캐비닛 등 사무/실험용 가구 및 집기)."""
    fixture_kws = [
        "책상", "의자", "파티션", "캐비닛", "수납장", "테이블", "책장", "칠판", "소파", "침대",
        "블라인드", "선반", "서랍장", "실험대", "작업대", "옷장", "싱크대", "신발장"
    ]
    return any(kw in clean_name for kw in fixture_kws) or "집기" in raw_cat


def is_software_item(clean_name: str, raw_cat: str = "") -> bool:
    """소프트웨어 품목 여부 판별."""
    sw_kws = ["소프트웨어", "sw", "s/w", "라이선스", "라이센스", "구독", "license", "licence"]
    return any(kw in clean_name for kw in sw_kws) or "소프트웨어" in raw_cat.lower()


def guess_category_by_keyword(name: str, price: int, unit_price: int = 0) -> str:
    """품목명 및 단가 기반 물품분류구분 (비품/소모품/구성품/용역/공사) 추천.
    
    세종대학교 기준:
    - 집기: 15만원 이상 비품 (15만원 미만 소모품)
    - 기계기구/일반장비: 30만원 이상 비품 (30만원 미만 소모품)
    - 소프트웨어: 사용기간 1년 이상 & 200만원 이상 비품 (그 외 소모품)
    - 집기/기계기구는 뭉뚱그려 '비품'으로 통일 표기
    """
    clean_name = name.replace(" ", "").lower()
    chk_price = unit_price if unit_price > 0 else price

    # 1. 구성품 키워드
    component_kws = ["구성품", "부속품", "옵션모듈", "확장보드", "추가모듈", "장착키트", "브라켓", "확장카드"]
    if any(kw in clean_name for kw in component_kws):
        return "구성품"

    # 2. 소프트웨어 (1년 이상 사용 & 200만원 이상 비품, 그 외 소모품)
    if is_software_item(clean_name):
        is_short_term = any(kw in clean_name for kw in ["월간", "1개월", "3개월", "6개월", "단기"])
        if price >= 2000000 and not is_short_term:
            return "비품"
        return "소모품"

    # 3. 집기류 (15만원 이상 비품, 15만원 미만 소모품)
    if is_fixture_item(clean_name):
        return "비품" if chk_price >= 150000 else "소모품"

    # 4. 순수 소모품 키워드
    consumable_kws = [
        "시약", "용액", "항체", "키트", "kit", "튜브", "팁", "케이블", "부품", "커버", "케이스",
        "필터", "토너", "잉크", "용지", "리필", "건전지", "배터리", "소모품", "초자", "웨이퍼",
        "타겟", "기판", "가스", "질소", "마우스", "사료", "시편", "필름", "테이프", "마스크",
        "장갑", "방진복", "소모", "유리관", "페트리", "비커", "플라스크", "피펫"
    ]
    if any(kw in clean_name for kw in consumable_kws):
        return "소모품"

    # 5. 시제작품 / 시작품 / 시제품 제작 (세종대 실무 기준: 용역이 아닌 물품구매로 처리)
    # 단가 30만원 이상 완제품/모듈 형태는 '비품', 30만원 미만 단순 소모성 시편 등은 '소모품'
    prototype_kws = ["시제작", "시작품", "시제품", "로봇제작", "기구제작", "의족제작", "mockup", "목업"]
    if any(kw in clean_name for kw in prototype_kws):
        return "비품" if chk_price >= 300000 else "소모품"

    # 6. 기계기구 및 일반 연구장비/물품 (30만원 이상 비품, 30만원 미만 소모품)
    return "비품" if chk_price >= 300000 else "소모품"


def generate_opinion(
    req_type: str,
    contract_type: str,
    amount: int,
    item_list: list[dict[str, Any]],
    raw_finance_text: str = "",
    vendor_contact: Optional[str] = None,
) -> str:
    """결재 의견 자동 생성 문안을 작성합니다."""
    rules_triggered = []
    if "간접비" in raw_finance_text:
        rules_triggered.append("간접비건 확인")

    # 1. 단위 규격 검증 (물품 -> u, 용역/공사 -> 식)
    if "물품" in req_type:
        if any(i.get("unit", "").lower() not in ["", "u"] for i in item_list):
            rules_triggered.append("단위 'u'로 수정 필요")
    elif "용역" in req_type or "공사" in req_type:
        if any(i.get("unit", "") not in ["", "식"] for i in item_list):
            rules_triggered.append("단위 '식'으로 수정 필요")

    # 2. 업체담당자 정보 유무 검증
    if vendor_contact is not None:
        clean_vc = re.sub(r"[\s\-\/\:\(\)\[\]]", "", str(vendor_contact)).strip()
        if not clean_vc or clean_vc in ["미검출", "없음", "확인불가", "NA", "담당자", "담당자연락처미기재"]:
            rules_triggered.append("업체담당자 정보확인필요")

    # 3. 품목별 물품분류구분 및 단가 기준 물품등록여부 검증 (세종대학교 기준)
    expected_cats = []
    for itm in item_list:
        erp_cat = itm.get("category", "공란").strip()
        reg_status = itm.get("reg_status", "")
        price = itm.get("price", 0)
        unit_price = itm.get("unit_price", 0)
        name = itm.get("name", "").upper()
        clean_n = name.replace(" ", "").lower()

        is_sw = is_software_item(clean_n, erp_cat)
        is_fix = is_fixture_item(clean_n, erp_cat)

        # 1) 단가 및 품목 유형별 물품등록여부 검증 (세종대학교 기준)
        # - 소프트웨어: 1년 이상 사용 & 200만원 이상만 등록대상
        # - 집기: 15만원 이상 등록대상
        # - 기계기구/일반물품: 30만원 이상 등록대상
        if is_sw:
            is_short = any(kw in clean_n for kw in ["월간", "1개월", "3개월", "6개월", "단기"])
            if (price < 2000000 or is_short) and "예" in reg_status:
                rules_triggered.append("물품등록여부 변경필요")
            elif (price >= 2000000 and not is_short) and "아니오" in reg_status:
                rules_triggered.append("물품등록여부 변경필요")
        elif is_fix:
            if unit_price >= 150000 and "아니오" in reg_status:
                rules_triggered.append("물품등록여부 변경필요")
            elif unit_price > 0 and unit_price < 150000 and "예" in reg_status:
                rules_triggered.append("물품등록여부 변경필요")
        else:
            if unit_price >= 300000 and "아니오" in reg_status:
                rules_triggered.append("물품등록여부 변경필요")
            elif unit_price > 0 and unit_price < 300000 and "예" in reg_status:
                rules_triggered.append("물품등록여부 변경필요")

        # 2) 물품분류구분 판정 (허용값: 비품 / 소모품 / 구성품 / 용역 / 공사)
        if "공사" in req_type:
            exp_cat = "공사"
        elif "용역" in req_type:
            exp_cat = "용역"
        elif erp_cat == "구성품":
            exp_cat = "구성품"
        else:
            exp_cat = guess_category_by_keyword(name, price, unit_price)

        expected_cats.append(exp_cat)

        if erp_cat in ["", "공란"]:
            rules_triggered.append("물품분류구분 입력필요")
        else:
            # ERP 분류가 집기/기계기구인 경우 비품으로 표준화하여 비교
            norm_erp_cat = "비품" if erp_cat in ["비품", "기계기구", "집기"] else erp_cat
            if norm_erp_cat != exp_cat and exp_cat in ["비품", "소모품", "구성품", "용역", "공사"]:
                rules_triggered.append("물품분류구분 변경필요")

    # 4. 금액 기준 검증 (추정가격 2천만원 / 부가세 포함 2,200만원 초과 여부)
    # 국가계약법 및 산학협력단 규정상 추정가격 2,000만원(VAT 포함 2,200만원) 초과 시 수의계약사유서 또는 전자입찰 대상
    if amount > 22000000 and "비교견적" in contract_type:
        rules_triggered.append("추정가격 2천만원(VAT포함 2,200만원) 초과 건: 수의계약 사유서(또는 전자입찰) 대상 여부 확인 필요")

    # 최종 물품분류구분 표기: 비품 / 소모품 / 구성품 / 용역 / 공사
    valid_cats = [c for c in expected_cats if c in ["비품", "소모품", "구성품", "용역", "공사"]]
    if valid_cats:
        final_category = " / ".join(list(dict.fromkeys(valid_cats)))
    else:
        final_category = "공사" if "공사" in req_type else ("용역" if "용역" in req_type else "소모품")

    unique_rules = list(dict.fromkeys(rules_triggered))
    issues_text = " - 특이사항 없음" if not unique_rules else "\n".join([f" - {r}" for r in unique_rules])

    return f"1. {contract_type}에 의한 구매건\n - 물품분류구분 : {final_category}\n\n2. 확인사항\n{issues_text}"


def smart_item_matching(erp_items: list[dict[str, Any]], doc_items: list[Any]) -> list[dict[str, Any]]:
    """ERP 품목과 견적서 추출 품목 간 문자열 유사도 및 금액 기반 스마트 매칭."""
    match_results = []
    for erp_itm in erp_items:
        erp_name = str(erp_itm.get("name", ""))
        erp_qty = safe_int(erp_itm.get("qty", 1)) or 1
        erp_price = safe_int(erp_itm.get("price", 0))

        best_match, best_score, best_doc_qty, best_doc_amount = None, 0, 0, 0

        for d_itm in doc_items:
            doc_name = getattr(d_itm, "name", "") if hasattr(d_itm, "name") else d_itm.get("품명", "")
            doc_qty = safe_int(getattr(d_itm, "quantity", 1) if hasattr(d_itm, "quantity") else d_itm.get("수량", 1)) or 1
            doc_unit_price = safe_int(getattr(d_itm, "unit_price", 0) if hasattr(d_itm, "unit_price") else d_itm.get("단가", 0))
            doc_amount = safe_int(getattr(d_itm, "amount", 0) if hasattr(d_itm, "amount") else d_itm.get("금액", 0))
            if doc_amount == 0:
                doc_amount = doc_unit_price * doc_qty

            norm_erp = re.sub(r"[^\w가-힣]", "", erp_name).lower()
            norm_doc = re.sub(r"[^\w가-힣]", "", str(doc_name)).lower()
            text_score = difflib.SequenceMatcher(None, norm_erp, norm_doc).ratio() * 100

            price_score = 0
            if erp_price > 0 and doc_amount > 0:
                if erp_price == doc_amount:
                    price_score = 500
                elif erp_price == round(doc_amount * 1.1):
                    price_score = 500
                elif abs(erp_price - round(doc_amount * 1.1)) <= 10:
                    price_score = 400

            total_score = text_score + price_score
            if total_score > best_score:
                best_score = total_score
                best_match = doc_name
                best_doc_qty = doc_qty
                best_doc_amount = doc_amount

        is_matched = best_score >= 400 or best_score >= 50
        name_status = "✅ 매칭 성공" if is_matched else "🚨 매칭 실패"
        qty_status = (
            "✅ 일치"
            if (is_matched and erp_qty == best_doc_qty)
            else (f"🚨 다름 (ERP: {erp_qty} / 견적: {best_doc_qty})" if is_matched else "➖ 비교 불가")
        )

        # 금액 검증 (공급가 일치, VAT 10% 일치, 차액 판정)
        if is_matched and erp_price > 0 and best_doc_amount > 0:
            if erp_price == best_doc_amount:
                amt_status = "✅ 일치"
            elif erp_price == round(best_doc_amount * 1.1):
                amt_status = "✅ VAT(10%) 일치"
            elif abs(erp_price - round(best_doc_amount * 1.1)) <= 10:
                amt_status = "✅ 절사 일치"
            else:
                amt_status = f"🚨 차액 ({erp_price - best_doc_amount:+,}원)"
        elif is_matched and (erp_price > 0 or best_doc_amount > 0):
            amt_status = "➖ 금액 확인요망"
        else:
            amt_status = "➖ 비교 불가"

        match_results.append({
            "ERP 입력 품명": erp_name,
            "견적서 매칭 품명": best_match if is_matched else "매칭 실패",
            "매칭 기준": "💰 금액 매칭" if best_score >= 400 else ("📝 이름 매칭" if best_score >= 50 else "실패"),
            "수량 검증": qty_status,
            "금액 검증": amt_status,
            "최종 판정": name_status,
        })
    return match_results


def build_smart_matrix(documents: list[Any], erp_items: list[dict[str, Any]]) -> Any:
    """견적서 목록과 ERP 품목 내역을 대조하여 종합 품목 매칭 DataFrame을 생성합니다."""
    import pandas as pd
    if not erp_items:
        return pd.DataFrame([{"안내": "ERP 물품 내역이 입력되지 않았습니다."}])

    quotes = [
        d for d in documents
        if getattr(d, "document_type", None) in ("QUOTE", "ONLINE_STORE_CART")
        or getattr(getattr(d, "document_type", None), "name", "") in ("QUOTE", "ONLINE_STORE_CART")
    ]
    if not quotes:
        return pd.DataFrame([{"안내": "비교할 견적서가 없습니다."}])

    target = min(quotes, key=lambda q: (getattr(getattr(q, "total", None), "value", None) or 0)) if quotes else quotes[0]
    doc_items = getattr(target, "items", []) or []
    if not doc_items:
        return pd.DataFrame([{"안내": f"선택된 견적서({getattr(target, 'filename', '견적서')})에서 추출된 세부 품목이 없습니다."}])

    return pd.DataFrame(smart_item_matching(erp_items, doc_items))


def save_to_cache(req_no: str, meta_data: dict[str, Any], raw_data: dict[str, Any]) -> None:
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            pass

    existing_record = cache.get(req_no, {"meta": {}, "raw": {}})
    existing_record["meta"].update(meta_data)
    existing_record["raw"].update(raw_data)
    cache[req_no] = existing_record

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_cache_list() -> list[dict[str, Any]]:
    if not os.path.exists(CACHE_FILE):
        return []
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        result = []
        for k, v in cache.items():
            meta = v.get("meta", {})
            result.append({
                "구매요구번호": meta.get("구매요구번호", k),
                "과제번호": meta.get("과제번호", ""),
                "구매요청액": meta.get("구매요청액", ""),
                "연구책임자정보": meta.get("연구책임자정보", ""),
                "물품담당자": meta.get("물품담당자", ""),
                "업체명": meta.get("업체명", ""),
                "업체담당자정보": meta.get("업체담당자정보", ""),
            })
        return result[::-1]
    except Exception:
        return []


def load_from_cache(req_no: str) -> Optional[dict[str, Any]]:
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get(req_no, {}).get("raw", None)
    except Exception:
        return None
