import datetime
from purchase_verifier.erp_parser import (
    build_smart_matrix,
    calculate_business_days,
    extract_base_info,
    get_accurate_total,
    parse_goods_erp,
    parse_service_erp,
    generate_opinion,
    smart_item_matching,
)
from purchase_verifier.core import Amount, DocumentType, Item, PurchaseDocument


def test_calculate_business_days():
    # 2026-09-14 (월) ~ 2026-09-18 (금) = 4 영업일
    start = datetime.date(2026, 9, 14)
    end = datetime.date(2026, 9, 18)
    assert calculate_business_days(start, end) == 4

    # 2026-09-14 (월) ~ 2026-09-21 (다음 주 월) = 5 영업일
    end2 = datetime.date(2026, 9, 21)
    assert calculate_business_days(start, end2) == 5

    # 과거 날짜
    past = datetime.date(2026, 9, 11)
    assert calculate_business_days(start, past) == -1


def test_extract_base_info_with_prj_end():
    sample_req = (
        "구매요구번호: 202609090001\n"
        "과제번호: 2024-1234-01\n"
        "연구책임자정보 홍길동 / 교수 / 컴퓨터공학과 / 01012345678 / hong@sejong.ac.kr 연차연구기간 2026-03-01 ~ 2026-10-31\n"
        "합계액 1,500,000"
    )
    info = extract_base_info(sample_req)
    assert info["req_no"] == "202609090001"
    assert info["p_no"] == "2024-1234-01"
    assert info["pi_name"] == "홍길동"
    assert info["pi_dept"] == "컴퓨터공학과"
    assert info["prj_end"] == "2026-10-31"


def test_get_accurate_total():
    sample_text = "물품 합계액: 2,500,000원 (VAT포함)"
    total = get_accurate_total(sample_text, "합계액")
    assert total == 2_500_000


def test_parse_goods_erp():
    sample_row = "1\t맥북 프로 16인치\t스페이스그레이\tM3 Max\t1\tea\t3,000,000\t3,000,000\t비품\t예\t홍길동 010-1234-5678"
    items, is_valid = parse_goods_erp(sample_row)
    assert is_valid
    assert len(items) == 1
    assert "맥북" in items[0]["name"]
    assert items[0]["qty"] == 1
    assert items[0]["price"] == 3_000_000


def test_smart_item_matching():
    erp_items = [{"name": "맥북 프로 16 M3", "qty": 1, "price": 3_500_000}]
    doc_items = [Item(name="MacBook Pro 16형 M3", quantity=1, unit_price=3_500_000, amount=3_500_000)]
    matched = smart_item_matching(erp_items, doc_items)
    assert len(matched) == 1
    assert matched[0]["최종 판정"] == "✅ 매칭 성공"
    assert matched[0]["수량 검증"] == "✅ 일치"
    assert matched[0]["금액 검증"] == "✅ 일치"


def test_build_smart_matrix():
    erp_items = [{"name": "모니터 27인치", "qty": 2, "price": 600_000}]
    doc = PurchaseDocument(
        filename="견적서_알파.pdf",
        document_type=DocumentType.QUOTE,
        raw_text="견적서",
        total=Amount(value=600_000),
        items=[Item(name="27인치 4K 모니터", quantity=2, unit_price=300_000, amount=600_000)],
    )
    df = build_smart_matrix([doc], erp_items)
    assert len(df) == 1
    assert df.iloc[0]["최종 판정"] == "✅ 매칭 성공"
    assert df.iloc[0]["수량 검증"] == "✅ 일치"
    assert df.iloc[0]["금액 검증"] == "✅ 일치"


def test_generate_opinion_category_and_thresholds():
    # 1. 30만원 이상 + 등록여부 '아니오' -> 물품등록여부 변경필요
    items_over_300k = [{
        "name": "고성능 그래픽카드",
        "unit": "u",
        "unit_price": 500_000,
        "price": 500_000,
        "category": "비품",
        "reg_status": "아니오",
        "contact": "홍길동",
    }]
    op = generate_opinion("물품", "비교견적", 500_000, items_over_300k, vendor_contact="김담당 010-1234-5678")
    assert "물품분류구분 : 비품" in op
    assert "물품등록여부 변경필요" in op

    # 2. 15만원 이하 + 등록여부 '예' -> 물품등록여부 변경필요
    items_under_150k = [{
        "name": "USB 케이블",
        "unit": "u",
        "unit_price": 50_000,
        "price": 50_000,
        "category": "소모품",
        "reg_status": "예",
        "contact": "홍길동",
    }]
    op2 = generate_opinion("물품", "비교견적", 50_000, items_under_150k, vendor_contact="김담당 010-1234-5678")
    assert "물품등록여부 변경필요" in op2
    assert "물품분류구분 : 소모품" in op2

    # 3. 공사인데 단위가 식이 아님 -> 단위 '식'으로 수정 필요
    items_const = [{
        "name": "실내 칸막이 공사",
        "unit": "개",
        "unit_price": 3_000_000,
        "price": 3_000_000,
        "category": "공사",
        "reg_status": "N/A",
        "contact": "홍길동",
    }]
    op3 = generate_opinion("공사", "수의계약", 3_000_000, items_const, vendor_contact="김담당 010-1234-5678")
    assert "단위 '식'으로 수정 필요" in op3
    assert "물품분류구분 : 공사" in op3

    # 4. 업체담당자 정보 없음 -> 업체담당자 정보확인필요
    op4 = generate_opinion("물품", "비교견적", 100_000, items_under_150k, vendor_contact="")
    assert "업체담당자 정보확인필요" in op4

    # 5. 추정가격 2천만원(VAT포함 2,200만원) 기준 테스트
    # 21,230,000원 (2,200만원 이하) -> 경고 미발생 (소액 비교견적 가능 범위)
    op_under_22m = generate_opinion("물품", "비교견적", 21_230_000, items_under_150k, vendor_contact="김담당 010-1234-5678")
    assert "추정가격 2천만원(VAT포함 2,200만원) 초과" not in op_under_22m

    # 23,000,000원 (2,200만원 초과) -> 수의계약사유서/전자입찰 확인 필요 경고 발생
    op_over_22m = generate_opinion("물품", "비교견적", 23_000_000, items_under_150k, vendor_contact="김담당 010-1234-5678")
    assert "추정가격 2천만원(VAT포함 2,200만원) 초과 건: 수의계약 사유서(또는 전자입찰) 대상 여부 확인 필요" in op_over_22m


def test_sejong_univ_specific_registration_rules():
    # A. 집기 (책상): 15만원 이상 -> 비품(등록대상), '아니오'면 변경필요
    desk_over_150k = [{
        "name": "연구용 실험대 책상",
        "unit": "u",
        "unit_price": 200_000,
        "price": 200_000,
        "category": "집기",
        "reg_status": "아니오",
        "contact": "홍길동",
    }]
    op = generate_opinion("물품", "비교견적", 200_000, desk_over_150k, vendor_contact="담당자 010-1234-5678")
    assert "물품분류구분 : 비품" in op  # 집기가 아니라 '비품'으로 뭉뚱그려 표기!
    assert "물품등록여부 변경필요" in op

    # 집기 (의자): 15만원 미만 -> 소모품(등록비대상), '예'면 변경필요
    chair_under_150k = [{
        "name": "사무용 의자",
        "unit": "u",
        "unit_price": 100_000,
        "price": 100_000,
        "category": "집기",
        "reg_status": "예",
        "contact": "홍길동",
    }]
    op2 = generate_opinion("물품", "비교견적", 100_000, chair_under_150k, vendor_contact="담당자 010-1234-5678")
    assert "물품분류구분 : 소모품" in op2
    assert "물품등록여부 변경필요" in op2

    # B. 기계기구 (분석기): 30만원 미만(25만원) -> 소모품(등록비대상), '예'면 변경필요
    device_under_300k = [{
        "name": "소형 측정 센서기",
        "unit": "u",
        "unit_price": 250_000,
        "price": 250_000,
        "category": "기계기구",
        "reg_status": "예",
        "contact": "홍길동",
    }]
    op3 = generate_opinion("물품", "비교견적", 250_000, device_under_300k, vendor_contact="담당자 010-1234-5678")
    assert "물품분류구분 : 소모품" in op3
    assert "물품등록여부 변경필요" in op3

    # C. 소프트웨어: 1년 이상 사용 & 200만원 미만(150만원) -> 소모품, '예'면 변경필요
    sw_under_2m = [{
        "name": "통계분석 소프트웨어 라이선스",
        "unit": "u",
        "unit_price": 1_500_000,
        "price": 1_500_000,
        "category": "소프트웨어",
        "reg_status": "예",
        "contact": "홍길동",
    }]
    op4 = generate_opinion("물품", "비교견적", 1_500_000, sw_under_2m, vendor_contact="담당자 010-1234-5678")
    assert "물품분류구분 : 소모품" in op4
    assert "물품등록여부 변경필요" in op4

    # D. 소프트웨어: 200만원 이상(300만원) & 영구 -> 비품, '아니오'면 변경필요
    sw_over_2m = [{
        "name": "시뮬레이션 소프트웨어 영구 라이선스",
        "unit": "u",
        "unit_price": 3_000_000,
        "price": 3_000_000,
        "category": "소프트웨어",
        "reg_status": "아니오",
        "contact": "홍길동",
    }]
    op5 = generate_opinion("물품", "비교견적", 3_000_000, sw_over_2m, vendor_contact="담당자 010-1234-5678")
    assert "물품분류구분 : 비품" in op5
    assert "물품등록여부 변경필요" in op5


def test_prototype_fabrication_is_goods_not_service():
    """시제작품/시제품/로봇의족 제작 건은 용역이 아닌 물품구매(비품, 등록대상 예)로 처리함을 검증."""
    prototype_items = [{
        "name": "통합구동모듈 기반의 대퇴 로봇의족 제작",
        "unit": "u",
        "unit_price": 19_448_000,
        "price": 19_448_000,
        "category": "비품",
        "reg_status": "예",
        "contact": "정기수",
    }]
    opinion = generate_opinion("물품", "비교견적", 19_448_000, prototype_items, vendor_contact="정기수 010-8240-8455")
    assert "물품분류구분 : 비품" in opinion
    assert "물품등록여부 변경필요" not in opinion
    assert "단위 '식'으로 수정 필요" not in opinion



