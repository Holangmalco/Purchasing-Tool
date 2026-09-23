from purchase_verifier.core import (
    DocumentType, VerificationStatus, extract_document, find_lowest_quote,
    is_valid_business_number, normalize_business_number, verify_documents,
)


def quote(name, total, items="노트북 2 500,000 1,000,000"):
    return extract_document(f"견적서\n상호: {name}\n사업자번호: 220-81-62517\n수신자: 구매팀\n품명 수량 단가 금액\n{items}\n합계\n{total:,}원", f"{name}.pdf")


def test_classifies_document_types():
    assert extract_document("사업자등록증\n사업자번호 220-81-62517", "license.pdf").document_type == DocumentType.BUSINESS_LICENSE
    assert extract_document("통장사본\n예금주: 홍길동\n계좌번호: 123-456", "bank.png").document_type == DocumentType.BANK_COPY
    assert extract_document("공사 확인서\n공사명: 실내공사", "confirm.pdf").document_type == DocumentType.CONSTRUCTION_CONFIRMATION


def test_unknown_does_not_guess_amount():
    doc = extract_document("회의록\n총액 999,999", "note.txt")
    assert doc.document_type == DocumentType.UNKNOWN
    assert doc.total.value is None


def test_subtotal_and_vat_support_line_breaks():
    doc = extract_document("견적서\n소계\n1,000,000원\n부가세: 100,000원", "q.pdf")
    assert doc.subtotal.value == 1_000_000
    assert doc.vat.value == 100_000
    assert doc.total.value == 1_100_000
    assert doc.total.evidence.inferred


def test_business_number_normalization_and_checksum():
    assert normalize_business_number("220 81-62517") == "220-81-62517"
    assert is_valid_business_number("220-81-62517")
    assert not is_valid_business_number("220-81-62518")


def test_own_business_number_is_not_selected():
    doc = extract_document("견적서\n사업자번호: 216-82-12345\n합계: 10,000원", "q.pdf")
    assert doc.business_number.raw == ""
    assert any("자사" in warning for warning in doc.warnings)


def test_lowest_price_confirmed_only_for_comparable_quotes():
    result = find_lowest_quote([quote("A", 1_100_000), quote("B", 900_000)])
    assert result.candidate_filename == "B.pdf"
    assert result.amount == 900_000
    assert result.confirmed


def test_lowest_price_is_not_confirmed_when_items_differ():
    result = find_lowest_quote([quote("A", 1_100_000), quote("B", 900_000, "의자 1 900,000 900,000")])
    assert result.candidate_filename == "B.pdf"
    assert not result.confirmed


def test_comparative_quotes_recognized_not_vendor_conflict():
    first = quote("A", 100_000)
    second = quote("B", 200_000)
    results = verify_documents([first, second], "물품")
    # 다수 견적서가 있을 때 업체명이 다르더라도 충돌(CONFLICT)로 처리하지 않고 비교견적 현황(PASS)으로 확인
    comp_check = [r for r in results if r.name == "비교견적 현황"]
    assert comp_check and comp_check[0].status == VerificationStatus.PASS
    assert not any(r.name == "업체명" and r.status == VerificationStatus.CONFLICT for r in results)


def test_construction_confirmation_missing_warning():
    results = verify_documents([quote("A", 100_000)], "공사")
    missing = [r for r in results if r.name == "공사 확인서"]
    assert missing and missing[0].status == VerificationStatus.MISSING


# 220-81-62517 is a checksum-valid Korean business registration number used for tests.
_VALID_BIZ = "220-81-62517"


def _full_quote(name, total, recipient="산학협력단", contact="담당자: 김담당\n전화: 02-1234-5678\n이메일: a@b.com", items="노트북 2 500,000 1,000,000", biz=_VALID_BIZ, filename=None):
    fname = filename or f"{name}.pdf"
    text = (
        f"견적서\n상호: {name}\n사업자번호: {biz}\n수신자: {recipient}\n{contact}\n"
        f"품명 수량 단가 금액\n{items}\n합계\n{total:,}원"
    )
    return extract_document(text, fname)


def _license(name, biz=_VALID_BIZ):
    return extract_document(f"사업자등록증\n상호: {name}\n사업자번호 {biz}", "license.pdf")


def _bank(holder, account, biz=""):
    body = f"통장사본\n예금주: {holder}\n계좌번호: {account}"
    if biz:
        body += f"\n(사업자번호 참고용 아님: {biz})"
    return extract_document(body, "bank.pdf")


def _confirmation(name, biz, amount):
    return extract_document(f"공사 확인서\n상호: {name}\n사업자번호: {biz}\n공사금액: {amount:,}원", "confirm.pdf")


def test_business_license_matches_lowest_quote():
    q_low = _full_quote("아이티코리아", 1_000_000)
    q_high = _full_quote("베타상사", 1_200_000)
    lic = _license("아이티코리아", _VALID_BIZ)
    results = verify_documents([q_low, q_high, lic], "물품")
    name_results = [r for r in results if r.name == "사업자등록증 업체명"]
    number_results = [r for r in results if r.name == "사업자등록증 사업자번호"]
    assert name_results and name_results[0].status == VerificationStatus.PASS
    assert number_results and number_results[0].status == VerificationStatus.PASS


def test_business_license_conflict_with_lowest_quote():
    q_low = _full_quote("아이티코리아", 1_000_000)
    q_high = _full_quote("베타상사", 1_200_000)
    lic = _license("다른회사", "220-81-62516")
    results = verify_documents([q_low, q_high, lic], "물품")
    name_results = [r for r in results if r.name == "사업자등록증 업체명"]
    number_results = [r for r in results if r.name == "사업자등록증 사업자번호"]
    assert name_results and name_results[0].status == VerificationStatus.CONFLICT
    assert number_results and number_results[0].status == VerificationStatus.CONFLICT


def test_bank_copy_holder_matches_lowest_quote_vendor():
    q_low = _full_quote("아이티코리아", 1_000_000, contact="담당자: 김\n전화: 02-1\n이메일: a@b.com")
    q_high = _full_quote("베타상사", 1_200_000, contact="담당자: 이\n전화: 02-2\n이메일: c@d.com")
    bank = _bank("아이티코리아", "123-456-7890")
    results = verify_documents([q_low, q_high, bank], "물품")
    holder = [r for r in results if r.name == "통장사본 예금주"]
    assert holder and holder[0].status == VerificationStatus.PASS


def test_recipient_without_sanhan_review_and_missing_contact_review():
    q_low = _full_quote("아이티코리아", 1_000_000, recipient="구매팀", contact="")
    q_high = _full_quote("베타상사", 1_200_000, recipient="구매팀", contact="")
    results = verify_documents([q_low, q_high], "물품")
    recipient = [r for r in results if r.name == "수신자 산학협력단"]
    assert recipient and recipient[0].status == VerificationStatus.REVIEW
    assert any(r.name == "담당자" and r.status == VerificationStatus.REVIEW for r in results)
    assert any(r.name == "전화번호" and r.status == VerificationStatus.REVIEW for r in results)
    assert any(r.name == "이메일" and r.status == VerificationStatus.REVIEW for r in results)


def test_construction_confirmation_cross_check_against_lowest_quote():
    q_low = _full_quote("아이티코리아", 1_000_000)
    q_high = _full_quote("베타상사", 1_200_000)
    confirm = _confirmation("아이티코리아", _VALID_BIZ, 1_000_000)
    results = verify_documents([q_low, q_high, confirm], "공사")
    conf_check = [r for r in results if r.name == "공사 확인서"]
    assert conf_check and conf_check[0].status == VerificationStatus.PASS


def test_construction_confirmation_missing_when_not_provided():
    q_low = _full_quote("아이티코리아", 1_000_000)
    q_high = _full_quote("베타상사", 1_200_000)
    results = verify_documents([q_low, q_high], "공사")
    conf_check = [r for r in results if r.name == "공사 확인서"]
    assert conf_check and conf_check[0].status == VerificationStatus.MISSING


def test_exact_policy_biz_number_hyphen_agnostic():
    q = _full_quote("아이티코리아", 1_000_000, biz="2208162517")
    lic = _license("아이티코리아", "220-81-62517")
    results = verify_documents([q, lic], "물품")
    biz_res = [r for r in results if r.name == "사업자등록증 사업자번호"]
    assert biz_res and biz_res[0].status == VerificationStatus.PASS


def test_masked_account_number_returns_review():
    q = _full_quote("아이티코리아", 1_000_000)
    bank = _bank("아이티코리아", "123-***-7890")
    results = verify_documents([q, bank], "물품")
    acc_res = [r for r in results if r.name == "통장사본 계좌번호"]
    assert acc_res and acc_res[0].status == VerificationStatus.REVIEW


def test_multi_word_item_name():
    doc = extract_document("견적서\n품명 수량 단가 금액\n맥북 프로 16인치 M3 Max 2 3,500,000 7,000,000\n합계: 7,000,000원", "q.pdf")
    assert len(doc.items) == 1
    assert doc.items[0].name == "맥북 프로 16인치 M3 Max"
    assert doc.items[0].quantity == 2.0
    assert doc.items[0].unit_price == 3_500_000
    assert doc.items[0].amount == 7_000_000


def test_selected_quote_overrides_lowest():
    q_low = _full_quote("아이티코리아", 1_000_000, biz=_VALID_BIZ, filename="low.pdf")
    q_high = _full_quote("베타상사", 1_200_000, biz="123-45-67891", filename="high.pdf")
    lic_high = _license("베타상사", "123-45-67891")
    # 최저가는 아이티코리아지만 사용자가 베타상사(high.pdf)를 최종 선택한 경우
    results = verify_documents([q_low, q_high, lic_high], "물품", selected_filename="high.pdf")
    name_res = [r for r in results if r.name == "사업자등록증 업체명"]
    num_res = [r for r in results if r.name == "사업자등록증 사업자번호"]
    assert name_res and name_res[0].status == VerificationStatus.PASS
    assert num_res and num_res[0].status == VerificationStatus.PASS


def test_vendor_name_filters_out_sejong_and_sanhan():
    text = (
        "견적서\n"
        "상호 세종대학교 산학협력단 김동순 단장님\n"
        "담당자 민경래 부장\n"
        "상호 ㈜테라텍\n"
        "사업자번호: 106-81-63656\n"
        "합계: 10,780,000원\n"
    )
    doc = extract_document(text, "quote.pdf")
    assert doc.vendor_name.raw == "㈜테라텍"
    assert "세종대학교" in doc.recipient.raw


def test_recipient_sejong_univ_without_sanhan_is_blocked():
    q = _full_quote("아이티코리아", 1_000_000, recipient="세종대학교")
    results = verify_documents([q], "물품")
    recipient_check = [r for r in results if r.name == "수신자 산학협력단"]
    assert recipient_check and recipient_check[0].status == VerificationStatus.BLOCKED
    assert "독립 법인" in recipient_check[0].detail


def test_recipient_sejong_sanhan_is_pass():
    q = _full_quote("아이티코리아", 1_000_000, recipient="세종대학교 산학협력단")
    results = verify_documents([q], "물품")
    recipient_check = [r for r in results if r.name == "수신자 산학협력단"]
    assert recipient_check and recipient_check[0].status == VerificationStatus.PASS


def test_spaced_korean_labels_and_corporate_name():
    license_text = (
        "사 업 자 등 록 증\n"
        "등록번호 : 615-88-01134\n"
        "법인명(단체명) : 주식회사 피앤티링크\n"
        "대 표 자 : 조은용\n"
    )
    doc = extract_document(license_text, "license.pdf")
    assert doc.document_type == DocumentType.BUSINESS_LICENSE
    assert doc.business_number.raw == "615-88-01134"
    assert doc.vendor_name.raw == "주식회사 피앤티링크"
    assert doc.representative.raw == "조은용"


def test_real_teratec_sample_extraction():
    text = (
        "상 호 세종대학교 산학협력단 김동순 단장님\n"
        "담 당 자 민경래 부장\n"
        "연락처 / email 010-3202-6422 / eric.min@teratec.co.kr 등 록 번 호 106-81-63656\n"
        "견적 유효 기간 전 화 / 팩 스 02-701-4422 / 02-3272-2221\n"
        "결 제 조 건 납품 후 익월말 현금결재 대 표 이 사 공 영 삼\n"
        "견 적 담 당 사 업 장 서울특별시 금천구 가산디지털1로 171\n"
        "견 적 일 자 업 태 제조업, 서비스\n"
        "프로젝트명 종 목 컴퓨터 서버, 소프트웨어 개발 및 공급\n"
        "원정 (VAT포함)\n"
        "제조사 수량 공급 단가 공급가 합계 비고\n"
        "각각 다름 1 ₩9,800,000 ₩9,800,000 택배배송\n"
        "소 계 ₩9,800,000\n"
        "부가세 ₩980,000\n"
        "합 계 ₩10,780,000\n"
        "구분 규격\n"
        "시스템 DGX Spark 4TB 사양\n"
        "₩10,780,000\n"
        " 견 적 서\n"
        "상 호 ㈜테라텍\n"
        "2026-09-04\n"
        "DGX\n"
        "2026-09-23\n"
    )
    doc = extract_document(text, "teratec.pdf")
    assert doc.document_type == DocumentType.QUOTE
    assert doc.business_number.raw == "106-81-63656"
    assert doc.vendor_name.raw == "㈜테라텍"
    assert "세종대학교" in doc.recipient.raw
    assert doc.representative.raw == "공 영 삼" or doc.representative.raw == "공영삼"
    assert doc.total.value == 10_780_000


def test_inferred_corporate_name_without_label():
    text = (
        "견 적 서\n"
        "(주)리얼텍브릿지\n"
        "수신 : 세종대학교 산학협력단\n"
        "견적금액: ₩14,245,000(VAT 포함)\n"
    )
    doc = extract_document(text, "quote.pdf")
    assert doc.vendor_name.raw == "(주)리얼텍브릿지"
    assert doc.total.value == 14_245_000


def test_business_number_spaced_digits_and_bogus_number_discarded():
    text = (
        "견 적 서\n"
        "상 호 한국 프라임 성 명\n"
        "1 3 2 - 1 8 - 7 5 4 8 1\n"
        "TEL : 031-574-4129\n"
        "견적번호 : RTB20260910-01\n"
        "합계 1,702,000원\n"
    )
    doc = extract_document(text, "quote.pdf")
    assert doc.business_number.raw == "132-18-75481"
    assert doc.vendor_name.raw == "한국 프라임"
    assert doc.total.value == 1_702_000


def test_spaced_korean_corp_itstone():
    text = (
        "문서번호 : 260907-leesh08\n"
        "견적일자 : 2026년 9월 7일 월요일 주 식 회 사 아 이 티 스 톤 \n"
        "수 신 : 세종대학교산학협력단 1 0 7 - 8 6 - 3 0 4 0 1 \n"
        "합 계 금 액 ₩ 3,850,000 (V.A.T 포함)\n"
    )
    doc = extract_document(text, "itstone.pdf")
    assert doc.vendor_name.raw == "주식회사 아이티스톤"
    assert doc.business_number.raw == "107-86-30401"
    assert doc.quote_date.raw == "2026-09-07"
    assert doc.total.value == 3_850_000


def test_hanul_subtotal_vat_guard_and_vat_inc_amount():
    text = (
        "견적서\n"
        "세종대학교 산학협력단 귀중\n"
        "상 호 ㈜한울 솔루션\n"
        "대 표 신 명 철\n"
        "\\19,900,000 부가세 포함\n"
        "소 계 1,809,091\n"
        "부 가 세 1,809,091\n"
    )
    doc = extract_document(text, "hanul.pdf")
    assert doc.vendor_name.raw == "㈜한울 솔루션"
    assert doc.representative.raw == "신 명 철"
    assert doc.total.value == 19_900_000


def test_server_quotes_lowest_evaluation():
    q1 = _full_quote("한울솔루션", 19_900_000, biz="113-81-81759", filename="hanul.pdf")
    q2 = _full_quote("알엠에스케이알", 21_120_000, biz=_VALID_BIZ, filename="rms.pdf")
    q3 = _full_quote("시네솔", 22_110_000, biz="220-81-62517", filename="synesol.pdf")
    lowest = find_lowest_quote([q1, q2, q3])
    assert lowest.candidate_filename == "hanul.pdf"
    assert lowest.amount == 19_900_000


def test_expiry_5_business_days():
    import datetime
    today = datetime.date.today()
    # 10일 뒤 (충분히 5영업일 이상)
    valid_date = (today + datetime.timedelta(days=14)).isoformat()
    q_valid = extract_document(f"견적서\n상호: 테스트\n사업자번호: {_VALID_BIZ}\n수신자: 산학협력단\n만료일: {valid_date}\n합계: 100,000원", "valid.pdf")
    results = verify_documents([q_valid], "물품")
    exp_res = [r for r in results if r.name == "견적서 유효기간"]
    assert exp_res and exp_res[0].status == VerificationStatus.PASS

    # 1일 뒤 (5영업일 미만 -> BLOCKED)
    short_date = (today + datetime.timedelta(days=1)).isoformat()
    q_short = extract_document(f"견적서\n상호: 테스트\n사업자번호: {_VALID_BIZ}\n수신자: 산학협력단\n만료일: {short_date}\n합계: 100,000원", "short.pdf")
    results_short = verify_documents([q_short], "물품")
    exp_short_res = [r for r in results_short if r.name == "견적서 유효기간"]
    assert exp_short_res and exp_short_res[0].status == VerificationStatus.BLOCKED


def test_multiple_quotes_recipient_check():
    q1 = _full_quote("업체A", 1_000_000, recipient="세종대학교 산학협력단", filename="q1.pdf")
    q2 = _full_quote("업체B", 1_200_000, recipient="세종대학교", filename="q2.pdf")
    results = verify_documents([q1, q2], "물품")
    rec_res = [r for r in results if r.name == "수신자 산학협력단"]
    assert rec_res and rec_res[0].status == VerificationStatus.BLOCKED
    assert "독립 법인" in rec_res[0].detail
    assert "q2.pdf" in rec_res[0].detail


def test_unicode_emdash_biz_and_rep_with_qualifier():
    text = (
        "사 업 자 등 록 증\n"
        "등록번호 : 836-81 ―03763\n"
        "법인명(단체명) : 주식회사 로보시지\n"
        "대표자(대 표유형) : 김현우\n"
    )
    doc = extract_document(text, "license.pdf")
    assert doc.business_number.raw == "836-81-03763"
    assert doc.vendor_name.raw == "주식회사 로보시지"
    assert doc.representative.raw == "김현우"


def test_vendor_prefix_corporate_name():
    text = (
        "견 적 서\n"
        "상호(법인명) (주)와이와이시스템\n"
        "Tel. 010-7373-9305\n"
        "합계 3,499,000\n"
    )
    doc = extract_document(text, "quote.pdf")
    assert doc.vendor_name.raw == "(주)와이와이시스템"
    assert doc.phone.raw == "010-7373-9305"
    assert doc.total.value == 3_499_000


def test_hanja_date_and_recipient_prefix():
    text = (
        "견 적 서\n"
        "업체명 세종대학교 산학협력단\n"
        "일자 : 2026年 09月 09日\n"
        "상호 : (주)브레인어스\n"
        "총 합계 1,360,000\n"
    )
    doc = extract_document(text, "quote.pdf")
    assert doc.recipient.raw == "세종대학교 산학협력단"
    assert doc.quote_date.raw == "2026-09-09"
    assert doc.total.value == 1_360_000


def test_bank_account_holder_inferred_with_rep_and_shop():
    text = (
        "통장사본\n"
        "KB국민은행\n"
        "SWIFT CODE: CZNBKRSEXXX\n"
        "권철규(미르미디어) 님\n"
        "계좌번호 025-21-0508-155\n"
        "저축예금 (사업용계좌)\n"
    )
    doc = extract_document(text, "bank.pdf")
    assert doc.document_type == DocumentType.BANK_COPY
    assert doc.account_holder.raw == "권철규(미르미디어)"
    assert doc.account_number.raw == "025-21-0508-155"


def test_rep_stamp_and_birthdate_cleaned():
    text_quote = "견적서\n상호 미르미디어 성명 권철규 (인)\n합계 7,997,000\n"
    text_license = "사업자등록증\n상호 미르미디어\n성명 : 권철규 생년월일 : 1961 년 02 월 11 일\n등록번호: 106-24-74368\n"
    doc_q = extract_document(text_quote, "quote.pdf")
    doc_l = extract_document(text_license, "license.pdf")
    assert doc_q.representative.raw == "권철규"
    assert doc_l.representative.raw == "권철규"


def test_vat_separate_and_included_amounts():
    text = (
        "QUOTATION SHEET\n"
        "주식회사 엠티스\n"
        "To. 세종대학교 산학협력단 견적제출일 2026-08-24\n"
        "1 . 장비Setting(m-FTIR） 20 개 ₩150,000 ₩3,000,000\n"
        "2 . 성분분석1(m-FTIR) 20 개 ₩440,000 ₩8,800,000\n"
        "3 . 유기성분 정밀분석 （GC-MS） 20 개 ₩480,000 ₩9,600,000\n"
        "합계금액(부가세별도) : ₩21,400,000\n"
        "합계금액 金 이천삼백오십사만 원정 합계금액(부가세포함) : ₩23,540,000\n"
    )
    doc = extract_document(text, "quote.pdf")
    assert doc.recipient.raw == "세종대학교 산학협력단"
    assert doc.total.value == 23_540_000
    assert doc.subtotal.value == 21_400_000
    assert len(doc.items) == 3
    assert doc.items[0].name == "장비Setting(m-FTIR）"


def test_items_ignore_tel_fax_metadata():
    text = (
        "견적서\n"
        "TEL : 031-8019-5010 FAX : 070-7966-5310\n"
        "P.I.C : 강경민 대리 C.P : 010-8009-6693\n"
        "1 장비Setting m-FTIR 20 개 ₩100,000 ₩2,000,000\n"
        "합계금액: ₩2,000,000\n"
    )
    doc = extract_document(text, "quote.pdf")
    assert len(doc.items) == 1
    assert doc.items[0].name == "장비Setting m-FTIR"


def test_online_store_cart_classification():
    apple_txt = "교육 스토어 홈 | 나가기\n장바구니 총액: ₩3,440,000\n결제\nStudio Display\n1 v\n₩3,440,000"
    assert extract_document(apple_txt, "apple_bag.png").document_type == DocumentType.ONLINE_STORE_CART

    samsung_txt = "주문상품\n갤럭시 Z 폴드8 자급제\n1개\n결제 예정 금액 3,057,900 원\n배송지 세종대학교"
    assert extract_document(samsung_txt, "samsung_order.pdf").document_type == DocumentType.ONLINE_STORE_CART


def test_apple_cart_extraction_studio_display():
    text = (
        "교육 스토어 홈 | 나가기\n"
        "장바구니 총액: ₩3,440,000\n"
        "모든 주문에 무료 배송 서비스가 제공됩니다.\n"
        "결제\n"
        "Studio Display - Nano-texture 글래스 - 기울기 및 높이 조절 스탠드\n"
        "1 v\n"
        "₩3,440,000\n"
        "삭제\n"
        "교육 할인\n"
        "Studio Display를 위한 AppleCare+ 추가, ₩209,000\n"
        "추가\n"
        "교육 할인\n"
    )
    doc = extract_document(text, "apple_display.png")
    assert doc.document_type == DocumentType.ONLINE_STORE_CART
    assert "Apple" in doc.vendor_name.raw or "애플" in doc.vendor_name.raw
    assert doc.total.value == 3_440_000
    assert len(doc.items) == 1
    assert "Studio Display" in doc.items[0].name
    assert doc.items[0].amount == 3_440_000


def test_apple_cart_extraction_macbook_air():
    text = (
        "10/04/2026, 15:48 장바구니 - Apple (KR)\n"
        "https://www.apple.com/kr/shop/bag 1/3\n"
        "장바구니 총액: ₩2,990,000\n"
        "결제\n"
        "M5 칩 탑재 미드나이트 MacBook Air\n"
        "15 모델\n"
        "제품 세부 정보 숨기기\n"
        "10코어 CPU, 10코어 GPU, 16코어 Neural Engine\n"
        "32GB 통합 메모리\n"
        "1TB SSD 저장 장치\n"
        "1\n"
        "₩2,990,000\n"
        "삭제\n"
        "소계 ₩2,990,000\n"
        "배송 무료\n"
        "총계 ₩2,990,000\n"
        "₩271,818의 VAT 포함\n"
        "마음에 들 만한 액세서리\n"
        "AirPods Pro 3\n"
        "₩369,000\n"
        "AirTag\n"
        "₩49,000\n"
    )
    doc = extract_document(text, "macbook_air_bag.pdf")
    assert doc.document_type == DocumentType.ONLINE_STORE_CART
    assert doc.total.value == 2_990_000
    assert doc.vat.value == 271_818
    assert len(doc.items) == 1
    assert "MacBook Air" in doc.items[0].name
    assert doc.items[0].amount == 2_990_000


def test_samsung_cart_extraction_discounted_total():
    text = (
        "주문상품\n"
        "갤럭시 Z 폴드8 자급제\n"
        "SM-F971NZKVK00\n"
        "그라파이트/1TB\n"
        "1개\n"
        "3,152,600원 3,057,900 원 v\n"
        "배송 정보\n"
        "주문자 손양주\n"
        "배송지 세종대학교 손*주 010****2162 서울 광진구 능동로 209\n"
        "주문 금액 3,152,600 원\n"
        "할인 금액 -94,700 원\n"
        "결제 예정 금액 3,057,900 원\n"
    )
    doc = extract_document(text, "samsung_zfold.pdf")
    assert doc.document_type == DocumentType.ONLINE_STORE_CART
    assert "삼성" in doc.vendor_name.raw or "Samsung" in doc.vendor_name.raw
    assert doc.total.value == 3_057_900
    assert len(doc.items) == 1
    assert "갤럭시 Z 폴드8" in doc.items[0].name


def test_comparative_quotes_with_online_store_cart():
    foxsoft_text = (
        "견적서\n"
        "상호 : (주)폭스소프트\n"
        "사업자No 123-86-00705\n"
        "수 신 : 세종대학교 산학협력단\n"
        "견적일자: 2026년 09월 15일\n"
        "견적유효기간: 14일\n"
        "품명 수량 단가 금액\n"
        "M6 칩 탑재 Mac mini 1 1,880,000 1,880,000\n"
        "소계 1,880,000\n"
        "VAT 188,000\n"
        "총액(VAT포함) 2,068,000\n"
    )
    apple_text = (
        "장바구니 - Apple (KR)\n"
        "https://www.apple.com/kr/shop/bag\n"
        "장바구니 총액: ₩2,179,000\n"
        "M6 칩 탑재 Mac mini\n"
        "1 v\n"
        "₩2,179,000\n"
    )
    license_text = "사업자등록증\n법인명 : (주)폭스소프트\n등록번호 : 123-86-00705\n대표자 : 홍경철"
    bank_text = "통장사본\n예금주 : (주)폭스소프트\n계좌번호 : 138-910038-27704 하나은행"

    doc_fox = extract_document(foxsoft_text, "폭스소프트_견적서.pdf")
    doc_apple = extract_document(apple_text, "애플_장바구니.png")
    doc_lic = extract_document(license_text, "사업자등록증.pdf")
    doc_bank = extract_document(bank_text, "통장사본.pdf")

    assert doc_fox.document_type == DocumentType.QUOTE
    assert doc_apple.document_type == DocumentType.ONLINE_STORE_CART

    # 1. 최저가 분석: 폭스소프트(2,068,000원) < 애플(2,179,000원)
    lowest = find_lowest_quote([doc_fox, doc_apple])
    assert lowest.candidate_filename == "폭스소프트_견적서.pdf"
    assert lowest.amount == 2_068_000
    assert lowest.confirmed

    # 2. 통합 서류 검증: 폭스소프트가 최저가이므로 사업자등록증/통장사본 정상 대조 통과
    results = verify_documents([doc_fox, doc_apple, doc_lic, doc_bank], "물품")
    statuses = {r.name: r.status for r in results}

    assert statuses.get("비교견적 현황") == VerificationStatus.PASS
    assert statuses.get("수신자 산학협력단") == VerificationStatus.PASS
    assert statuses.get("사업자등록증 업체명") == VerificationStatus.PASS
    assert statuses.get("사업자등록증 사업자번호") == VerificationStatus.PASS
    assert statuses.get("통장사본 예금주") == VerificationStatus.PASS


def test_multi_item_cart_with_added_applecare():
    text = (
        "Pad iPhone watch Vision ArPeds TV WB 218\n"
        "장바구니총액싸5042000\n"
        "iPad Pro 13 Wi-Fi + Cellular 1TB   1v   w4,069,000\n"
        "Nano-texture 글래스 모델 - 스페이스 블랙\n"
        "iPad Pro 13(M5 모델)을 위한 AppleCare+   W259,000\n"
        "Apple Pencil Pro   1v   W195,000\n"
        "iPad Pro 13(M5 모델)용 Magic Keyboard - 한국어 - 블랙   1v   w519,000\n"
        "총계   w5,042,000\n"
        "W458,363VAT\n"
    )
    doc = extract_document(text, "ipad_bundle_cart.png")
    assert doc.document_type == DocumentType.ONLINE_STORE_CART
    assert doc.total.value == 5_042_000
    assert doc.vat.value == 458_363
    assert len(doc.items) == 4
    assert doc.items[0].amount == 4_069_000
    assert doc.items[1].amount == 259_000
    assert doc.items[2].amount == 195_000
    assert doc.items[3].amount == 519_000
    assert sum(it.amount for it in doc.items) == 5_042_000


def test_macbook_with_hardware_specs_and_unadded_applecare():
    text = (
        "M5칩탑재실버MacBookPro   W3,630,000\n"
        "14모델   삭제\n"
        "제품세부정보숨기기\n"
        "하드웨어\n"
        "10코어 CPU, 10코어 GPU, 16코어 Neural Engine\n"
        "24GB 통합 메모리\n"
        "1TB SSD\n"
        "스탠다드 디스플레이\n"
        "70W USB-C 전원 어댑터\n"
        "백라이트 Magic Keyboard (Touch ID 탑재) - 한국어\n"
        "소프트웨어\n"
        "macOS\n"
        "MacBook Pro 14(M5 모델)를 위한 AppleCare+ 추가, ₩429,000   추가\n"
        "소계   w3,630,000\n"
        "총계   w3,630,000\n"
        "330000VAT\n"
    )
    doc = extract_document(text, "macbook_pro.pdf")
    assert doc.document_type == DocumentType.ONLINE_STORE_CART
    assert doc.total.value == 3_630_000
    assert doc.vat.value == 330_000
    assert len(doc.items) == 1
    assert doc.items[0].amount == 3_630_000


def test_optical_components_comparative_quotes_and_lowest_price():
    k2m_text = (
        "견 적 서\n"
        "세종대학교산학협력단 貴中\n"
        "내 역 : Fiber Polarization Controller 외\n"
        "케이투엠플러스\n"
        "금 액 : 一金이천오백칠십일만팔천원정 (₩25,718,000)\n"
        "견적일자 : 2026년 3월 6일 代 表: 金 慶 吾\n"
        "유효기간 : 견적일로부터 30일\n"
        "1 FPC033 EA 1 Thorlabs 1,040,000 1,040,000\n"
        "2 FPC034 EA 1 Thorlabs 1,040,000 1,040,000\n"
        "3 FPC032 EA 2 Thorlabs 1,050,000 2,100,000\n"
        "4 PN1550R5A1 EA 2 Thorlabs 1,650,000 3,300,000\n"
        "5 PN1550R2A1 EA 2 Thorlabs 1,650,000 3,300,000\n"
        "6 PN1550R1A1 EA 2 Thorlabs 1,650,000 3,300,000\n"
        "7 RX INSTR, 10GHz, SM, IR RXM10AF EA 1 Thorlabs 9,300,000 9,300,000\n"
        "금액(Gross Amount) ₩23,380,000\n"
        "부가세(Tax Amount) ₩2,338,000\n"
        "합 계(Total Amount Price) ₩25,718,000\n"
    )
    testlink_text = (
        "견 적 서\n"
        "견적일자 2026/03/06\n"
        "수 신 세종대학교산학협력단\n"
        "사업자번호 214-86-85034\n"
        "회사명/대표 (주)테스트링크 / 김문백\n"
        "유효기간 30일 이내\n"
        "금 액 : 이천일백이십삼만원 정 (￦ 21,230,000원) / VAT포함\n"
        "1 FPC033 1 870,000 870,000 87,000\n"
        "2 FPC034 1 870,000 870,000 87,000\n"
        "3 FPC032 2 880,000 1,760,000 176,000\n"
        "4 PN1550R5A1 2 1,350,000 2,700,000 270,000\n"
        "5 PN1550R2A1 2 1,350,000 2,700,000 270,000\n"
        "6 PN1550R1A1 2 1,300,000 2,600,000 260,000\n"
        "7 RX INSTR, 10GHz, SM, IR RXM10AF 1 7,800,000 7,800,000 780,000\n"
        "공급가액 19,300,000 VAT 1,930,000 합계 21,230,000\n"
    )
    doc_k2m = extract_document(k2m_text, "k2m.pdf")
    doc_tl = extract_document(testlink_text, "testlink.pdf")

    assert doc_k2m.vendor_name.raw == "케이투엠플러스"
    assert doc_k2m.representative.raw == "金 慶 吾"
    assert doc_k2m.total.value == 25_718_000
    assert len(doc_k2m.items) == 7

    assert doc_tl.vendor_name.raw == "(주)테스트링크"
    assert doc_tl.representative.raw == "김문백"
    assert doc_tl.business_number.raw == "214-86-85034"
    assert doc_tl.total.value == 21_230_000
    assert len(doc_tl.items) == 7

    lowest = find_lowest_quote([doc_k2m, doc_tl])
    assert lowest.candidate_filename == "testlink.pdf"
    assert lowest.amount == 21_230_000
    assert lowest.confirmed is True


def test_business_license_and_passbook_extraction():
    lic_text = (
        "사업자등록증\n"
        "등록번호: 214-86-85034\n"
        "법인명6단체명]을   [주]테스트링크\n"
        "자$김문백\n"
        "개업: 2001-07-01\n"
        "사업장 소재지: 경기도 광명시 새빛공원로 67 자이타워\n"
    )
    bank_text = (
        "희망을 키우는 평생은행 IBK기업은행\n"
        "[주테스트링크\n"
        "계좌번호 420-006266-04-023\n"
        "예금종류 기업자유예금\n"
        "중소기업은행\n"
    )
    doc_lic = extract_document(lic_text, "license.jpg")
    doc_bank = extract_document(bank_text, "bank.png")

    assert doc_lic.vendor_name.raw == "(주)테스트링크"
    assert doc_lic.representative.raw == "김문백"
    assert doc_lic.business_number.raw == "214-86-85034"

    assert doc_bank.account_holder.raw == "(주)테스트링크"
    assert doc_bank.account_number.raw == "420-006266-04-023"


def test_daeyeong_rental_quote_extraction():
    text = (
        "見 積 書\n"
        "203 - 87 - 01837\n"
        "세종대학교 산학협력단 貴中 주식회사 대영오에이\n"
        "합 계 금 액 金 일백칠십일만육천 整 ₩1,716,000\n"
        "소 계 ₩1,560,000\n"
        "부가세10% ₩156,000\n"
        "합 계 ₩1,716,000\n"
        "대표전화:(02)454-4901\n"
    )
    doc = extract_document(text, "daeyeong.pdf")
    assert doc.vendor_name.raw == "주식회사 대영오에이"
    assert doc.business_number.raw == "203-87-01837"
    assert doc.total.value == 1_716_000
    assert doc.subtotal.value == 1_560_000
    assert doc.vat.value == 156_000





