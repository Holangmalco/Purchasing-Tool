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


def test_cross_document_conflict_is_reported():
    first = quote("A", 100_000)
    second = quote("B", 200_000)
    results = verify_documents([first, second], "물품")
    assert any(r.name == "업체명" and r.status == VerificationStatus.CONFLICT for r in results)


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
    amount = [r for r in results if r.name == "공사 확인서 금액"]
    name = [r for r in results if r.name == "공사 확인서 업체명"]
    number = [r for r in results if r.name == "공사 확인서 사업자번호"]
    assert amount and amount[0].status == VerificationStatus.PASS
    assert name and name[0].status == VerificationStatus.PASS
    assert number and number[0].status == VerificationStatus.PASS


def test_construction_confirmation_amount_conflict():
    q_low = _full_quote("아이티코리아", 1_000_000)
    q_high = _full_quote("베타상사", 1_200_000)
    confirm = _confirmation("아이티코리아", _VALID_BIZ, 900_000)
    results = verify_documents([q_low, q_high, confirm], "공사")
    amount = [r for r in results if r.name == "공사 확인서 금액"]
    assert amount and amount[0].status == VerificationStatus.CONFLICT
