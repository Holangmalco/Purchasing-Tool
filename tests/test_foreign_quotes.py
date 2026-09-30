"""영문 및 외자 견적서 전용 독립 검증 테스트 스위트.

한국어 정규 견적서(200여 개 회귀 테스트)와 분리하여 영문 서식, IACF/Industry-Academy 수신자 표기,
외화 금액 및 다중 개행 구조만 핀포인트로 고속 검증합니다.
"""
from __future__ import annotations

import datetime
from purchase_verifier.core import (
    DocumentType,
    PurchaseDocument,
    VerificationStatus,
    extract_document,
    verify_documents,
)


def test_reaction_biology_real_foreign_quote():
    """실제 외자 서류(Reaction Biology Corp) 3줄 수신자 및 IACF 인정 테스트."""
    text = (
        "Shipping Address:\n"
        "Shipping and Receiving\n"
        "Reaction Biology Corp.\n"
        "One Great Valley Parkway\n"
        "Suite 2\n"
        "Malvern, PA 19355\n"
        "Tel: 1-877-347-2368\n"
        "QUOTATION\n"
        "TO: Hwangseo Park Quotation #: 20260928-SU-HP-Cell\n"
        "Sejong University Industry-Academy Cooperation\n"
        "Foundation\n"
        "209 Neungdong-ro, Kwangjin-gu Issued By: Kaitlyn Burns\n"
        "05006, Seoul, Korea\n"
        "hspark@sejong.ac.kr Date: 9/28/2026\n"
        "Item DESCRIPTION Total (US $)\n"
        "1 CellTiter-Glo Assay $3,506.40\n"
        "Total: $3,506.40\n"
    )
    doc = extract_document(text, "Reaction_Biology_Quotation.pdf")
    assert doc.document_type == DocumentType.QUOTE
    # 수신자가 Sejong University Industry-Academy Cooperation Foundation 으로 결합되어야 함
    assert "Industry-Academy Cooperation" in doc.recipient.raw
    assert "Foundation" in doc.recipient.raw

    results = verify_documents([doc], purchase_type="용역")
    rec_check = next((r for r in results if r.name == "수신자 산학협력단"), None)
    assert rec_check is not None
    assert rec_check.status == VerificationStatus.PASS
    assert "정상 확인" in rec_check.detail


def test_english_quote_with_iacf_abbreviation():
    """영문 약어 IACF(SJU-IACF) 표기 정상 승인 테스트."""
    text = (
        "QUOTATION / INVOICE\n"
        "Vendor: Thermo Fisher Scientific\n"
        "Bill To: SJU-IACF (Prof. Kim)\n"
        "Date: 2026-09-15\n"
        "Amount: $1,200.00\n"
    )
    doc = extract_document(text, "thermo_quote.pdf")
    results = verify_documents([doc], purchase_type="물품")
    rec_check = next((r for r in results if r.name == "수신자 산학협력단"), None)
    assert rec_check is not None
    assert rec_check.status == VerificationStatus.PASS


def test_english_quote_with_academic_variation():
    """Industry-Academic Cooperation Foundation 변형 표기 정상 승인 테스트."""
    text = (
        "QUOTATION\n"
        "Supplier: Illumina Inc.\n"
        "Customer: Sejong University Industry-Academic Cooperation Foundation\n"
        "Date: 2026-08-20\n"
        "Total: $5,400.00\n"
    )
    doc = extract_document(text, "illumina_quote.pdf")
    results = verify_documents([doc], purchase_type="물품")
    rec_check = next((r for r in results if r.name == "수신자 산학협력단"), None)
    assert rec_check is not None
    assert rec_check.status == VerificationStatus.PASS


def test_english_quote_with_academia_variation():
    """Industry-Academia Cooperation Foundation 변형 표기 정상 승인 테스트."""
    text = (
        "QUOTATION\n"
        "Supplier: New England Biolabs\n"
        "To: Industry-Academia Cooperation Foundation of Sejong University\n"
        "Total: $850.00\n"
    )
    doc = extract_document(text, "neb_quote.pdf")
    results = verify_documents([doc], purchase_type="물품")
    rec_check = next((r for r in results if r.name == "수신자 산학협력단"), None)
    assert rec_check is not None
    assert rec_check.status == VerificationStatus.PASS


def test_english_quote_missing_iacf_is_blocked():
    """영문 견적서라도 산학협력단 표기 없이 본교(Sejong University)만 있으면 반려(BLOCKED)."""
    text = (
        "QUOTATION\n"
        "Vendor: Sigma-Aldrich Co.\n"
        "To: Sejong University (Department of Chemistry)\n"
        "Date: 2026-09-01\n"
        "Total: $2,100.00\n"
    )
    doc = extract_document(text, "sigma_quote.pdf")
    results = verify_documents([doc], purchase_type="물품")
    rec_check = next((r for r in results if r.name == "수신자 산학협력단"), None)
    assert rec_check is not None
    assert rec_check.status == VerificationStatus.BLOCKED
    assert "독립 법인" in rec_check.detail
