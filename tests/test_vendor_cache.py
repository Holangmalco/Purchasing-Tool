"""Unit tests for SQLite-backed Vendor Master Cache module."""
import gc
import os
import tempfile
import pytest

from purchase_verifier.core import (
    DocumentType, Evidence, PurchaseDocument, extract_document, is_valid_business_number,
)
from purchase_verifier.vendor_cache import (
    auto_cache_verified_documents, delete_vendor, get_all_vendors,
    get_vendor_by_biz_num, get_vendor_by_name, init_vendor_db,
    normalize_biz_num, save_vendor,
)

VALID_BIZ_NUM = "220-81-62517"


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    gc.collect()
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def test_normalize_biz_num():
    assert normalize_biz_num("2208162517") == "220-81-62517"
    assert normalize_biz_num("220-81-62517") == "220-81-62517"
    assert normalize_biz_num(" 220 81 62517 ") == "220-81-62517"
    assert normalize_biz_num("123") == "123"


def test_save_and_get_vendor_by_biz_num(temp_db):
    assert is_valid_business_number(VALID_BIZ_NUM)

    ok = save_vendor(
        biz_num=VALID_BIZ_NUM,
        vendor_name="(주)테스트링크",
        representative="김문백",
        bank="국민은행",
        account_num="123-456-7890",
        account_holder="(주)테스트링크",
        phone="02-1234-5678",
        email="test@testlink.co.kr",
        db_path=temp_db,
    )
    assert ok is True

    record = get_vendor_by_biz_num(VALID_BIZ_NUM, db_path=temp_db)
    assert record is not None
    assert record["business_number"] == VALID_BIZ_NUM
    assert record["vendor_name"] == "(주)테스트링크"
    assert record["representative"] == "김문백"
    assert record["phone"] == "02-1234-5678"
    assert record["email"] == "test@testlink.co.kr"
    assert record["verified_count"] == 1
    assert len(record["accounts"]) == 1
    assert record["accounts"][0]["account"] == "123-456-7890"

    # Digits only lookup
    record_raw = get_vendor_by_biz_num("2208162517", db_path=temp_db)
    assert record_raw is not None
    assert record_raw["vendor_name"] == "(주)테스트링크"


def test_get_vendor_by_name_canonical_and_alias(temp_db):
    save_vendor(
        biz_num=VALID_BIZ_NUM,
        vendor_name="(주)테스트링크",
        representative="김문백",
        db_path=temp_db,
    )

    # 1. Exact lookup
    r1 = get_vendor_by_name("(주)테스트링크", db_path=temp_db)
    assert r1 is not None
    assert r1["business_number"] == VALID_BIZ_NUM

    # 2. Canonical lookup (without '(주)' or whitespace)
    r2 = get_vendor_by_name("테스트링크", db_path=temp_db)
    assert r2 is not None
    assert r2["business_number"] == VALID_BIZ_NUM

    r3 = get_vendor_by_name("주식회사 테스트링크", db_path=temp_db)
    assert r3 is not None
    assert r3["business_number"] == VALID_BIZ_NUM


def test_update_vendor_increments_count_and_appends_account(temp_db):
    save_vendor(
        biz_num=VALID_BIZ_NUM,
        vendor_name="(주)테스트링크",
        representative="김문백",
        bank="국민은행",
        account_num="111-222-333",
        db_path=temp_db,
    )

    # Re-save with alternate name alias and new bank account
    save_vendor(
        biz_num=VALID_BIZ_NUM,
        vendor_name="테스트링크코리아",
        bank="신한은행",
        account_num="444-555-666",
        db_path=temp_db,
    )

    updated = get_vendor_by_biz_num(VALID_BIZ_NUM, db_path=temp_db)
    assert updated["verified_count"] == 2
    assert "테스트링크코리아" in updated["aliases"]
    assert len(updated["accounts"]) == 2
    assert updated["representative"] == "김문백"  # Preserved earlier representative


def test_delete_vendor(temp_db):
    save_vendor(biz_num=VALID_BIZ_NUM, vendor_name="(주)테스트링크", db_path=temp_db)
    assert get_vendor_by_biz_num(VALID_BIZ_NUM, db_path=temp_db) is not None

    deleted = delete_vendor(VALID_BIZ_NUM, db_path=temp_db)
    assert deleted is True
    assert get_vendor_by_biz_num(VALID_BIZ_NUM, db_path=temp_db) is None


def test_auto_cache_verified_documents(temp_db):
    docs = [
        PurchaseDocument(
            filename="biz_license.pdf",
            document_type=DocumentType.BUSINESS_LICENSE,
            raw_text="사업자등록증",
            business_number=Evidence(raw=VALID_BIZ_NUM, evidence=VALID_BIZ_NUM),
            vendor_name=Evidence(raw="(주)테스트링크", evidence="(주)테스트링크"),
            representative=Evidence(raw="김문백", evidence="김문백"),
        ),
        PurchaseDocument(
            filename="bank_copy.pdf",
            document_type=DocumentType.BANK_COPY,
            raw_text="통장사본",
            account_number=Evidence(raw="987-654-321", evidence="987-654-321"),
            account_holder=Evidence(raw="(주)테스트링크", evidence="(주)테스트링크"),
        ),
    ]

    cached = auto_cache_verified_documents(docs, db_path=temp_db)
    assert cached is not None
    assert cached["business_number"] == VALID_BIZ_NUM
    assert cached["vendor_name"] == "(주)테스트링크"

    saved = get_vendor_by_biz_num(VALID_BIZ_NUM, db_path=temp_db)
    assert saved is not None
    assert saved["vendor_name"] == "(주)테스트링크"
    assert saved["representative"] == "김문백"
    assert len(saved["accounts"]) == 1
    assert saved["accounts"][0]["account"] == "987-654-321"


def test_extract_document_fallback_with_vendor_cache(temp_db):
    # Seed vendor into cache
    save_vendor(
        biz_num=VALID_BIZ_NUM,
        vendor_name="(주)테스트링크",
        representative="김문백",
        db_path=temp_db,
    )

    # Document OCR where vendor name line is missing, but valid biz num was detected
    noisy_ocr_text = f"""
    견 적 서
    등록번호: {VALID_BIZ_NUM}
    수신: 세종대학교 산학협력단
    합계금액: 일금 일천만원정 (₩10,000,000)
    품명 규격 수량 단가 금액
    연구용서버 1식 1 10,000,000 10,000,000
    """

    doc = extract_document(noisy_ocr_text, filename="noisy_quote.pdf", use_vendor_cache=True, db_path=temp_db)
    assert doc.business_number.raw == VALID_BIZ_NUM
    # Vendor name successfully fallback-inferred from cache!
    assert doc.vendor_name.raw == "(주)테스트링크"
    assert doc.vendor_name.inferred is True
    assert "[거래처 마스터 DB 참조]" in doc.vendor_name.evidence
    # Representative fallback-inferred
    assert doc.representative.raw == "김문백"
    assert doc.representative.inferred is True

    # Reverse test: Document with vendor name but missing business number
    quote_without_biz_num = """
    견 적 서
    상호: (주)테스트링크
    수신: 세종대학교 산학협력단
    합계: 5,000,000원
    품명 수량 단가 금액
    모니터 1 5,000,000 5,000,000
    """
    doc2 = extract_document(quote_without_biz_num, filename="quote2.pdf", use_vendor_cache=True, db_path=temp_db)
    assert doc2.vendor_name.raw == "(주)테스트링크"
    assert doc2.business_number.raw == VALID_BIZ_NUM
    assert doc2.business_number.inferred is True
    assert "[거래처 마스터 DB 참조]" in doc2.business_number.evidence
