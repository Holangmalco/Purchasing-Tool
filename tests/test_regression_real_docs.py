import json
import os
import pytest
from purchase_verifier.core import extract_document

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "golden_baseline.json")


def load_baseline():
    if not os.path.exists(BASELINE_PATH):
        pytest.skip(f"Golden baseline not found: {BASELINE_PATH}")
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


BASELINE = load_baseline() if os.path.exists(BASELINE_PATH) else {}


@pytest.mark.parametrize("filename, expected", list(BASELINE.items()))
def test_real_doc_regression(filename, expected):
    """
    사용자가 직접 업로드했던 48개 실제 실물 문서 원본 텍스트에 대한 전수 회귀 검증(Regression Test).
    파서 로직 수정 시 과거 문서에서 1원이나 1글자라도 어긋나면 즉시 테스트가 실패합니다.
    """
    raw_text = expected.get("raw_text", "")
    doc = extract_document(raw_text, filename, use_vendor_cache=False)

    doc_type_str = doc.document_type.value if hasattr(doc.document_type, "value") else str(doc.document_type)
    vendor_str = doc.vendor_name.raw if hasattr(doc.vendor_name, "raw") else str(doc.vendor_name)
    total_val = doc.total.value if hasattr(doc.total, "value") else None
    biz_num_str = doc.business_number.raw if hasattr(doc.business_number, "raw") else str(doc.business_number)
    acct_num_str = doc.account_number.raw if hasattr(doc.account_number, "raw") else str(doc.account_number)

    assert doc_type_str == expected["document_type"], f"[{filename}] 문서 종류 불일치: {doc_type_str} != {expected['document_type']}"
    assert vendor_str == expected["vendor_name"], f"[{filename}] 상호명 불일치: '{vendor_str}' != '{expected['vendor_name']}'"
    assert total_val == expected["total_amount"], f"[{filename}] 총금액 불일치: {total_val} != {expected['total_amount']}"
    assert biz_num_str == expected["business_number"], f"[{filename}] 사업자번호 불일치: '{biz_num_str}' != '{expected['business_number']}'"
    assert acct_num_str == expected["account_number"], f"[{filename}] 계좌번호 불일치: '{acct_num_str}' != '{expected['account_number']}'"
