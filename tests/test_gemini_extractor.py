"""Unit tests for Gemini extractor and key pooling."""
import os
import pytest
from purchase_verifier.gemini_extractor import GeminiKeyPool, _convert_gemini_json_to_documents
from purchase_verifier.core import DocumentType


def test_gemini_key_pool_rotation():
    pool = GeminiKeyPool(["key1", "key2", "key3"])
    assert pool.total_keys() == 3
    assert pool.get_key() == "key1"
    assert pool.get_key() == "key2"
    assert pool.get_key() == "key3"
    assert pool.get_key() == "key1"


def test_convert_gemini_json_to_documents():
    gemini_data = [
        {
            "filename": "quote1.pdf",
            "document_type": "QUOTE",
            "vendor_name": "유진공조",
            "representative": "신승현",
            "biz_number": "267-68-00049",
            "total_amount": 2966400,
            "subtotal": 2696727,
            "vat": 269673,
            "recipient": "세종대학교 산학협력단",
            "quote_date": "2026-10-12",
            "items": [
                {"name": "기본설치비", "quantity": 1, "unit_price": 330000, "amount": 330000}
            ]
        },
        {
            "filename": "bank.pdf",
            "document_type": "BANK_COPY",
            "vendor_name": "메인공조",
            "account_holder": "김윤(메인공조)",
            "account_number": "933901-01-176638",
            "bank_name": "KB국민은행"
        }
    ]
    files = [("quote1.pdf", b"pdf1"), ("bank.pdf", b"pdf2")]
    docs = _convert_gemini_json_to_documents(gemini_data, files)

    assert len(docs) == 2
    assert docs[0].filename == "quote1.pdf"
    assert docs[0].document_type == DocumentType.QUOTE
    assert docs[0].vendor_name.raw == "유진공조"
    assert docs[0].representative.raw == "신승현"
    assert docs[0].business_number.raw == "267-68-00049"
    assert docs[0].total.value == 2966400
    assert len(docs[0].items) == 1
    assert docs[0].items[0].name == "기본설치비"

    assert docs[1].filename == "bank.pdf"
    assert docs[1].document_type == DocumentType.BANK_COPY
    assert docs[1].account_holder.raw == "김윤(메인공조)"
    assert docs[1].account_number.raw == "933901-01-176638"
