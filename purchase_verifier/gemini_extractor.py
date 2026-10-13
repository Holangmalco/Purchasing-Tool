"""High-accuracy Gemini Vision integration with Multi-Key Pool and Smart Backoff."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from purchase_verifier.core import (
    Amount,
    DocumentType,
    Evidence,
    Item,
    PurchaseDocument,
    classify_document,
    extract_document,
)

logger = logging.getLogger(__name__)

# Load local .env
load_dotenv()


class GeminiKeyPool:
    """Manages multiple Gemini API keys with automatic failover and round-robin load balancing."""

    def __init__(self, keys: Optional[list[str]] = None):
        self.keys: list[str] = []
        if keys:
            self.keys = [k.strip() for k in keys if k.strip()]
        else:
            self._load_from_env()
        self._current_idx: int = 0

    def _load_from_env(self) -> None:
        raw_keys = os.getenv("GEMINI_API_KEYS", "")
        if raw_keys:
            self.keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        if not self.keys:
            single = os.getenv("GEMINI_API_KEY", "")
            if single:
                self.keys = [single.strip()]

    def add_keys(self, keys: list[str]) -> None:
        for k in keys:
            k = k.strip()
            if k and k not in self.keys:
                self.keys.append(k)

    def get_key(self) -> Optional[str]:
        if not self.keys:
            self._load_from_env()
        if not self.keys:
            return None
        key = self.keys[self._current_idx % len(self.keys)]
        self._current_idx = (self._current_idx + 1) % len(self.keys)
        return key

    def total_keys(self) -> int:
        return len(self.keys)


_GLOBAL_KEY_POOL = GeminiKeyPool()


def get_key_pool() -> GeminiKeyPool:
    return _GLOBAL_KEY_POOL


def extract_purchase_documents_with_gemini(
    files: list[tuple[str, bytes]],
    custom_key: Optional[str] = None,
) -> list[PurchaseDocument]:
    """Analyzes a purchase dossier (quotes, biz license, passbook, confirmation)
    using Gemini 3.8 Flash in a single coordinated multimodal call.

    Args:
        files: List of (filename, file_bytes) tuples.
        custom_key: Optional custom API key override.

    Returns:
        list[PurchaseDocument]: Fully parsed purchase documents.
    """
    if not files:
        return []

    pool = get_key_pool()
    if custom_key:
        pool.add_keys([custom_key])

    attempts = max(1, pool.total_keys()) * 2
    last_error = None

    for attempt in range(attempts):
        key = custom_key or pool.get_key()
        if not key:
            logger.warning("[Gemini] No API key available. Falling back to local OCR.")
            break

        try:
            client = genai.Client(api_key=key)
            contents: list[Any] = []

            for filename, file_bytes in files:
                contents.append(f"--- 파일명: {filename} ---")
                mime = "application/pdf"
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    mime = f"image/{filename.split('.')[-1].lower()}"
                contents.append(types.Part.from_bytes(data=file_bytes, mime_type=mime))

            prompt = """당신은 대학교 산학협력단 및 공공기관의 구매/회계 감사 전문 인공지능입니다.
첨부된 서류들을 정밀 분석하여 각 파일별로 아래 JSON 배열(Array of Objects) 형식으로 응답하세요:
[
  {
    "filename": "첨부된 파일명 정확히 일치",
    "document_type": "QUOTE | BUSINESS_LICENSE | BANK_COPY | CONSTRUCTION_CONFIRMATION | SPECIFICATION | UNKNOWN",
    "vendor_name": "공급자 상호명 (예: 메인공조, 유진공조)",
    "representative": "대표자 성명 (예: 김윤, 신승현)",
    "biz_number": "사업자등록번호 (000-00-00000 형태)",
    "total_amount": 합계금액(정수 숫자),
    "subtotal": 공급가액(정수 숫자),
    "vat": 부가세(정수 숫자),
    "recipient": "수신처 (예: 세종대학교 산학협력단)",
    "quote_date": "견적일자 또는 발행일자 (YYYY-MM-DD)",
    "expiry_date": "유효기간 만료일 (YYYY-MM-DD 또는 빈문자열)",
    "bank_name": "은행명 (통장사본인 경우)",
    "account_holder": "예금주명 (통장사본인 경우)",
    "account_number": "계좌번호 (통장사본인 경우)",
    "contact_person": "담당자 성명",
    "phone": "전화번호 또는 핸드폰",
    "email": "이메일 주소",
    "items": [
      {
        "name": "품목명",
        "quantity": 수량(숫자 또는 1),
        "unit_price": 단가(숫자),
        "amount": 금액(숫자)
      }
    ]
  }
]
규칙:
1. 상호명과 성명이 표에 나란히 있더라도 절대로 합치지 말고, 상호명(vendor_name)과 대표자(representative)로 분리하세요.
2. 통장사본의 경우 예금주명(account_holder)과 은행명(bank_name), 계좌번호(account_number)를 정확히 추출하세요.
3. 견적서의 모든 세부 품목(items)을 추출하세요.
4. 반드시 유효한 JSON 형식으로만 응답하세요."""

            contents.append(prompt)

            # 2026 current production models
            model_candidates = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
            response = None
            for m_name in model_candidates:
                try:
                    response = client.models.generate_content(
                        model=m_name,
                        contents=contents,
                        config=types.GenerateContentConfig(response_mime_type="application/json"),
                    )
                    if response and response.text:
                        break
                except Exception as m_err:
                    if "NOT_FOUND" in str(m_err) or "404" in str(m_err):
                        continue
                    raise m_err

            if not response or not response.text:
                raise RuntimeError("Empty response from Gemini API")

            raw_json = response.text.strip()
            # Clean possible markdown fence
            if raw_json.startswith("```json"):
                raw_json = raw_json[7:]
            if raw_json.startswith("```"):
                raw_json = raw_json[3:]
            if raw_json.endswith("```"):
                raw_json = raw_json[:-3]

            parsed_list = json.loads(raw_json.strip())
            return _convert_gemini_json_to_documents(parsed_list, files)

        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                logger.warning(f"[Gemini 429 Rate Limit] Key {key[:8]}... exhausted. Switching key and waiting 2s...")
                time.sleep(2.0)
                continue
            else:
                logger.error(f"[Gemini Error] {e}")
                break

    logger.warning(f"[Gemini Fallback] Gemini failed ({last_error}). Falling back to local OCR parsing.")
    from purchase_verifier.ocr import extract_text_from_pdf_data

    fallback_docs: list[PurchaseDocument] = []
    for fname, fbytes in files:
        txt = extract_text_from_pdf_data(fbytes)
        fallback_docs.append(extract_document(txt, fname))
    return fallback_docs


def _convert_gemini_json_to_documents(
    gemini_data: list[dict[str, Any]],
    original_files: list[tuple[str, bytes]],
) -> list[PurchaseDocument]:
    """Converts structured JSON from Gemini into PurchaseDocument objects."""
    file_map = {fname: fbytes for fname, fbytes in original_files}
    docs: list[PurchaseDocument] = []

    type_mapping = {
        "QUOTE": DocumentType.QUOTE,
        "ONLINE_STORE_CART": DocumentType.ONLINE_STORE_CART,
        "BUSINESS_LICENSE": DocumentType.BUSINESS_LICENSE,
        "BANK_COPY": DocumentType.BANK_COPY,
        "CONSTRUCTION_CONFIRMATION": DocumentType.CONSTRUCTION_CONFIRMATION,
        "SPECIFICATION": DocumentType.SUPPORTING_DOC,
        "SUPPORTING_DOC": DocumentType.SUPPORTING_DOC,
        "COST_BREAKDOWN": DocumentType.COST_BREAKDOWN,
        "APPROVAL_MEMO": DocumentType.APPROVAL_MEMO,
        "EXCHANGE_RATE_PROOF": DocumentType.EXCHANGE_RATE_PROOF,
        "UNKNOWN": DocumentType.UNKNOWN,
    }

    for item in gemini_data:
        fname = item.get("filename", "")
        # Fuzzy match filename if slightly transformed
        matched_fname = fname
        if fname not in file_map:
            for cand in file_map:
                if cand in fname or fname in cand:
                    matched_fname = cand
                    break

        raw_type = item.get("document_type", "UNKNOWN").upper()
        doc_type = type_mapping.get(raw_type, DocumentType.UNKNOWN)

        # Parse amounts
        total_val = item.get("total_amount")
        total_int = int(total_val) if (total_val is not None and str(total_val).isdigit()) else None

        subtotal_val = item.get("subtotal")
        subtotal_int = int(subtotal_val) if (subtotal_val is not None and str(subtotal_val).isdigit()) else None

        vat_val = item.get("vat")
        vat_int = int(vat_val) if (vat_val is not None and str(vat_val).isdigit()) else None

        # Parse items
        parsed_items: list[Item] = []
        for it in item.get("items", []):
            i_name = str(it.get("name", "")).strip()
            if not i_name:
                continue
            qty = it.get("quantity")
            qty_f = float(qty) if qty is not None else 1.0
            u_price = it.get("unit_price")
            u_price_i = int(u_price) if u_price is not None else None
            amt = it.get("amount")
            amt_i = int(amt) if amt is not None else None
            parsed_items.append(
                Item(
                    name=i_name,
                    quantity=qty_f,
                    unit_price=u_price_i,
                    amount=amt_i,
                    evidence=Evidence(raw=i_name, confidence=0.98),
                )
            )

        v_name = item.get("vendor_name") or ""
        rep_name = item.get("representative") or ""
        biz_num = item.get("biz_number") or ""
        recip = item.get("recipient") or ""
        q_date = item.get("quote_date") or ""
        exp_date = item.get("expiry_date") or ""
        acc_holder = item.get("account_holder") or ""
        acc_num = item.get("account_number") or ""
        bank_name = item.get("bank_name") or ""
        c_person = item.get("contact_person") or ""
        c_phone = item.get("phone") or ""
        c_email = item.get("email") or ""

        # If BANK_COPY and acc_holder empty but vendor_name exists
        if doc_type == DocumentType.BANK_COPY:
            if not acc_holder and v_name:
                acc_holder = v_name
            if acc_holder and not v_name:
                v_name = acc_holder

        doc = PurchaseDocument(
            filename=matched_fname,
            document_type=doc_type,
            raw_text=json.dumps(item, ensure_ascii=False),
            vendor_name=Evidence(raw=v_name, confidence=0.98) if v_name else Evidence(),
            representative=Evidence(raw=rep_name, confidence=0.98) if rep_name else Evidence(),
            business_number=Evidence(raw=biz_num, confidence=0.98) if biz_num else Evidence(),
            recipient=Evidence(raw=recip, confidence=0.98) if recip else Evidence(),
            quote_date=Evidence(raw=q_date, confidence=0.98) if q_date else Evidence(),
            expiry_date=Evidence(raw=exp_date, confidence=0.98) if exp_date else Evidence(),
            account_holder=Evidence(raw=acc_holder, confidence=0.98) if acc_holder else Evidence(),
            account_number=Evidence(raw=acc_num, confidence=0.98) if acc_num else Evidence(),
            contact_person=Evidence(raw=c_person, confidence=0.98) if c_person else Evidence(),
            phone=Evidence(raw=c_phone, confidence=0.98) if c_phone else Evidence(),
            email=Evidence(raw=c_email, confidence=0.98) if c_email else Evidence(),
            total=Amount(value=total_int, evidence=Evidence(raw=str(total_int) if total_int else "", confidence=0.98)),
            subtotal=Amount(value=subtotal_int, evidence=Evidence(raw=str(subtotal_int) if subtotal_int else "", confidence=0.98)),
            vat=Amount(value=vat_int, evidence=Evidence(raw=str(vat_int) if vat_int else "", confidence=0.98)),
            items=parsed_items,
            warnings=[],
        )
        docs.append(doc)

    return docs
