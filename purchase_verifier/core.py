"""Pure Python rules for local OCR purchase-document verification."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Callable, Optional


class DocumentType(str, Enum):
    QUOTE = "QUOTE"
    BUSINESS_LICENSE = "BUSINESS_LICENSE"
    BANK_COPY = "BANK_COPY"
    CONSTRUCTION_CONFIRMATION = "CONSTRUCTION_CONFIRMATION"
    UNKNOWN = "UNKNOWN"


class VerificationStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    REVIEW = "REVIEW"
    MISSING = "MISSING"
    CONFLICT = "CONFLICT"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class Evidence:
    raw: str = ""
    evidence: str = ""
    confidence: float = 0.0
    inferred: bool = False


@dataclass(frozen=True)
class Amount:
    value: Optional[int] = None
    evidence: Evidence = field(default_factory=Evidence)
    label: str = ""


@dataclass(frozen=True)
class Item:
    name: str
    quantity: Optional[float] = None
    unit_price: Optional[int] = None
    amount: Optional[int] = None
    evidence: Evidence = field(default_factory=Evidence)


@dataclass
class PurchaseDocument:
    filename: str
    document_type: DocumentType
    raw_text: str
    vendor_name: Evidence = field(default_factory=Evidence)
    business_number: Evidence = field(default_factory=Evidence)
    recipient: Evidence = field(default_factory=Evidence)
    contact_person: Evidence = field(default_factory=Evidence)
    phone: Evidence = field(default_factory=Evidence)
    email: Evidence = field(default_factory=Evidence)
    account_holder: Evidence = field(default_factory=Evidence)
    account_number: Evidence = field(default_factory=Evidence)
    quote_date: Evidence = field(default_factory=Evidence)
    expiry_date: Evidence = field(default_factory=Evidence)
    total: Amount = field(default_factory=Amount)
    subtotal: Amount = field(default_factory=Amount)
    vat: Amount = field(default_factory=Amount)
    items: list[Item] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: VerificationStatus
    detail: str


@dataclass(frozen=True)
class LowestPriceResult:
    candidate_filename: Optional[str]
    amount: Optional[int]
    confirmed: bool
    reason: str


_BIZ_RE = re.compile(r"(?<!\d)(\d[\s-]?\d[\s-]?\d[\s-]?[-\s]?\d[\s-]?\d[\s-]?[-\s]?\d[\s-]?\d[\s-]?\d[\s-]?\d[\s-]?\d)(?!\d)")
_DATE_RE = re.compile(r"(20\d{2})\s*[./년-]\s*(\d{1,2})\s*[./월-]\s*(\d{1,2})")
_AMOUNT_RE = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})+|\d{4,})(?:\s*원)?(?!\d)")


def _clean_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_name(value: str) -> str:
    return re.sub(r"[^0-9가-힣A-Za-z]", "", value).lower()


def normalize_business_number(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}" if len(digits) == 10 else digits


def is_valid_business_number(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if len(digits) != 10 or len(set(digits)) == 1:
        return False
    weights = (1, 3, 7, 1, 3, 7, 1, 3, 5)
    total = sum(int(digits[i]) * weights[i] for i in range(9))
    total += (int(digits[8]) * 5) // 10
    return (10 - (total % 10)) % 10 == int(digits[9])


def _evidence(raw: str, source: str, confidence: float, inferred: bool = False) -> Evidence:
    return Evidence(raw=raw, evidence=source, confidence=confidence, inferred=inferred)


def classify_document(text: str, filename: str = "") -> DocumentType:
    haystack = f"{filename} {text}".lower()
    scores = {kind: 0 for kind in DocumentType}
    keywords = {
        DocumentType.QUOTE: ("견적", "quotation", "quote", "공급가액", "합계", "subtotal", "단가"),
        DocumentType.BUSINESS_LICENSE: ("사업자등록증", "사업자 등록증", "사업자번호", "대표자", "업태"),
        DocumentType.BANK_COPY: ("통장사본", "통장 사본", "예금주", "계좌번호", "은행"),
        DocumentType.CONSTRUCTION_CONFIRMATION: ("공사확인서", "공사 확인서", "준공", "공사명", "시공"),
    }
    for kind, terms in keywords.items():
        scores[kind] = sum(haystack.count(term) for term in terms)
    explicit_titles = {
        DocumentType.BUSINESS_LICENSE: ("사업자등록증", "사업자 등록증"),
        DocumentType.BANK_COPY: ("통장사본", "통장 사본"),
        DocumentType.CONSTRUCTION_CONFIRMATION: ("공사확인서", "공사 확인서"),
        DocumentType.QUOTE: ("견적서", "견적", "quotation"),
    }
    title_hits = {kind: any(term in haystack for term in terms) for kind, terms in explicit_titles.items()}
    if sum(title_hits.values()) == 1:
        for kind, hit in title_hits.items():
            if hit:
                return kind
    best = max(scores, key=lambda kind: scores[kind])
    return best if scores[best] > 0 else DocumentType.UNKNOWN


def _labeled(text: str, labels: tuple[str, ...], value_pattern: str = r"[^\n:：]{1,100}") -> Evidence:
    label = "|".join(re.escape(x) for x in labels)
    match = re.search(rf"(?:{label})\s*[:：]?\s*({value_pattern})", text, re.IGNORECASE)
    if match:
        value = _clean_spaces(match.group(1)).strip("|,;")
        return _evidence(value, match.group(0).strip(), 0.92)
    return Evidence()


def _amount_after_label(text: str, labels: tuple[str, ...]) -> Amount:
    label = "|".join(re.escape(x) for x in labels)
    match = re.search(rf"(?:{label})[^0-9]{{0,80}}({_AMOUNT_RE.pattern[1:-1]})", text, re.IGNORECASE)
    if not match:
        return Amount()
    raw = match.group(1)
    return Amount(int(raw.replace(",", "")), _evidence(raw, match.group(0).strip(), 0.95), labels[0])


def _date_evidence(text: str, labels: tuple[str, ...]) -> Evidence:
    label = "|".join(re.escape(x) for x in labels)
    match = re.search(rf"(?:{label})[^\d]*(20\d{{2}}\s*[./년-]\s*\d{{1,2}}\s*[./월-]\s*\d{{1,2}})", text, re.IGNORECASE)
    if not match:
        return Evidence()
    return _evidence(match.group(1), match.group(0).strip(), 0.95)


def _parse_date(value: str) -> Optional[date]:
    match = _DATE_RE.search(value)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def calculate_expiry(text: str, quote_date: Evidence) -> Evidence:
    explicit = _date_evidence(text, ("만료일", "유효만료일", "valid until", "expiry"))
    if explicit.raw:
        return explicit
    validity = re.search(r"(?:유효기간|견적유효기간|validity)[^0-9]{0,20}(\d{1,3})\s*(?:일|days?)", text, re.IGNORECASE)
    start = _parse_date(quote_date.raw)
    if validity and start:
        expiry = start + timedelta(days=int(validity.group(1)))
        raw = expiry.isoformat()
        return _evidence(raw, validity.group(0), 0.82, True)
    return Evidence()


def _extract_items(text: str) -> list[Item]:
    items: list[Item] = []
    for line in text.splitlines():
        line_clean = _clean_spaces(line)
        if not line_clean or re.search(r"(품명|품목|description).*(수량|단가|금액)", line_clean, re.I):
            continue
        numeric_tokens = re.findall(r"(?<!\d)\d+(?:,\d{3})*(?!\d)", line_clean)
        numbers = [int(x.replace(",", "")) for x in numeric_tokens]
        if len(numbers) >= 3:
            name = re.split(r"\s+", line_clean)[0]
            quantity = float(numbers[-3])
            items.append(Item(name=name, quantity=quantity, unit_price=numbers[-2], amount=numbers[-1], evidence=_evidence(line_clean, line_clean, 0.72)))
    return items


def extract_document(text: str, filename: str = "") -> PurchaseDocument:
    kind = classify_document(text, filename)
    document = PurchaseDocument(filename=filename, document_type=kind, raw_text=text)
    if kind != DocumentType.QUOTE and kind != DocumentType.BUSINESS_LICENSE and kind != DocumentType.BANK_COPY and kind != DocumentType.CONSTRUCTION_CONFIRMATION:
        document.warnings.append("문서 유형을 확정하지 못했습니다. 원문을 검토하세요.")
    own_prefix = re.sub(r"\D", "", my_business_prefix())
    business_candidates = [m for m in _BIZ_RE.finditer(text) if not re.sub(r"\D", "", m.group(1)).startswith(own_prefix)]
    if business_candidates:
        biz_match = business_candidates[0]
        normalized = normalize_business_number(biz_match.group(1))
        document.business_number = _evidence(normalized, biz_match.group(0), 0.9 if is_valid_business_number(normalized) else 0.55)
        if not is_valid_business_number(normalized):
            document.warnings.append("사업자번호 형식은 발견했지만 체크섬이 유효하지 않습니다.")
    elif _BIZ_RE.search(text):
        document.warnings.append("자사 사업자번호만 발견되어 업체 번호로 선택하지 않았습니다.")
    document.vendor_name = _labeled(text, ("상호", "업체명", "회사명", "공급자", "시공사", "시공자", "공사업체"))
    document.recipient = _labeled(text, ("수신자", "수신", "귀하", "납품처"))
    document.contact_person = _labeled(text, ("담당자", "담당"))
    document.phone = _labeled(text, ("전화", "연락처", "휴대폰"), r"[0-9() .+-]{8,}")
    document.email = _labeled(text, ("이메일", "email"), r"[\w.+-]+@[\w.-]+")
    document.account_holder = _labeled(text, ("예금주", "예금주명"))
    document.account_number = _labeled(text, ("계좌번호", "계좌"), r"[0-9 -]{6,}")
    document.quote_date = _date_evidence(text, ("견적일자", "견적일", "작성일", "발행일"))
    document.expiry_date = calculate_expiry(text, document.quote_date)
    if kind in (DocumentType.QUOTE, DocumentType.CONSTRUCTION_CONFIRMATION):
        document.total = _amount_after_label(text, ("합계", "총계", "총액", "공급대가", "결제금액", "공사금액", "Total"))
        document.vat = _amount_after_label(text, ("부가세", "부가가치세", "세액", "VAT"))
        document.subtotal = _amount_after_label(text, ("공급가액", "공급가", "Subtotal", "소계"))
        document.items = _extract_items(text)
        if document.total.value is None and document.subtotal.value is not None and document.vat.value is not None:
            value = document.subtotal.value + document.vat.value
            document.total = Amount(value, _evidence(str(value), "공급가액+부가세 산술 추론", 0.65, True), "추론 합계")
        if document.total.value is None:
            document.warnings.append("라벨 근거가 있는 합계 금액을 찾지 못했습니다. 최대 숫자를 합계로 확정하지 않았습니다.")
        if not document.items and kind == DocumentType.QUOTE:
            document.warnings.append("비교 가능한 품목 행을 찾지 못했습니다.")
    return document


def _value(field: Evidence) -> str:
    return _clean_spaces(field.raw)


def compare_documents(documents: list[PurchaseDocument]) -> list[CheckResult]:
    if not documents:
        return [CheckResult("documents", VerificationStatus.MISSING, "문서가 없습니다.")]
    quotes = [doc for doc in documents if doc.document_type == DocumentType.QUOTE]
    results: list[CheckResult] = []
    fields = (("업체명", [_value(d.vendor_name) for d in quotes if d.vendor_name.raw], normalize_name), ("사업자번호", [_value(d.business_number) for d in quotes if d.business_number.raw], lambda value: re.sub(r"\D", "", value)), ("계좌번호", [_value(d.account_number) for d in quotes if d.account_number.raw], lambda value: re.sub(r"\D", "", value)), ("수신자", [_value(d.recipient) for d in quotes if d.recipient.raw], normalize_name))
    for label, values, normalizer in fields:
        unique = {normalizer(v) for v in values}
        results.append(CheckResult(label, VerificationStatus.PASS if len(unique) <= 1 and values else VerificationStatus.CONFLICT if len(unique) > 1 else VerificationStatus.REVIEW, "일치" if len(unique) <= 1 and values else "문서 간 값이 다릅니다." if len(unique) > 1 else "근거가 부족합니다."))
    if not quotes:
        results.append(CheckResult("견적서", VerificationStatus.MISSING, "견적서가 없습니다."))
    else:
        item_sets = [tuple(sorted(normalize_name(item.name) for item in q.items)) for q in quotes]
        results.append(CheckResult("품목 구성", VerificationStatus.PASS if len(set(item_sets)) == 1 and item_sets[0] else VerificationStatus.REVIEW, "비교 가능" if len(set(item_sets)) == 1 and item_sets[0] else "품목 구성이 다르거나 불완전합니다."))
    return results


def find_lowest_quote(documents: list[PurchaseDocument]) -> LowestPriceResult:
    quotes = [d for d in documents if d.document_type == DocumentType.QUOTE and d.total.value is not None]
    if not quotes:
        return LowestPriceResult(None, None, False, "확정 합계가 있는 견적서가 없습니다.")
    comparable = all(q.items for q in quotes) and len({tuple(sorted(normalize_name(i.name) for i in q.items)) for q in quotes}) == 1
    candidate = min(quotes, key=lambda q: q.total.value or 0)
    confirmed = comparable and all(not q.total.evidence.inferred for q in quotes)
    reason = "확정 합계와 동일 품목 구성" if confirmed else "후보만 표시: 합계 또는 품목 구성이 불확실합니다."
    return LowestPriceResult(candidate.filename, candidate.total.value, confirmed, reason)


def _lowest_quote_candidate(documents: list[PurchaseDocument]) -> Optional[PurchaseDocument]:
    quotes = [d for d in documents if d.document_type == DocumentType.QUOTE and d.total.value is not None]
    if not quotes:
        return None
    return min(quotes, key=lambda q: q.total.value or 0)


def _compare_field(label: str, expected: str, actual: str, normalizer: Callable[[str], str], doc_name: str) -> Optional[CheckResult]:
    if not expected and not actual:
        return None
    if not expected or not actual:
        missing = "최저가 견적" if not expected else doc_name
        return CheckResult(label, VerificationStatus.REVIEW, f"{missing}에 값이 없어 {label} 대조를 완료하지 못했습니다.")
    norm_expected = normalizer(expected)
    norm_actual = normalizer(actual)
    if norm_expected and norm_actual and norm_expected == norm_actual:
        return CheckResult(label, VerificationStatus.PASS, f"{label} 일치 (최저가 견적 기준).")
    return CheckResult(label, VerificationStatus.CONFLICT, f"{label} 불일치: 최저가 견적='{expected}', {doc_name}='{actual}'.")


def verify_documents(documents: list[PurchaseDocument], purchase_type: str = "물품") -> list[CheckResult]:
    results = compare_documents(documents)
    license_doc = next((d for d in documents if d.document_type == DocumentType.BUSINESS_LICENSE), None)
    bank_doc = next((d for d in documents if d.document_type == DocumentType.BANK_COPY), None)
    confirm_doc = next((d for d in documents if d.document_type == DocumentType.CONSTRUCTION_CONFIRMATION), None)
    required = ((DocumentType.BUSINESS_LICENSE, "사업자등록증", license_doc), (DocumentType.BANK_COPY, "통장사본", bank_doc))
    for _kind, name, doc in required:
        results.append(CheckResult(name, VerificationStatus.PASS if doc else VerificationStatus.MISSING, "문서 확인" if doc else "필수 문서가 없습니다."))
    if purchase_type == "공사":
        results.append(CheckResult("공사 확인서", VerificationStatus.PASS if confirm_doc else VerificationStatus.MISSING, "문서 확인" if confirm_doc else "공사 선택 시 공사 확인서가 필요합니다."))
    lowest = _lowest_quote_candidate(documents)
    if lowest:
        if license_doc:
            name_check = _compare_field("사업자등록증 업체명", _value(lowest.vendor_name), _value(license_doc.vendor_name), normalize_name, "사업자등록증")
            if name_check:
                results.append(name_check)
            number_check = _compare_field("사업자등록증 사업자번호", _value(lowest.business_number), _value(license_doc.business_number), lambda v: re.sub(r"\D", "", v), "사업자등록증")
            if number_check:
                results.append(number_check)
        if bank_doc:
            holder_check = _compare_field("통장사본 예금주", _value(lowest.vendor_name), _value(bank_doc.account_holder), normalize_name, "통장사본")
            if holder_check:
                results.append(holder_check)
            account_check = _compare_field("통장사본 계좌번호", _value(lowest.account_number), _value(bank_doc.account_number), lambda v: re.sub(r"\D", "", v), "통장사본")
            if account_check:
                results.append(account_check)
        if confirm_doc:
            amount_check = _compare_field("공사 확인서 금액", str(lowest.total.value or ""), str(confirm_doc.total.value or ""), lambda v: re.sub(r"\D", "", v), "공사 확인서")
            if amount_check:
                results.append(amount_check)
            name_check = _compare_field("공사 확인서 업체명", _value(lowest.vendor_name), _value(confirm_doc.vendor_name), normalize_name, "공사 확인서")
            if name_check:
                results.append(name_check)
            number_check = _compare_field("공사 확인서 사업자번호", _value(lowest.business_number), _value(confirm_doc.business_number), lambda v: re.sub(r"\D", "", v), "공사 확인서")
            if number_check:
                results.append(number_check)
        recipient_raw = _value(lowest.recipient)
        if recipient_raw:
            normalized_recipient = normalize_name(recipient_raw)
            has_keyword = "산학협력단" in normalized_recipient or "산단" in normalized_recipient
            results.append(CheckResult(
                "수신자 산학협력단",
                VerificationStatus.PASS if has_keyword else VerificationStatus.REVIEW,
                "수신자에 산학협력단/산단이 포함되어 있습니다." if has_keyword else f"수신자 '{recipient_raw}'에 산학협력단/산단이 없습니다. 확인 필요.",
            ))
        else:
            results.append(CheckResult("수신자 산학협력단", VerificationStatus.WARNING, "수신자 값이 없어 산학협력단 여부를 확인하지 못했습니다."))
        contact_fields = (
            ("담당자", lowest.contact_person),
            ("전화번호", lowest.phone),
            ("이메일", lowest.email),
        )
        for label, evidence in contact_fields:
            if not evidence.raw:
                results.append(CheckResult(label, VerificationStatus.REVIEW, f"최저가 견적 '{lowest.filename}'에 {label} 정보가 없습니다. 확인 필요."))
    for document in documents:
        for warning in document.warnings:
            results.append(CheckResult(document.filename, VerificationStatus.WARNING, warning))
    return results


def mask_account_number(value: str) -> str:
    clean = re.sub(r"\s+", "", value)
    if len(clean) <= 4:
        return "*" * len(clean)
    return "*" * (len(clean) - 4) + clean[-4:]


def my_business_prefix() -> str:
    return os.getenv("MY_BUSINESS_NUMBER_PREFIX", "216-82-")
