"""Pure Python rules for local OCR purchase-document verification."""
from __future__ import annotations

import os
import re
import difflib
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Callable, Optional


from purchase_verifier.erp_parser import calculate_business_days


class DocumentType(str, Enum):
    QUOTE = "QUOTE"
    ONLINE_STORE_CART = "ONLINE_STORE_CART"
    BUSINESS_LICENSE = "BUSINESS_LICENSE"
    BANK_COPY = "BANK_COPY"
    CONSTRUCTION_CONFIRMATION = "CONSTRUCTION_CONFIRMATION"
    UNKNOWN = "UNKNOWN"


class FieldPolicy(str, Enum):
    EXACT = "EXACT"
    TEXT_REVIEW = "TEXT_REVIEW"


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
    representative: Evidence = field(default_factory=Evidence) 
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
    expected_raw: str = ""  # 견적서 원본 텍스트 저장용
    actual_raw: str = ""    # 비교 대상 문서 원본 텍스트 저장용


@dataclass(frozen=True)
class LowestPriceResult:
    candidate_filename: Optional[str]
    amount: Optional[int]
    confirmed: bool
    reason: str


def detect_ocr_anomaly(text: str) -> tuple[bool, str]:
    """로컬 OCR 결과에서 자모 분리나 텍스트 깨짐, 어색한 단어 여부를 감지합니다."""
    if not text or len(text.strip()) < 10:
        return True, "추출된 텍스트가 너무 적거나 없습니다."

    if "[스캔 이미지" in text or "[OCR" in text or "미설치" in text:
        return True, "스캔 이미지 문서 감지 (로컬 OCR 엔진 미설치로 텍스트 자동 판독 불가)"

    # 1. 분리된 자음/모음(ㄱ-ㅎㅏ-ㅣ)이 단독으로 다수 등장하는지 검사
    jamo_matches = re.findall(r"[\u3131-\u318E]", text)
    complete_hangul = re.findall(r"[\uAC00-\uD7A3]", text)

    if len(jamo_matches) >= 5 and len(complete_hangul) > 0:
        jamo_ratio = len(jamo_matches) / (len(jamo_matches) + len(complete_hangul))
        if jamo_ratio > 0.1:
            return True, f"한글 자모 분리 현상이 다수 발견되었습니다 (자모 비율 {jamo_ratio*100:.1f}%)."

    # 2. 특수문자 깨짐 ( 등 유니코드 대체문자 또는 비정상 특수문자 연속)
    if "" in text or len(re.findall(r"[^\w\s.,;:()/\-–—\[\]{}가-힣]", text)) > len(text) * 0.25:
        return True, "특수문자 및 기호 깨짐 비율이 높습니다."

    return False, ""



_BIZ_RE = re.compile(
    r"(?<!\d)"
    r"(\d(?:\s*\d){2}\s*[-–—―─\s]\s*\d(?:\s*\d)\s*[-–—―─\s]\s*\d(?:\s*\d){4}"
    r"|\d{3}\s*[-–—―─]?\s*\d{2}\s*[-–—―─]?\s*\d{5})"
    r"(?!\d)"
)
_DATE_RE = re.compile(r"(20\d{2})\s*[./년年-]\s*(\d{1,2})\s*[./월月-]\s*(\d{1,2})\s*[일日]?")
_AMOUNT_NUM_PATTERN = r"(?:\d{1,3}(?:\s*,\s*\d{3})+|\d{4,})"
_AMOUNT_RE = re.compile(rf"(?<!\d)({_AMOUNT_NUM_PATTERN})(?:\s*원)?(?!\d)")


def _clean_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


_KOREAN_SURNAMES = r"[김이박최정강조윤장임한오서신권황안송전홍류유고문양손배백허남심노하곽성차주우구라나민진채원천방공현함변염여추도소석설마길연위표명기반왕금옥육인맹제모탁국]"


def _clean_vendor_name(name: str) -> tuple[str, str]:
    cleaned = name
    while True:
        prev = cleaned
        cleaned = re.sub(r"^[0-9\s()\[\]{}]*?(?:상호\s*[\(\[{]법인명[\)\]}]|상호|법인명\s*[\(\[{]단체명[\)\]}]|법인명|단체명|상호명|협력사명|협력사|공사업체|시공사)[0-9\s()\[\]{}:：.을은는이가]*", "", cleaned)
        cleaned = re.sub(r"^[0-9:\s]+", "", cleaned)
        cleaned = re.sub(r"^[\)\]\}\s]+", "", cleaned)
        cleaned = re.sub(r"^\[주\]\s*", "(주)", cleaned)
        cleaned = re.sub(r"^[웅원도고영이]?주[\]\)）\s]+", "(주)", cleaned)
        cleaned = re.sub(r"^[\[({](?:법인명|단체명|상호명|상호)[\])}]\s*", "", cleaned)
        if cleaned == prev:
            break
    # 상호명 뒤에 괄호로 붙은 사업자등록번호 분리/제거 (예: ㈜ 현원상사 ( 807-86-01103) -> ㈜ 현원상사)
    cleaned = re.sub(r"\s*\(?\s*\d{3}[\s-]*\d{2}[\s-]*\d{5}\s*\)?", "", cleaned).strip()
    cleaned = re.sub(
        r"\s*(?:견\s*적\s*번\s*호|견\s*적\s*일|일\s*시|발\s*행\s*일|납\s*품\s*기\s*일|성\s*명|대\s*표\s*자|대\s*표\s*이\s*사|대\s*표|사\s*업\s*장|주\s*소|업\s*태|종\s*목|연\s*락\s*처|전\s*화|담\s*당\s*부\s*서|담\s*당\s*자|담\s*당|영\s*업\s*팀|솔\s*루\s*션\s*사\s*업\s*부|솔\s*루\s*션\s*팀|사\s*업\s*부|영\s*업\s*부|홈\s*페\s*이\s*지)\s*[:：]?\s*.*$",
        "",
        cleaned,
    ).strip()
    cleaned = re.sub(r"\s*(?:0[17]\d[- ]?\d{3,4}[- ]?\d{4}|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+).*$", "", cleaned).strip()

    # 슬래시(/)로 상호와 대표자가 구분된 경우 (예: (주)테스트링크 / 김문백)
    if "/" in cleaned:
        slash_parts = [p.strip() for p in cleaned.split("/") if p.strip()]
        if len(slash_parts) == 2 and _is_representative_candidate(slash_parts[1]):
            return slash_parts[0], slash_parts[1]

    parts = cleaned.split()
    if len(parts) == 2 and re.fullmatch(rf"{_KOREAN_SURNAMES}[가-힣]{{1,2}}", parts[1]):
        if parts[0] not in ("성", "성명", "대표", "대", "표", "명", "대표자", "상호"):
            return parts[0], parts[1]
        return "", parts[1]
    return cleaned, ""


def _is_representative_candidate(name: str) -> bool:
    clean = _clean_spaces(name)
    if re.fullmatch(r"\(?[^\w가-힣\u4e00-\u9fff]*\)?", clean) or re.fullmatch(r"\([^)]*\)", clean):
        return False
    if re.search(r"^[)\]}>'\"`.,·]", clean):
        return False
    if re.search(r"[A-Za-z][가-힣]|[가-힣][A-Za-z]", clean) and not re.search(r"\([A-Za-z\s]+\)", clean):
        return False
    # 전화번호, 금액 등 숫자가 포함된 문자열은 대표자명 후보에서 제외
    if re.search(r"\d", clean):
        return False
    invalid_keywords = (
        "담당자", "휴대폰", "전화", "이메일", "사업장", "주소", "상호", "업태", "종목", "성명", "유형", "대표유형",
        "회사", "회사명", "타워", "빌딩", "센터", "아파트", "오피스텔", "주식회사", "하이츠", "밸리"
    )
    if any(k in clean for k in invalid_keywords):
        return False
    if clean in ("이사", "대표이사", "사장", "부장", "과장", "팀장", "상무", "전무", "대표", "대 표", "유형", "표자"):
        return False
    digits_or_letters = re.sub(r"[^\w가-힣\u4e00-\u9fff]", "", clean)
    return 1 < len(digits_or_letters) <= 12


def _is_contact_person_candidate(val: str) -> bool:
    norm = normalize_name(val)
    if any(k in norm for k in ("귀하", "귀중", "산학협력", "세종", "연구원", "부서", "구매담당", "담당부서", "대표", "대표이사")):
        return False
    clean = re.sub(r"[^\w가-힣]", "", val)
    return len(clean) >= 2


def normalize_name(value: str) -> str:
    clean = re.sub(r"[（\(]\s*(?:주|유)\s*[）\)]|주식회사|유한회사", "", value)
    return re.sub(r"[^0-9가-힣A-Za-z]", "", clean).lower()


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
    haystack_nospace = re.sub(r"\s+", "", haystack)
    scores = {kind: 0 for kind in DocumentType}
    keywords = {
        DocumentType.QUOTE: ("견적", "quotation", "quote", "공급가액", "합계", "subtotal", "단가", "見積"),
        DocumentType.ONLINE_STORE_CART: (
            "장바구니총액", "장바구니", "apple.com/kr/shop/bag", "교육스토어홈", "applecare+", "apple(kr)",
            "결제예정금액", "삼성닷컴", "samsung.com", "주문상품", "멤버십포인트", "shoppingcart", "주문금액"
        ),
        DocumentType.BUSINESS_LICENSE: ("사업자등록증", "사업자번호", "등록번호", "대표자", "업태", "법인명", "세무서", "세무서장", "법인사업자", "일반과세자", "사면자", "들록증"),
        DocumentType.BANK_COPY: ("통장사본", "예금주", "계좌번호", "은행", "통장", "보통예금", "국민은행", "기업은행", "신한은행", "우리은행", "하나은행", "농협", "kb", "ibk"),
        DocumentType.CONSTRUCTION_CONFIRMATION: ("공사확인서", "준공", "공사명", "시공", "시설공사", "장비설치", "설치확인서"),
    }
    for kind, terms in keywords.items():
        scores[kind] = sum(haystack_nospace.count(re.sub(r"\s+", "", term)) for term in terms)
    explicit_titles = {
        DocumentType.BUSINESS_LICENSE: ("사업자등록증", "사면자들록"),
        DocumentType.BANK_COPY: ("통장사본", "통장", "보통예금"),
        DocumentType.CONSTRUCTION_CONFIRMATION: ("공사확인서", "시설공사", "장비설치확인서", "장비설치", "설치확인서"),
        DocumentType.ONLINE_STORE_CART: ("장바구니총액", "장바구니-apple", "apple.com/kr/shop/bag", "결제예정금액", "삼성닷컴", "교육스토어홈"),
        DocumentType.QUOTE: ("견적서", "견적", "quotation", "見積書", "見積"),
    }
    title_hits = {kind: any(re.sub(r"\s+", "", term) in haystack_nospace for term in terms) for kind, terms in explicit_titles.items()}
    # 장바구니 화면 강한 핑거프린트 감지
    if "장바구니" in haystack_nospace and any(k in haystack_nospace for k in ("총액", "apple", "samsung", "삼성", "스토어", "결제", "bag", "shop")):
        title_hits[DocumentType.ONLINE_STORE_CART] = True
    if "결제예정금액" in haystack_nospace and any(k in haystack_nospace for k in ("삼성", "samsung", "주문상품", "멤버십")):
        title_hits[DocumentType.ONLINE_STORE_CART] = True

    if sum(title_hits.values()) == 1:
        for kind, hit in title_hits.items():
            if hit:
                return kind
    best = max(scores, key=lambda kind: scores[kind])
    return best if scores[best] > 0 else DocumentType.UNKNOWN


def _make_label_pattern(labels: tuple[str, ...]) -> str:
    patterns = []
    # 길이가 긴 라벨부터 매칭되도록 역순 정렬 (예: "사업자등록번호" > "사업자")
    sorted_labels = sorted(labels, key=len, reverse=True)
    for label in sorted_labels:
        if re.fullmatch(r"[a-zA-Z]+", label):
            patterns.append(rf"\b{label}\b")
        else:
            escaped_chars = [re.escape(c) for c in label]
            inner = r"[ \t'’`.,·]*".join(escaped_chars)
            if label in ("성명", "대표자", "대표자명", "대표자 성명", "대표", "상호", "상호명"):
                patterns.append(inner + rf"(?!(?:부서|이사|유형|번호|주소|전화)(?![가-힣]))(?=[ \t:：.\-\(\[\{{]|\n|$)")
            else:
                patterns.append(inner + r"(?![가-힣])")
    return "|".join(patterns)


def _labeled(
    text: str,
    labels: tuple[str, ...],
    value_pattern: str = r"[^\n:：]{1,100}",
    validator: Optional[Callable[[str], bool]] = None,
) -> Evidence:
    label_pattern = _make_label_pattern(labels)
    # 1. 같은 줄에서 탐색
    for match in re.finditer(rf"(?:{label_pattern})[\s'’`.,·]*[:：.\-]?\s*({value_pattern})", text, re.IGNORECASE):
        val_str = match.group(1)
        val_str = re.sub(r"^[\[\(\{]\s*(?:법인명|단체명|대표유형|상호명|상호)\s*[\]\)\}]\s*[:：]?\s*", "", val_str)
        value = _clean_spaces(val_str).strip("|,;: .\\-")
        if not value:
            continue
        if validator and not validator(value):
            continue
        return _evidence(value, match.group(0).strip(), 0.92)

    # 2. 줄바꿈 뒤 다음 줄에서 탐색 (예: 상호\n명성창호, 성명\n김의진)
    for line_match in re.finditer(rf"(?:{label_pattern})[\s'’`.,·]*[:：.\-]?\s*\n([^\n]+)", text, re.IGNORECASE):
        next_line = line_match.group(1)
        next_val = _clean_spaces(next_line).strip("|,;: .\\-")
        if not next_val:
            continue
        if validator and not validator(next_val):
            continue
        return _evidence(next_val, line_match.group(0).strip(), 0.90)

    return Evidence()


def parse_korean_currency(text: str) -> Optional[int]:
    """Parses Korean currency written in words like '삼백구만일천' or '일금 삼백구만일천원정'."""
    num_map = {'일': 1, '이': 2, '삼': 3, '사': 4, '오': 5, '육': 6, '칠': 7, '팔': 8, '구': 9}
    small_units = {'십': 10, '백': 100, '천': 1000}
    big_units = {'만': 10000, '억': 100000000, '조': 1000000000000}
    
    clean = re.sub(r'[^일이삼사오육칠팔구십백천만억조]', '', text)
    if not clean:
        return None
    # 화폐 단위(원, 금) 또는 자릿수 단위(십, 백, 천, 만, 억, 조)가 없는 단순 한글 음절은 금액이 아님 (예: '변동이'의 '이')
    if not any(u in clean for u in ("십", "백", "천", "만", "억", "조")) and "원" not in text and "금" not in text:
        return None
    total = 0
    section = 0
    curr = 0
    for c in clean:
        if c in num_map:
            curr = num_map[c]
        elif c in small_units:
            section += (curr if curr > 0 else 1) * small_units[c]
            curr = 0
        elif c in big_units:
            section += curr
            total += (section if section > 0 else 1) * big_units[c]
            section = 0
            curr = 0
    total += section + curr
    return total if total > 0 else None


_CURRENCY_PREFIX = r"(?:[₩￦\$]|\\b[Ww]\\b)"


def _amount_after_label(
    text: str,
    labels: tuple[str, ...],
    exclude_keywords: tuple[str, ...] = (),
) -> Amount:
    label_pattern = _make_label_pattern(labels)
    # 1. 같은 줄에서 먼저 탐색 (부가세 10% 등 세율 표기 허용, 줄바꿈 완전 배제)
    for match in re.finditer(
        rf"(?<!sub\s)(?<!sub)({label_pattern})([^\d\n]{{0,30}}?(?:\d{{1,2}}\s*%)?[^\d\n]{{0,30}}?)(?:{_CURRENCY_PREFIX}[^\S\n]*)?({_AMOUNT_NUM_PATTERN})",
        text,
        re.IGNORECASE,
    ):
        mid_text = match.group(2)
        if exclude_keywords and any(k in mid_text for k in exclude_keywords):
            continue
        end_of_line = text.find("\n", match.start())
        line_after = text[match.end():end_of_line] if end_of_line != -1 else text[match.end():]
        if exclude_keywords and any(k in line_after for k in exclude_keywords):
            continue
        raw = match.group(3)
        clean_digits = re.sub(r"[^\d]", "", raw)
        if clean_digits:
            matched_label = match.group(1)
            # 같은 행에 [공급가액, 부가세, 합계] 3개 숫자가 나란히 있는 행인지 검사 (예: 총 견적금액 9,700,000 970,000 10,670,000)
            line_full = text[match.start():end_of_line] if end_of_line != -1 else text[match.start():]
            line_amts = [int(re.sub(r"[^\d]", "", m.group(1))) for m in re.finditer(_AMOUNT_RE, line_full)]
            if len(line_amts) >= 3:
                if abs(line_amts[0] + line_amts[1] - line_amts[2]) <= max(10, int(line_amts[2] * 0.02)):
                    return Amount(line_amts[2], _evidence(f"{line_amts[2]:,}", line_full.strip(), 0.98), matched_label)
                elif abs(line_amts[-3] + line_amts[-2] - line_amts[-1]) <= max(10, int(line_amts[-1] * 0.02)):
                    return Amount(line_amts[-1], _evidence(f"{line_amts[-1]:,}", line_full.strip(), 0.98), matched_label)

            return Amount(int(clean_digits), _evidence(raw.strip(), match.group(0).strip(), 0.95), matched_label)

    # 2. 줄바꿈 뒤 다음 1~3줄에서 탐색 (한글 금액 표기 "삼백구만일천" 건너뛰기 및 교차 검증 지원)
    for match in re.finditer(
        rf"(?<!sub\s)(?<!sub)({label_pattern})([^\d]{{0,80}}?)(?:{_CURRENCY_PREFIX}\s*)?({_AMOUNT_NUM_PATTERN})",
        text,
        re.IGNORECASE,
    ):
        mid_text = match.group(2)
        if exclude_keywords and any(k in mid_text for k in exclude_keywords):
            continue
        raw = match.group(3)
        clean_digits = re.sub(r"[^\d]", "", raw)
        if clean_digits:
            num_val = int(clean_digits)
            matched_label = match.group(1)
            kor_val = parse_korean_currency(mid_text)
            # 한글 표기와 아라비아 숫자가 상호 일치하는 경우 최고 신뢰도
            conf = 0.99 if (kor_val is not None and kor_val == num_val) else 0.93
            return Amount(num_val, _evidence(raw.strip(), match.group(0).strip(), conf), matched_label)

    # 3. 아라비아 숫자가 없고 순수 한글 금액만 있는 경우
    for match in re.finditer(rf"(?<!sub\s)(?<!sub)({label_pattern})([^\d\n]{{0,50}})", text, re.IGNORECASE):
        mid_text = match.group(2)
        if exclude_keywords and any(k in mid_text for k in exclude_keywords):
            continue
        kor_val = parse_korean_currency(mid_text)
        if kor_val:
            matched_label = match.group(1)
            return Amount(kor_val, _evidence(mid_text.strip(), match.group(0).strip(), 0.90), matched_label)

    return Amount()


def _date_evidence(text: str, labels: tuple[str, ...]) -> Evidence:
    label_pattern = _make_label_pattern(labels)
    match = re.search(rf"(?:{label_pattern})[^\d]*(20\d{{2}}\s*[./년年-]\s*\d{{1,2}}\s*[./월月-]\s*\d{{1,2}}\s*[일日]?)", text, re.IGNORECASE)
    if not match:
        return Evidence()
    parsed = _parse_date(match.group(1))
    formatted = parsed.isoformat() if parsed else match.group(1).strip()
    return _evidence(formatted, match.group(0).strip(), 0.95)


def _parse_date(value: str) -> Optional[date]:
    match = _DATE_RE.search(value)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def calculate_expiry(text: str, quote_date: Evidence) -> Evidence:
    explicit = _date_evidence(text, ("만료일", "유효만료일", "견적유효기간", "견적 유효 기간", "견적유효", "유효기간", "valid until", "expiry"))
    if explicit.raw:
        return explicit
    start = _parse_date(quote_date.raw)
    
    # 1. '유효기간은 9월 30일까지' 형태 (연도 생략형)
    m_month_day = re.search(r"(?:유효기간|견적유효|만료일|유효\s*만료일)[^\d\n]{0,30}(\d{1,2})\s*월\s*(\d{1,2})\s*일", text)
    if m_month_day and start:
        m_val = int(m_month_day.group(1))
        d_val = int(m_month_day.group(2))
        try:
            exp_d = date(start.year, m_val, d_val)
            if exp_d < start:
                exp_d = date(start.year + 1, m_val, d_val)
            return _evidence(exp_d.isoformat(), m_month_day.group(0), 0.90, True)
        except ValueError:
            pass

    # 2. '견적일로부터 15일', 'PERIOD OF VALIDY : 4주' 등 기간 표기
    validity = re.search(r"(?:유효기간|견적유효기간|견적유효|견적일로부터|견적\s*후|발행일로부터|validity|validy|period\s*of\s*valid(?:ity|y))[^\d\n]{0,40}(\d{1,3})\s*(일|日|days?|주|weeks?|개월|months?)", text, re.IGNORECASE)
    if validity and start:
        num = int(validity.group(1))
        unit = validity.group(2).lower()
        if "개월" in unit or "month" in unit:
            days = num * 30
        elif "주" in unit or "week" in unit:
            days = num * 7
        else:
            days = num
        expiry = start + timedelta(days=days)
        raw = expiry.isoformat()
        return _evidence(raw, validity.group(0), 0.85, True)
    return Evidence()


def _extract_items(text: str) -> list[Item]:
    items: list[Item] = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    header_idx = -1
    for idx, l in enumerate(lines):
        l_nospace = re.sub(r"\s+", "", l).lower()
        if any(k in l_nospace for k in ("품명", "품목", "품명및사양", "description")) and any(k in l_nospace for k in ("수량", "단가", "금액", "공급가", "공급가액", "amount")):
            header_idx = idx
            break

    target_lines = lines[header_idx + 1:] if header_idx >= 0 else lines

    pending_prefix = ""
    for idx, line in enumerate(target_lines):
        line_clean = _clean_spaces(line)
        line_clean = re.sub(r"(\d+)\s*,\s*(\d{3})", r"\1,\2", line_clean)
        line_clean = re.sub(r"\b(\d{1,2})\s+(0,\d{3}(?:,\d{3})*)\b", r"\1\2", line_clean)
        line_clean = re.sub(r"\b(\d{1,2})\s+(\d{2},\d{3})\b", r"\1\2", line_clean)
        if not line_clean or re.search(r"(품명|품목|description).*(수량|단가|금액)", line_clean, re.I):
            continue
        line_lower = line_clean.lower()
        line_nospace = re.sub(r"\s+", "", line_clean).lower()
        if any(k in line_nospace for k in ("견적", "일자", "유효", "기간", "전화", "연락처", "팩스", "email", "이메일", "사업자", "등록번호", "대표", "주소", "소재지", "결제", "통장", "계좌", "담당", "특기사항", "특기", "비고", "참고사항", "주의사항")):
            continue
        if re.search(r"^\s*(?:합\s*계|총\s*계|총\s*합\s*계|총\s*견적금액)\b", line_clean, re.I):
            if items:
                break
            continue
        if re.search(r"^\s*(?:소\s*계|계|VAT|부가세)\b", line_clean, re.I):
            continue
        if any(k in line_nospace for k in ("공급가액", "세액", "부가세", "부가가치세", "총합계", "합계", "총계", "소계", "vat")):
            if not any(k in line_clean for k in ("인텔", "쿨러", "보드", "RAM", "SSD", "VGA", "파워", "케이스")):
                continue
        if re.search(r"[-•*]?\s*(?:금\s*액|합계금액|총\s*합계|공\s*급\s*가|부\s*가\s*세|납품\s*기일|납기|워런티|유효\s*기간|견적\s*일자|결제\s*조건|프로젝트\s*명|배송\s*방법)\s*[:：]", line_clean):
            continue
        if re.search(r"(?:gross\s*amount|total\s*amount|tax\s*amount|원정|\(\s*[₩\\])", line_clean, re.I):
            continue
        if any(k in line_lower for k in ("tel", "fax", "phone", "e-mail", "email", "http", "www", "p.i.c", "c.p")):
            continue
        if _DATE_RE.search(line_clean) or _BIZ_RE.search(line_clean) or re.search(r"01[0-9][- ]?\d{3,4}[- ]?\d{4}", line_clean):
            continue

        num_matches = list(re.finditer(r"(?<!\d)\d+(?:,\d{3})*(?!\d)", line_clean))
        if not num_matches:
            if header_idx >= 0 and len(line_clean) <= 40 and not any(k in line_nospace for k in ("견적", "산학협력", "등록번호", "합계", "비고", "규격", "quotation", "sheet", "주식회사")):
                if items and any(k in line_clean.lower() for k in ("wifi", "argb", "plus", "pro", "max", "edition", "정품", "케이블")):
                    items[-1] = Item(
                        name=f"{items[-1].name} {line_clean}".strip(),
                        quantity=items[-1].quantity,
                        unit_price=items[-1].unit_price,
                        amount=items[-1].amount,
                        evidence=items[-1].evidence,
                    )
                else:
                    pending_prefix = (pending_prefix + " " + line_clean).strip()
            continue

        numbers = [int(m.group(0).replace(",", "")) for m in num_matches]
        matched = False
        quantity, unit_price, amount = 1.0, None, None
        name_cutoff = len(line_clean)

        # 0. 마지막 3개 숫자가 [공급가액, 세액, 합계] 인 열 구조 (예: 9,700,000 + 970,000 = 10,670,000)
        if len(numbers) >= 3 and numbers[-1] >= 1000 and numbers[-3] >= 10000:
            if abs(numbers[-3] + numbers[-2] - numbers[-1]) <= max(10, int(numbers[-1] * 0.02)):
                amount = numbers[-3]
                rem_matches = num_matches[:-3]
                rem_numbers = numbers[:-3]
                if len(rem_numbers) >= 2 and abs(rem_numbers[-2] * rem_numbers[-1] - amount) <= max(10, int(amount * 0.02)):
                    quantity = float(rem_numbers[-2])
                    unit_price = rem_numbers[-1]
                    name_cutoff = rem_matches[-2].start()
                elif len(rem_numbers) >= 2 and abs(rem_numbers[-1] * rem_numbers[-2] - amount) <= max(10, int(amount * 0.02)):
                    quantity = float(rem_numbers[-1])
                    unit_price = rem_numbers[-2]
                    name_cutoff = rem_matches[-2].start()
                elif len(rem_numbers) >= 1 and 1 <= rem_numbers[-1] <= 100:
                    quantity = float(rem_numbers[-1])
                    unit_price = amount // int(quantity)
                    name_cutoff = rem_matches[-1].start()
                else:
                    name_cutoff = num_matches[-3].start()
                    unit_price = amount
                matched = True

        # 1. 마지막 숫자가 부가세(공급가액의 약 10%)인 열 구조: [..., 공급가액, 세액]
        if not matched and len(numbers) >= 2 and numbers[-1] >= 1000 and numbers[-2] >= 10000:
            if abs(numbers[-1] - numbers[-2] * 0.1) <= max(10, int(numbers[-2] * 0.05)):
                amount = numbers[-2]
                rem_matches = num_matches[:-2]
                rem_numbers = numbers[:-2]
                if len(rem_numbers) >= 2 and abs(rem_numbers[-2] * rem_numbers[-1] - amount) <= max(10, int(amount * 0.02)):
                    quantity = float(rem_numbers[-2])
                    unit_price = rem_numbers[-1]
                    name_cutoff = rem_matches[-2].start()
                elif len(rem_numbers) >= 1 and 1 <= rem_numbers[-1] <= 100:
                    quantity = float(rem_numbers[-1])
                    unit_price = amount // int(quantity)
                    name_cutoff = rem_matches[-1].start()
                else:
                    name_cutoff = num_matches[-2].start()
                    unit_price = amount
                matched = True

        # 2. 일반 열 구조: [..., 수량, 단가, 금액] 또는 [..., 수량, 금액]
        if not matched and len(numbers) >= 1 and numbers[-1] >= 1000:
            amount = numbers[-1]
            rem_matches = num_matches[:-1]
            rem_numbers = numbers[:-1]
            if len(rem_numbers) >= 2 and abs(rem_numbers[-2] * rem_numbers[-1] - amount) <= max(10, int(amount * 0.02)):
                quantity = float(rem_numbers[-2])
                unit_price = rem_numbers[-1]
                name_cutoff = rem_matches[-2].start()
                matched = True
            elif len(rem_numbers) >= 1 and 1 <= rem_numbers[-1] <= 100:
                cand_qty_text = line_clean[rem_matches[-1].start():min(len(line_clean), rem_matches[-1].end() + 5)]
                if re.search(r"\d+\s*세대|\bi\d+\b", cand_qty_text, re.I) or (amount < 30000 and (amount // int(rem_numbers[-1])) < 5000):
                    matched = False
                else:
                    quantity = float(rem_numbers[-1])
                    unit_price = amount // int(quantity)
                    name_cutoff = rem_matches[-1].start()
                    matched = True
            elif len(numbers) == 1 or (len(rem_numbers) >= 1 and rem_numbers[-1] > 100):
                quantity = 1.0
                unit_price = amount
                name_cutoff = num_matches[-1].start()
                matched = True

        if matched and amount is not None:
            raw_name = line_clean[:name_cutoff].strip()
            raw_name = re.sub(r"^\d+\s*(?:[\.\)\-]\s*|\s+)", "", raw_name).strip()
            if pending_prefix:
                full_name = f"{pending_prefix} {raw_name}".strip()
                pending_prefix = ""
            else:
                full_name = raw_name
            full_name = re.sub(r"^\d+\s*(?:[\.\)\-]\s*|\s+)", "", full_name).strip()
            if full_name and len(full_name) >= 2:
                items.append(Item(
                    name=full_name,
                    quantity=quantity,
                    unit_price=unit_price,
                    amount=amount,
                    evidence=_evidence(line_clean, line_clean, 0.75)
                ))

    if not items:
        # 1. 분리되어 기재된 차량용 제어 IP + FPGA 플랫폼 패턴 우선 탐색
        m_ip = re.search(r"(차량용\s*기능안전\s*제어\s*IP)", text)
        m_fpga = re.search(r"((?:검증용\s*)?FPGA\s*플랫폼\s*V\d+)", text)
        if m_ip and m_fpga:
            cand_comb = f"{_clean_spaces(m_ip.group(1))} {_clean_spaces(m_fpga.group(1))}"
            items.append(Item(name=cand_comb, quantity=1.0, unit_price=None, amount=None, evidence=_evidence(cand_comb, cand_comb, 0.85, True)))

    if not items:
        # 2. 단일 품목/총액 견적서 품목명 탐색 (견적명, 공사명, 품명 라벨)
        proj_match = re.search(
            r"(?:견\s*적\s*명|공\s*사\s*명|용\s*역\s*명|물\s*품\s*명|제\s*목|(?:^|\n)\s*품\s*명)\s*[:：.]?\s*([가-힣A-Za-z0-9\s()\[\]/_-]{3,60})",
            text,
        )
        if proj_match:
            cand_p = _clean_spaces(proj_match.group(1)).strip()
            if not re.search(r"(?:description|amount|remark|규격|단가|수량|금액)", cand_p, re.I):
                cand_p = re.split(r"(?:수\s*량|단\s*가|금\s*액|일\s*자|규\s*격|귀\s*[하중]|납\s*품|VAT|AMOUNT|건\s*대해|에\s*대해)", cand_p, flags=re.I)[0].strip(" -:;,.")
                if len(cand_p) >= 3 and not any(k in cand_p for k in ("견적서", "공급가", "합계", "총액", "내역")):
                    items.append(Item(name=cand_p, quantity=1.0, unit_price=None, amount=None, evidence=_evidence(cand_p, proj_match.group(0).strip(), 0.85, True)))

    if not items:
        # 3. 본문 내 특정 품목/플랫폼 패턴 탐색
        broad_item = re.search(r"([가-힣A-Za-z0-9\s/_-]{3,30}(?:플랫폼|시스템|소프트웨어|하드웨어|장비|모듈|IP|키트|센서)\s*(?:V\d+|[A-Z0-9]+)?)", text)
        if broad_item:
            cand_b = _clean_spaces(broad_item.group(1)).strip()
            if len(cand_b) >= 4 and not any(k in cand_b for k in ("주식회사", "세종대", "산학협력", "견적", "결제", "납품", "은행", "통장", "시스템 대전", "시스템 서울")):
                items.append(Item(name=cand_b, quantity=1.0, unit_price=None, amount=None, evidence=_evidence(cand_b, broad_item.group(0).strip(), 0.80, True)))

    return items


def _extract_online_cart_items(text: str) -> list[Item]:
    """공식 온라인스토어(애플, 삼성닷컴 등) 장바구니 캡처 화면에서 품목, 수량, 금액을 추출합니다."""
    items: list[Item] = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    apple_keywords = (
        "studio display", "pro display", "mac mini", "macbook", "imac", "mac studio", "mac pro",
        "ipad pro", "ipad air", "ipad mini", "ipad", "apple pencil", "magic keyboard", "applecare",
        "pencil", "keyboard"
    )
    recommendation_starts = ("마음에 들 만한 액세서리", "새 액세서리", "더 많은 제품 표시", "구입 관련 질문", "자주 함께 구매하는 제품")

    clean_lines = []
    for line in lines:
        if any(rec in line for rec in recommendation_starts):
            break
        clean_lines.append(line)

    in_hardware_spec = False
    i = 0
    while i < len(clean_lines):
        line = clean_lines[i]
        line_lower = line.lower()

        # 상단 네비게이션 메뉴 바 제외 (예: 스토어 Mac iPad iPhone Watch Vision...)
        nav_count = sum(1 for kw in ("스토어", "ipad", "iphone", "watch", "vision", "airpods", "tv") if kw in line_lower)
        if nav_count >= 3:
            i += 1
            continue

        # 하드웨어 세부 사양 구역 추적
        if re.search(r"제품\s*세부\s*정보|하드[웨페]어", line):
            in_hardware_spec = True
        if in_hardware_spec and any(k in line for k in ("applecare", "소계", "총계", "이 제품을 얼마나", "무료 각인")):
            in_hardware_spec = False

        amt_on_line = list(re.finditer(r"(?:[₩\\Ww]?\s*)(\d{1,3}(?:,\d{3})+|\d{5,})\s*원?", line))
        # 하드웨어 내장 스펙 목록(키보드 포함 구성, 어댑터 등)이고 동일 라인에 가격이 없으면 건너뜀
        if in_hardware_spec and not amt_on_line:
            i += 1
            continue

        # AppleCare 추천 배너(추가/호가 버튼이 있는 미담기 상태)는 건너뜀
        is_applecare_rec = ("applecare" in line_lower or "애플케어" in line_lower) and any(k in line for k in ("추가", "호가", "[추가]"))
        if is_applecare_rec:
            i += 1
            continue

        is_applecare_added = ("applecare" in line_lower or "애플케어" in line_lower) and not is_applecare_rec
        is_apple_prod = any(k in line_lower for k in apple_keywords) or bool(re.search(r"m\d+\s*칩", line_lower)) or is_applecare_added
        is_samsung_prod = ("갤럭시" in line or "galaxy" in line_lower) and not any(k in line for k in ("적립", "멤버십", "포인트", "혜택", "약관", "동의"))

        # 스펙/배송 등 메타 라인 제외 (노트북/일체형 PC 내장 키보드/마우스 사양 등 제외)
        is_spec_line = any(k in line_lower for k in (
            "하드웨어", "하드페어", "소프트웨어", "저장장치", "메모리", "어댑터", "전원", "포트", "카메라", "헤드폰",
            "디스플레이 최대", "액세서리 키트", "center stage", "retina디스플레이", "neural engine",
            "우편번호", "픽업", "배송지", "각인도 추가", "백라이트", "벽라이트", "구성된 모습", "구성된모습"
        ))

        if (is_apple_prod or is_samsung_prod) and not is_spec_line:
            prod_name = line
            price = None
            qty = 1.0

            # 1. 동일 라인 금액 매칭
            if amt_on_line:
                raw_num = int(amt_on_line[-1].group(1).replace(",", ""))
                if raw_num >= 50000:
                    price = raw_num
                    prod_name = line[:amt_on_line[-1].start()].strip()

            # 2. 수량 매칭 ('1 v', '1v', '1개', '수량 1' 우선 매칭하여 제품 규격(13, 14, 15) 오인식 방지)
            qty_match = re.search(r"(?:^|\s)(\d{1,2})\s*(?:v|개)(?:\s|$)", prod_name, re.I)
            if not qty_match:
                qty_match = re.search(r"수량\s*[:：]?\s*(\d{1,2})(?:\s|$)", prod_name)

            if qty_match:
                val_q = float(qty_match.group(1))
                if 1 <= val_q <= 50:
                    qty = val_q
                    prod_name = prod_name[:qty_match.start()] + " " + prod_name[qty_match.end():]

            # 3. 품목명 정제
            prod_name = re.sub(r"^(?:주문상품|상품명|품목|품명)[\s:：.]*", "", prod_name).strip()
            prod_name = re.sub(r"\s+", " ", prod_name)

            # 4. 다음 줄이 모델명, 규격, 색상인 경우 병합
            if i + 1 < len(clean_lines):
                next_l = clean_lines[i + 1]
                if re.match(r"^\d+\s*모델", next_l) or (len(next_l) <= 35 and any(k in next_l for k in ("모델", "스페이스", "실버", "스탠다드", "에디션", "SM-", "핑크", "그라파이트", "자급제", "Nano-texture", "글래스", "한국어", "블랙"))):
                    clean_next = re.sub(r"\s*(?:삭제|추가).*$", "", next_l).strip()
                    if clean_next and not any(k in clean_next for k in ("소계", "총계", "배송", "소프트웨어", "하드웨어", "하드페어")):
                        prod_name = f"{prod_name} {clean_next}"
                        i += 1

            # 5. 동일 라인에 가격이 없었던 경우 다음 줄에서 금액 탐색
            if price is None:
                for j in range(i + 1, min(i + 10, len(clean_lines))):
                    sub_l = clean_lines[j]
                    sub_l_lower = sub_l.lower()

                    if any(rec in sub_l for rec in recommendation_starts) or ("applecare" in sub_l_lower and any(k in sub_l for k in ("추가", "호가"))):
                        break
                    if any(sum_k in sub_l for sum_k in ("소계", "총계", "장바구니 총액", "장바구니 소계", "결제 금액", "결제 예정 금액", "주문 금액", "주문 총액", "배송 정보")):
                        break
                    if sub_l.strip() in ("삭제", "선택삭제", "삭제하기") and price is not None:
                        break
                    if any(tax_k in sub_l_lower for tax_k in ("vat", "부가세", "부가가치세", "세금", "적립", "포인트", "마일리지", "배송비", "배송료")):
                        continue
                    if any(k in sub_l_lower for k in ("저장장치", "메모리", "어댑터", "포트", "카메라", "헤드폰")):
                        continue

                    amt_matches = list(re.finditer(r"(?:[₩\\Ww]?\s*)(\d{1,3}(?:,\d{3})+|\d{5,})\s*원?", sub_l))
                    if amt_matches:
                        for am in amt_matches:
                            raw_num = int(am.group(1).replace(",", ""))
                            if raw_num >= 50000:
                                price = raw_num

            if prod_name and price is not None and not any(it.name == prod_name for it in items):
                items.append(Item(
                    name=prod_name,
                    quantity=qty,
                    unit_price=price,
                    amount=price if (price and qty == 1) else (int(price * qty) if price else None),
                    evidence=_evidence(prod_name, f"공식스토어 장바구니 품목: {prod_name}", 0.90)
                ))
        i += 1

    return items


def _is_vendor_name_candidate(name: str) -> bool:
    norm = normalize_name(name)
    excluded = (
        "세종", "산학협력", "산단", "사업자", "공급자", "시공사", "발행자", "수신자",
        "귀하", "귀중", "전자세금계산서", "견적", "견적일", "일자", "등록번호",
        "대표자", "대표", "납품", "수량", "금액", "단가", "합계", "소계", "부가세", "날짜",
        "ocr", "tesseract", "스캔", "이미지",
        "도매", "소매", "업태", "종목", "주소", "전화", "팩스", "이메일",
        "비고", "품목", "모델명", "규격", "product", "remark", "description", "item", "qty", "quantity"
    )
    if any(k in norm for k in excluded):
        return False
    clean = re.sub(r"[^\w가-힣]", "", name)
    if clean in ("주식회사", "유한회사", "합자회사", "합명회사", "사단법인", "재단법인"):
        return False
    return len(clean) >= 2


# ==============================================================================
# 📋 문서 항목별 정규식 및 라벨 추출 규칙 테이블 (Extraction Rules Registry)
# ==============================================================================
# 한 화면에서 문서 유형별 정규식과 키워드 라벨을 한눈에 파악하고 수정할 수 있는 중앙 규칙 테이블입니다.
EXTRACTION_RULES = {
    "사업자번호": {
        "labels": ("사업자번호", "등록번호", "사업자등록번호", "사업자 등록번호", "사업자"),
        "pattern": r"[0-9\s\-–—―─]{10,30}",
    },
    "상호명": {
        "labels": (
            "상호/대표", "상호 / 대표", "상호/대표자", "상호 / 대표자", "상호/성명", "상호 / 성명",
            "상호(법인명 및 사업체명)", "상호(법인명)", "상호[법인명]", "상호및성명", "상호 및 성명",
            "상호명", "상호", "상 호 명", "상 호", "법인명(단체명)", "법인명 (단체명)", "법인명[단체명]",
            "법인명 [단체명]", "법인명", "단체명", "업체명", "회사명", "협력사명", "협력사",
            "공급자", "발신", "발신처", "발행처", "발행자", "시공사", "시공자", "공사업체", "시공업체", "설치업체",
        ),
    },
    "대표자": {
        "labels": (
            "代 表", "代表", "회사명/대표", "상호/대표", "대표자(대표유형)", "대표자 (대표유형)",
            "대표자[대표유형]", "대표자 [대표유형]", "대표자", "대표자명", "대표자 성명", "대표이사", "대표", "성명",
        ),
    },
    "수신자": {
        "labels": (
            "수신자", "수신처", "수신", "귀하", "납품처", "발주처", "고객명", "고객사",
            "Messrs", "MESSRS", "Messrs.", "To", "TO", "to",
        ),
    },
    "담당자": {
        "labels": ("담당자", "담당자명", "예약 담당자", "예약담당자", "담당", "작성자", "발신인"),
    },
    "전화번호": {
        "labels": ("대표전화", "전화", "연락처", "휴대폰", "TEL/FAX", "TEL / FAX", "TEL", "tel", "Tel", "HP", "hp", "H.P", "Phone", "Mobile"),
        "pattern": r"[0-9() .+-]{8,}",
    },
    "이메일": {
        "labels": ("이메일", "email", "MAIL", "mail", "Mail", "E-MAIL", "E-mail"),
        "pattern": r"[\w.+-]+@[\w.-]+",
    },
    "통장사본": {
        "holder_labels": ("예금주", "예금주명", "예금주(성명/상호)"),
        "account_labels": ("계좌번호", "계좌", "계변호", "입금안내", "입금은행", "입금처"),
        "account_pattern": r"[0-9\s*\-]{6,}",
    },
    "견적일자": {
        "labels": ("견적일자", "견적일", "견적제출일", "제출일", "발행일자", "발행일", "작성일", "일자", "일 자"),
    },
    "견적총액": {
        "revised_labels": (
            "Revised Price", "Revised Amount", "RevisedPrice", "RevisedAmount",
            "최종 견적", "최종견적", "최종금액", "결제금액", "네고금액", "조정금액", "D/C적용금액",
        ),
        "vat_inclusive_labels": (
            "부가가치세(10%) 포함 총 공급 가액", "부가가치세(10%) 포함 총 공급가액",
            "부가가치세(10%)포함 총 공급 가액", "부가가치세(10%)포함 총 공급가액",
            "부가가치세 포함 총 공급 가액", "부가가치세 포함 총 공급가액",
            "부가가치세포함 총 공급 가액", "부가가치세포함 총 공급가액", "부가가치세 포함 총공급가액",
            "부가세 포함 총 공급가액", "부가세 포함 총공급가액", "부가세포함 총 공급가액", "부가세포함 총공급가액",
            "VAT 포함 총 공급가액", "VAT 포함 총공급가액", "VAT포함 총 공급가액", "VAT포함 총공급가액",
            "총계(VAT 포함)", "총계 (VAT 포함)", "총계(VAT포함)", "총계 (VAT포함)",
            "총계(부가세 포함)", "총계 (부가세 포함)", "총계(부가세포함)", "총계 (부가세포함)",
            "총계(부가가치세 포함)", "총계 (부가가치세 포함)", "총계(부가가치세포함)", "총계 (부가가치세포함)",
            "합계금액(부가세포함)", "합계금액 (부가세포함)", "합계금액(VAT포함)", "합계금액 (VAT포함)",
            "합계금액(부가가치세포함)", "합계금액 (부가가치세포함)",
            "합계(부가세포함)", "합계 (부가세포함)", "합계(VAT포함)", "합계 (VAT포함)",
            "합계(부가가치세포함)", "합계 (부가가치세포함)",
            "합계금(부가세포함)", "합계금 (부가세포함)", "합계금(VAT포함)", "합계금 (VAT포함)",
            "제안가(부가세포함)", "제안가 (부가세포함)", "제안가(VAT포함)", "제안가 (VAT포함)",
            "제안가격(부가세포함)", "제안가격 (부가세포함)", "제안가격(VAT포함)", "제안가격 (VAT포함)",
            "TOTAL (+VAT)", "TOTAL(+VAT)", "TOTAL (VAT)", "TOTAL(VAT)",
            "TOTAL AMOUNT. (VAT 포함)", "TOTAL AMOUNT (VAT 포함)", "TOTAL AMOUNT(VAT 포함)", "TOTAL AMOUNT (VAT포함)",
            "TOTAL AMOUNT.(VAT 포함)", "TOTAL AMOUNT.(VAT포함)",
            "예정금액(부가가치세포함)", "예정금액 (부가가치세포함)",
            "예정금액(부가세포함)", "예정금액 (부가세포함)",
            "예정금액부가가치세포함", "예정금액부가세포함",
            "공사예정금액부가가치세포함", "공사예정금액부가세포함",
            "아래와 같이 견적합니다", "아래와같이 견적합니다", "아래와 같이 견적하나이다",
            "견적금액(VAT포함)", "견적금액 (VAT포함)", "견적금액(부가세포함)", "견적금액 (부가세포함)",
            "견적금액", "견적 금액", "견적총액", "견적 총액",
        ),
        "general_total_labels": (
            "견적합계금액", "전체합계", "합계금액", "총합계금액", "총합계", "합계금",
            "합계(Total)", "합계 (Total)", "합계(total)", "합계 (total)",
            "제안공급가", "제안가격", "공급총액", "총 공급 가액", "총 공급가액", "총공급가액", "총공급대가",
            "예정금액", "공사예정금액", "설치예정금액", "사업비", "소요예산",
            "합계", "총계", "총액",
            "공급대가", "공사금액", "금액", "Total Amount", "TOTAL AMOUNT", "Total", "TOTAL",
        ),
        "vat_labels": ("부가세", "부가가치세", "세액", "VAT"),
        "subtotal_labels": (
            "공급가액", "공급가", "Subtotal", "소계",
            "합계금액(부가세별도)", "합계금액 (부가세별도)", "합계(부가세별도)", "합계 (부가세별도)",
            "합계금액(VAT별도)", "합계금액 (VAT별도)", "합계(VAT별도)", "합계 (VAT별도)",
            "제안가(부가세별도)", "제안가 (부가세별도)", "제안가(VAT별도)", "제안가 (VAT별도)",
            "제안가격(부가세별도)", "제안가격 (부가세별도)",
        ),
    },
}


def extract_document(
    text: str,
    filename: str = "",
    use_vendor_cache: bool = True,
    db_path: Optional[str] = None,
) -> PurchaseDocument:
    kind = classify_document(text, filename)
    document = PurchaseDocument(filename=filename, document_type=kind, raw_text=text)
    if kind not in (DocumentType.QUOTE, DocumentType.BUSINESS_LICENSE, DocumentType.BANK_COPY, DocumentType.CONSTRUCTION_CONFIRMATION, DocumentType.ONLINE_STORE_CART):
        document.warnings.append("문서 유형을 확정하지 못했습니다. 원문을 검토하세요.")
    own_prefix = re.sub(r"\D", "", my_business_prefix())

    # 1. 라벨(등록번호, 사업자번호 등) 기반 번호 탐색
    labeled_biz = _labeled(text, EXTRACTION_RULES["사업자번호"]["labels"], EXTRACTION_RULES["사업자번호"]["pattern"])
    selected_biz = ""
    biz_evidence_str = ""
    is_valid_biz = False

    if labeled_biz.raw:
        clean_digits = re.sub(r"\D", "", labeled_biz.raw)
        if len(clean_digits) == 10 and not clean_digits.startswith(own_prefix):
            normalized = normalize_business_number(clean_digits)
            selected_biz = normalized
            biz_evidence_str = labeled_biz.evidence
            is_valid_biz = is_valid_business_number(normalized)

    # 2. 라벨에서 번호를 못 찾았거나, 찾은 번호의 체크섬이 유효하지 않은 경우, 본문 전체에서 체크섬이 유효한 10자리 번호 우선 탐색
    if not selected_biz or not is_valid_biz:
        for m in _BIZ_RE.finditer(text):
            digits = re.sub(r"\D", "", m.group(1))
            if len(digits) == 10 and not digits.startswith(own_prefix):
                normalized = normalize_business_number(digits)
                if is_valid_business_number(normalized):
                    selected_biz = normalized
                    biz_evidence_str = m.group(0)
                    is_valid_biz = True
                    break

    if selected_biz:
        document.business_number = _evidence(selected_biz, biz_evidence_str, 0.9 if is_valid_biz else 0.55)
        if not is_valid_biz:
            document.warnings.append("사업자번호 형식은 발견했지만 체크섬이 유효하지 않습니다.")
    elif _BIZ_RE.search(text) and any(re.sub(r"\D", "", m.group(1)).startswith(own_prefix) for m in _BIZ_RE.finditer(text)):
        document.warnings.append("자사 사업자번호만 발견되어 업체 번호로 선택하지 않았습니다.")

    raw_vendor = Evidence()
    extracted_rep_from_vendor = ""
    if kind != DocumentType.BANK_COPY:
        raw_vendor = _labeled(
            text,
            EXTRACTION_RULES["상호명"]["labels"],
        )
        if raw_vendor.raw:
            cleaned_vendor, extracted_rep_from_vendor = _clean_vendor_name(raw_vendor.raw)
            if _is_vendor_name_candidate(cleaned_vendor):
                document.vendor_name = _evidence(cleaned_vendor, raw_vendor.evidence, raw_vendor.confidence)

    # 직접적인 공급자 상호 라벨 탐색 (업체명/수신처 등이 한 줄에 혼재되어 _labeled가 건너뛴 경우)
    if not document.vendor_name.raw:
        direct_sangho = re.search(r"(?<![가-힣])(?:상\s*호\s*명|상\s*호|법\s*인\s*명)[\s'’`.,·]*[:：.\-]?\s*([^\n]{2,50})", text)
        if direct_sangho:
            cand = re.split(r"(?:대\s*표|성\s*명|사\s*업|업\s*태|종\s*목|주\s*소|등\s*록|전\s*화|팩\s*스|홈\s*페|H\s*P|E-?Mail|경\s*기\s*도|서\s*울|인\s*천|대\s*전|대\s*구|부\s*산|광\s*주)", direct_sangho.group(1))[0].strip()
            cand = re.sub(r"^명\s+", "", cand).strip()
            cand, _ = _clean_vendor_name(cand)
            if _is_vendor_name_candidate(cand):
                document.vendor_name = _evidence(cand, direct_sangho.group(0).strip(), 0.88, inferred=True)

    if not document.vendor_name.raw:
        below_sangho = re.search(r"(?:^|\n)[^\n]*상\s*호[^\n]*\n+([가-힣\s]{2,15})\n+[^\n]*(?:법\s*인\s*명|사\s*업\s*장|주\s*소|업\s*태|종\s*목)", text)
        if below_sangho:
            cand_b = re.sub(r"\s+", "", below_sangho.group(1)).strip()
            if _is_vendor_name_candidate(cand_b):
                document.vendor_name = _evidence(cand_b, below_sangho.group(0).strip(), 0.90, inferred=True)

    if kind == DocumentType.BUSINESS_LICENSE and not document.vendor_name.raw:
        lic_vendor = re.search(r"(?<![가-힣])(?:상\s*호|호)[\s'’`.,·]*[:：.\-]?\s*([가-힣A-Za-z0-9()㈜]{2,20})", text)
        if lic_vendor:
            cand_v = re.split(r"(?:대\s*표|성\s*명|사\s*업|업\s*태|종\s*목|주\s*소|면)", lic_vendor.group(1))[0].strip()
            if _is_vendor_name_candidate(cand_v):
                document.vendor_name = _evidence(cand_v, lic_vendor.group(0).strip(), 0.85, inferred=True)

    if not document.vendor_name.raw:
        # 케이투엠플러스 / K2M+ 전용 탐색
        m_k2m = re.search(r"(?:^|[^\w가-힣])(케이투[엠엘]플러스|K2M\+)", text)
        if m_k2m:
            cand_k2m = m_k2m.group(1).replace("케이투엘플러스", "케이투엠플러스")
            document.vendor_name = _evidence(cand_k2m, m_k2m.group(0).strip(), 0.88, inferred=True)

    # 상호 라벨이 없을 경우 본문 전체에서 (주)회사명, ㈜회사명, 회사명(주), 회사명㈜, 주식회사 회사명 및 씨티엘(CTL) 등 패턴 자동 추론
    if not document.vendor_name.raw:
        # 1-1. 띄어쓰기 된 "주 식 회 사 아 이 티 스 톤" 패턴 먼저 탐색 (줄바꿈 방지)
        spaced_corp = re.search(r"주\s*식\s*회\s*사[^\S\n]*([가-힣A-Za-z0-9 ]{2,20})", text)
        if spaced_corp:
            cand_name = re.sub(r"\s+", "", spaced_corp.group(1))
            cand_name = re.split(r"(?:수신|견적|등록|사업|대표|일자|합계)", cand_name)[0].strip()
            cand = f"주식회사 {cand_name}"
            if _is_vendor_name_candidate(cand):
                document.vendor_name = _evidence(cand, spaced_corp.group(0).strip(), 0.82, inferred=True)

    if not document.vendor_name.raw:
        # 1-2. 띄어쓰기 된 법인명 패턴 (예: "㈜ 영 진 산 업 안 전" -> "(주)영진산업안전")
        spaced_prefix = re.search(r"(?:\(주\)|㈜|주\s*식\s*회\s*사)[^\S\n]*((?:[가-힣A-Za-z0-9][^\S\n]+){1,6}[가-힣A-Za-z0-9])(?=[^\S\n]*(?:$|\n|[^\w가-힣]))", text)
        if spaced_prefix:
            s_name = re.sub(r"\s+", "", spaced_prefix.group(1))
            cand = f"(주){s_name}"
            if _is_vendor_name_candidate(cand):
                document.vendor_name = _evidence(cand, spaced_prefix.group(0).strip(), 0.82, inferred=True)

    if not document.vendor_name.raw:
        # 1-3. 일반 법인명 패턴 (주)OOO, ㈜OOO 등 (줄바꿈 없이 한 단어 단위)
        corp_matches = re.finditer(
            r"(?:^|(?<=[^\w가-힣]))"
            r"((?:\(주\)|㈜|주\s*식\s*회\s*사)[^\S\n]*[가-힣A-Za-z0-9]{2,15}"
            r"|[가-힣A-Za-z0-9]{2,15}[^\S\n]*(?:\(주\)|㈜|주\s*식\s*회\s*사|주\s*[\]\)）]))"
            r"(?:$|(?=[^\w가-힣]))",
            text,
        )
        for cm in corp_matches:
            cand = _clean_spaces(cm.group(1))
            cand = re.sub(r"주\s*[\]\)）]$", "(주)", cand)
            if _is_vendor_name_candidate(cand):
                document.vendor_name = _evidence(cand, cm.group(0).strip(), 0.8, inferred=True)
                break

    if not document.vendor_name.raw:
        # 1-3. 괄호 약칭이 포함된 상호 패턴: 예) 씨티엘(CTL), 로듀(RODU), 팹솔(FabSol)
        brand_matches = re.finditer(
            r"(?:^|(?<=[^\w가-힣]))"
            r"([가-힣]{2,15}\s*\([A-Za-z0-9]+\))"
            r"(?:$|(?=[^\w가-힣]))",
            text,
        )
        for bm in brand_matches:
            cand = _clean_spaces(bm.group(1))
            if _is_vendor_name_candidate(cand):
                document.vendor_name = _evidence(cand, bm.group(0).strip(), 0.78, inferred=True)
                break

    if not document.vendor_name.raw:
        # 1-4. 영문 법인 상호 (예: ROOTSEMICON Co., Ltd.)
        eng_corp = re.search(r"([A-Za-z0-9\s&.,-]{2,30}\s+(?:Co\.,\s*Ltd\.|Co\.,Ltd\.|Inc\.|Corp\.|Corporation|Limited))", text)
        if eng_corp:
            cand = _clean_spaces(eng_corp.group(1)).strip()
            if _is_vendor_name_candidate(cand):
                document.vendor_name = _evidence(cand, eng_corp.group(0).strip(), 0.85, inferred=True)

    if not document.vendor_name.raw:
        # 1-6. 이메일 도메인 및 로고 헤더 기반 상호 매핑 (예: u-modern.co.kr -> 유모던컴퓨팅, kjsolution.com -> 경진솔루션)
        if "u-modern.co.kr" in text or "유모던" in text:
            document.vendor_name = _evidence("유모던컴퓨팅(주)", "u-modern.co.kr / 유모던 도메인 추론", 0.90, inferred=True)
        elif "kjsolution.com" in text or "kyungjin" in text.lower():
            document.vendor_name = _evidence("경진솔루션", "kjsolution.com / KYUNGJIN 도메인 추론", 0.90, inferred=True)

    document.representative = _labeled(
        text,
        EXTRACTION_RULES["대표자"]["labels"],
        validator=_is_representative_candidate,
    )
    if not document.representative.raw:
        rep_direct = re.search(
            rf"(?:성\s*명|대\s*표\s*자|대\s*표(?!\s*유\s*형)|代\s*表|代表)[\s'’`.,·]*[:：.\-]?\s*({_KOREAN_SURNAMES}[가-힣]{{1,3}}|[가-힣\u4e00-\u9fff\s]{{2,6}})(?!\w)",
            text,
        )
        if rep_direct:
            cand_rep = _clean_spaces(rep_direct.group(1)).strip()
            if _is_representative_candidate(cand_rep):
                document.representative = _evidence(cand_rep, rep_direct.group(0).strip(), 0.88, inferred=True)
    if not document.representative.raw:
        rep_suffix = re.search(
            rf"({_KOREAN_SURNAMES}[가-힣]{{1,3}})\s*대\s*표(?!\w)",
            text,
        )
        if rep_suffix:
            cand_rep = rep_suffix.group(1).strip()
            if _is_representative_candidate(cand_rep):
                document.representative = _evidence(cand_rep, rep_suffix.group(0).strip(), 0.88, inferred=True)
    if kind == DocumentType.BUSINESS_LICENSE and not document.representative.raw:
        lic_rep = re.search(rf"(?:대\s*표\s*자|대\s*표(?!\s*유\s*형)|표\s*자|성\s*명)[- \s'’`.,·:이$*#@%^&!/\\~_]*\s*({_KOREAN_SURNAMES}[가-힣]{{1,2}})", text)
        if lic_rep:
            cand_r = lic_rep.group(1).strip()
            if _is_representative_candidate(cand_r):
                document.representative = _evidence(cand_r, lic_rep.group(0).strip(), 0.85, inferred=True)

    if document.representative.raw:
        rep_clean = re.split(r"(?:생\s*년\s*월\s*일|주\s*민|개\s*업|사\s*업\s*장)", document.representative.raw)[0].strip()
        rep_clean = re.sub(r"\s*\(?(?:인|직인|서명|직인생략|직인\s*생략)\)?$", "", rep_clean).strip()
        if "/" in rep_clean:
            slash_r = [p.strip() for p in rep_clean.split("/") if p.strip()]
            if len(slash_r) >= 2 and _is_representative_candidate(slash_r[-1]):
                rep_clean = slash_r[-1]
        if rep_clean:
            document.representative = _evidence(rep_clean, document.representative.evidence, document.representative.confidence)
    if not document.representative.raw and extracted_rep_from_vendor:
        document.representative = _evidence(extracted_rep_from_vendor, f"상호 뒤 성명 분리: {raw_vendor.raw}", 0.78, inferred=True)

    document.recipient = _labeled(text, EXTRACTION_RULES["수신자"]["labels"])
    if document.recipient.raw:
        cleaned_rec = re.sub(r"^(?:Messrs\.?|To\b\.?|To\b|업\s*체\s*명|처\s*상\s*호|수\s*신\s*처\s*상\s*호|상\s*호|수\s*신\s*자?|고\s*객\s*명|고\s*객\s*사)\s*[:：.]?\s*", "", document.recipient.raw, flags=re.I).strip(" .:,;|-")
        cleaned_rec = re.split(r"(?:DATE\b|Date\b|귀\s*[하중]|貴\s*[中下]|상\s*호|공\s*급|등\s*록\s*번\s*호|사\s*업\s*자|견\s*적\s*제\s*출\s*일|제\s*출\s*일|\d\s*\d\s*\d\s*-)", cleaned_rec, flags=re.I)[0].strip(" .:,;|-")
        if cleaned_rec:
            document.recipient = _evidence(cleaned_rec, document.recipient.evidence, document.recipient.confidence)

    if not document.recipient.raw or not any(k in normalize_name(document.recipient.raw) for k in ("세종", "산단", "산학협력")):
        candidate_lines = []
        for line in text.splitlines():
            line_clean = re.sub(r"^[▪▶•\-\*\s]+", "", _clean_spaces(line)).strip()
            norm = normalize_name(line_clean)
            if ("세종" in norm or "산단" in norm or "산학협력" in norm) and not any(k in norm for k in ("공급자", "시공사", "발행처", "세무서", "견적서", "대관", "컨벤션센터", "운영관리")):
                candidate_lines.append(line_clean)
        
        # '산학협력단' 또는 '산단'이 포함된 라인 우선 선택
        chosen_line = None
        for cl in candidate_lines:
            if "산학협력" in normalize_name(cl) or "산단" in normalize_name(cl):
                chosen_line = cl
                break
        if not chosen_line and candidate_lines:
            chosen_line = candidate_lines[0]

        if chosen_line:
            cleaned_rec = re.sub(r"^[▪▶•\-\*\s]+", "", chosen_line)
            cleaned_rec = re.sub(r"^(?:Messrs\.?|To\b\.?|To\b|업\s*체\s*명|처\s*상\s*호|수\s*신\s*처\s*상\s*호|상\s*호\s*명|상\s*호|수\s*신\s*자?|고\s*객\s*명|고\s*객\s*사|회\s*사\s*명)\s*[:：.]?\s*", "", cleaned_rec, flags=re.I).strip(" .:,;|-")
            cleaned_rec = re.split(r"(?:DATE\b|Date\b|귀\s*[하중]|貴\s*[中下]|\s+상\s*호|공\s*급|등\s*록\s*번\s*호|사\s*업\s*자|견\s*적\s*제\s*출\s*일|제\s*출\s*일|\d\s*\d\s*\d\s*-)", cleaned_rec, flags=re.I)[0].strip(" .:,;|-")
            if cleaned_rec:
                document.recipient = _evidence(cleaned_rec, chosen_line, 0.85, inferred=True)

    document.contact_person = _labeled(
        text,
        EXTRACTION_RULES["담당자"]["labels"],
        validator=_is_contact_person_candidate,
    )
    if document.contact_person.raw:
        cp_val = document.contact_person.raw
        # 담당자 줄에 연락처/이메일이 함께 적혀있는 경우 분리
        found_phone = re.search(r"01[016789][- ]?\d{3,4}[- ]?\d{4}|02[- ]?\d{3,4}[- ]?\d{4}|0\d{2}[- ]?\d{3,4}[- ]?\d{4}", cp_val)
        found_email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", cp_val)
        cp_clean = re.split(r"[,/|]|(?:\b01[0-9])|(?:\b[A-Za-z0-9._%+-]+@)", cp_val)[0].strip()
        document.contact_person = _evidence(cp_clean, document.contact_person.evidence, document.contact_person.confidence)
        if found_phone:
            document.phone = _evidence(found_phone.group(0).strip(), cp_val, 0.9, inferred=True)
        if found_email:
            document.email = _evidence(found_email.group(0).strip(), cp_val, 0.9, inferred=True)

    if not document.phone.raw:
        document.phone = _labeled(text, EXTRACTION_RULES["전화번호"]["labels"], EXTRACTION_RULES["전화번호"]["pattern"])
    if not document.phone.raw:
        body_phone = re.search(r"(?:02|0[3-6]\d|01[016789])[- )]?\d{3,4}[- ]?\d{4}", text)
        if body_phone:
            document.phone = _evidence(body_phone.group(0).strip(), body_phone.group(0), 0.8, inferred=True)
    if document.phone.raw:
        cleaned_p = re.sub(r"^[^\d]+", "", document.phone.raw).strip()
        cleaned_p = re.sub(r"[^\d]+$", "", cleaned_p).strip()
        if cleaned_p:
            document.phone = _evidence(cleaned_p, document.phone.evidence, document.phone.confidence)

    if not document.email.raw:
        document.email = _labeled(text, EXTRACTION_RULES["이메일"]["labels"], EXTRACTION_RULES["이메일"]["pattern"])
    if not document.email.raw:
        body_email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        if body_email:
            document.email = _evidence(body_email.group(0).strip(), body_email.group(0), 0.85, inferred=True)
    document.account_holder = _labeled(text, EXTRACTION_RULES["통장사본"]["holder_labels"])
    if not document.account_holder.raw and kind == DocumentType.BANK_COPY:
        nim_match = re.search(r"([가-힣A-Za-z0-9()（）\[\]【】 \t]{2,30})\s*님", text)
        if nim_match:
            cand_holder = _clean_spaces(nim_match.group(1)).strip()
            cand_holder = re.split(r"(?:SWIFT|CODE|[A-Z]{6,})", cand_holder)[-1].strip()
            cand_holder = re.sub(r"[1\]】]$", ")", cand_holder)
            cand_holder = re.sub(r"[\[【]", "(", cand_holder)
            cand_holder = re.sub(r"^[高도주기]+\s*", "(주) ", cand_holder)
            cand_holder = _clean_spaces(cand_holder).strip()
            if not any(k in cand_holder for k in ("고객", "회원", "귀하", "은행", "통장")):
                document.account_holder = _evidence(cand_holder, nim_match.group(0).strip(), 0.85, inferred=True)
        # 1-2. (주)상호 / [주]상호 / [주상호 패턴 직접 탐색
        if not document.account_holder.raw:
            corp_holder = re.search(r"(?:^|\n)\s*((?:\(주\)|㈜|\[주\]|\[주|주\s*식\s*회\s*사)\s*[가-힣A-Za-z0-9]{2,15}|[가-힣A-Za-z0-9]{2,15}\s*(?:\(주\)|㈜|\[주\]))", text)
            if corp_holder:
                cand_ch = _clean_spaces(corp_holder.group(1)).strip()
                cand_ch = re.sub(r"^\[주\]?\s*", "(주)", cand_ch)
                if not any(k in cand_ch for k in ("은행", "통장", "과목", "계좌", "신규", "예금")):
                    document.account_holder = _evidence(cand_ch, corp_holder.group(0).strip(), 0.88, inferred=True)
        # 2. 통장 상단 첫 줄 또는 계좌번호/과목 앞의 상호/성명 탐색
        if not document.account_holder.raw:
            for line in text.splitlines()[:6]:
                clean_l = _clean_spaces(line).strip()
                if not clean_l or len(clean_l) < 2 or len(clean_l) > 25:
                    continue
                if any(k in clean_l for k in ("은행", "통장", "과목", "계좌", "금융", "신규", "지점", "인감", "서명", "Bank", "BANK", "예금", "종류", "거래")):
                    continue
                if not re.search(r"[\uac00-\ud7a3]", clean_l):
                    continue
                if re.match(r"^(?:(?:\(?주\)?|㈜|\[주\]?|주식회사)?\s*[가-힣A-Za-z0-9]{2,15}|[가-힣]{2,4})$", clean_l):
                    cand_h = clean_l
                    cand_h = re.sub(r"^\[주\]?\s*", "(주)", cand_h)
                    if cand_h.startswith("주") and not cand_h.startswith("주식") and not cand_h.startswith("(주)"):
                        cand_h = f"(주){cand_h[1:].strip()}"
                    document.account_holder = _evidence(cand_h, line, 0.85, inferred=True)
                    break

    if document.account_holder.raw:
        cleaned_h = re.sub(r"\s*님\s*$", "", document.account_holder.raw).strip()
        if cleaned_h.count("(") > cleaned_h.count(")"):
            cleaned_h += ")"
        if cleaned_h:
            document.account_holder = _evidence(cleaned_h, document.account_holder.evidence, document.account_holder.confidence)
    document.account_number = _labeled(text, EXTRACTION_RULES["통장사본"]["account_labels"], EXTRACTION_RULES["통장사본"]["account_pattern"])
    if not document.account_number.raw:
        acc_match = re.search(r"(?:계좌번호|계좌|계변호|입금계좌|입금안내|입금처)[^\d\n]{0,20}([0-9\-–—]{6,30})", text)
        if acc_match:
            acc_cand = acc_match.group(1).strip()
            if len(re.sub(r"\D", "", acc_cand)) >= 8:
                document.account_number = _evidence(acc_cand, acc_match.group(0).strip(), 0.90, inferred=True)
    if not document.account_number.raw and kind == DocumentType.BANK_COPY:
        hyphen_acc = re.search(r"\b(\d{2,6}(?:-\d{2,7}){2,4})\b", text)
        if hyphen_acc:
            cand_acc = hyphen_acc.group(1).strip()
            digits_only = re.sub(r"\D", "", cand_acc)
            if len(digits_only) >= 10 and not cand_acc.startswith("02-") and not cand_acc.startswith("010-") and not cand_acc.startswith("070-") and not cand_acc.startswith("050-") and not cand_acc.startswith("080-"):
                if not is_valid_business_number(cand_acc):
                    document.account_number = _evidence(cand_acc, hyphen_acc.group(0).strip(), 0.90, inferred=True)

    # 공식 온라인스토어 업체명 기본 매핑
    if kind == DocumentType.ONLINE_STORE_CART and not document.vendor_name.raw:
        text_lower = text.lower()
        if any(k in text_lower for k in ("apple", "애플", "mac mini", "macbook", "ipad", "applecare", "아이패드")):
            document.vendor_name = _evidence("애플코리아(Apple)", "Apple 공식 온라인스토어", 0.95, inferred=True)
        elif any(k in text_lower for k in ("samsung", "삼성", "갤럭시", "galaxy")):
            document.vendor_name = _evidence("삼성전자(Samsung)", "삼성전자 공식 온라인스토어", 0.95, inferred=True)
        else:
            document.vendor_name = _evidence("제조사 공식스토어", "공식 온라인스토어", 0.85, inferred=True)

    # 수의계약 업체선정확인서 병기 시 행정 정보 추가 추출
    if "업체선정확인서" in text or "수의계약" in text:
        m_bn = re.search(r"사업자등록번호\s*[:：]\s*([0-9\s-]{10,14})", text)
        if m_bn and not document.business_number.raw:
            c_bn = re.sub(r"\D", "", m_bn.group(1))
            if len(c_bn) == 10:
                document.business_number = _evidence(normalize_business_number(c_bn), m_bn.group(0).strip(), 0.92)
        m_rep = re.search(r"대표자(?:명)?\s*[:：]\s*([가-힣A-Za-z0-9()]+)", text)
        if m_rep and not document.representative.raw:
            document.representative = _evidence(m_rep.group(1).strip(), m_rep.group(0).strip(), 0.90)
        m_prof = re.search(r"연구책임자(?:[^\n:]*)[:：]\s*([가-힣]{2,4})", text)
        if m_prof and not document.contact_person.raw:
            document.contact_person = _evidence(m_prof.group(1).strip(), m_prof.group(0).strip(), 0.88)

    document.quote_date = _date_evidence(text, EXTRACTION_RULES["견적일자"]["labels"])
    if not document.quote_date.raw:
        date_matches = list(_DATE_RE.finditer(text))
        if date_matches:
            valid_dms = [m for m in date_matches if 2023 <= int(m.group(1)) <= 2030]
            dm = valid_dms[-1] if valid_dms else date_matches[0]
            document.quote_date = _evidence(f"{dm.group(1)}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}", dm.group(0), 0.75, inferred=True)
        else:
            m_rev = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일\s*(20\d{2})\s*년", text)
            if m_rev:
                document.quote_date = _evidence(f"{m_rev.group(3)}-{int(m_rev.group(1)):02d}-{int(m_rev.group(2)):02d}", m_rev.group(0), 0.85, inferred=True)
            else:
                m_eng = re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(20\d{2})\b", text, re.I)
                if m_eng:
                    months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
                    m_idx = months.index(m_eng.group(1).lower()) + 1
                    document.quote_date = _evidence(f"{m_eng.group(3)}-{m_idx:02d}-{int(m_eng.group(2)):02d}", m_eng.group(0), 0.85, inferred=True)

    if not document.contact_person.raw and document.representative.raw and (document.phone.raw or document.email.raw):
        document.contact_person = _evidence(document.representative.raw, f"대표자 겸 담당자 ({document.representative.raw})", 0.85, inferred=True)

    document.expiry_date = calculate_expiry(text, document.quote_date)
    if kind in (DocumentType.QUOTE, DocumentType.CONSTRUCTION_CONFIRMATION):
        total_rules = EXTRACTION_RULES["견적총액"]
        # 1차: 최종 수정/할인/네고 금액 라벨 우선 탐색
        document.total = _amount_after_label(
            text,
            total_rules["revised_labels"],
            exclude_keywords=("별도", "제외", "excl"),
        )
        # 2차: 명시적 부가세 포함 최종 합계 라벨 탐색
        if document.total.value is None:
            document.total = _amount_after_label(
                text,
                total_rules["vat_inclusive_labels"],
                exclude_keywords=("별도", "제외", "excl"),
            )
        # 3차: 일반 합계/총액 라벨 탐색
        if document.total.value is None:
            document.total = _amount_after_label(
                text,
                total_rules["general_total_labels"],
                exclude_keywords=("별도", "제외", "excl"),
            )
        document.vat = _amount_after_label(text, total_rules["vat_labels"])
        document.subtotal = _amount_after_label(
            text,
            total_rules["subtotal_labels"],
            exclude_keywords=("포함", "inc"),
        )
        document.items = _extract_items(text)
        if document.total.value is None and document.subtotal.value is not None and document.vat.value is not None:
            # 부가세가 소계의 약 10%(5%~15%) 범위일 때만 산술 추론 허용 (표 깨짐 오추론 방지)
            if document.subtotal.value > 0 and 0.05 <= (document.vat.value / document.subtotal.value) <= 0.15:
                value = document.subtotal.value + document.vat.value
                document.total = Amount(value, _evidence(str(value), "공급가액+부가세 산술 추론", 0.65, True), "추론 합계")
        elif document.subtotal.value is None and document.total.value is not None and document.vat.value is not None:
            # 공급가액이 누락되고 합계금액과 부가세가 확정된 경우 역추론
            calc_subtotal = document.total.value - document.vat.value
            if calc_subtotal > 0 and 0.05 <= (document.vat.value / calc_subtotal) <= 0.15:
                document.subtotal = Amount(calc_subtotal, _evidence(str(calc_subtotal), "합계금액-부가세 산술 추론", 0.65, True), "추론 공급가액")
        if document.total.value is None:
            # 금액 뒤에 "(VAT포함)" 또는 "부가세 포함"이 명시된 경우 탐색
            vat_inc_match = re.search(
                r"(?:[₩\\W]?\s*(\d{1,3}(?:,\d{3})+|\d{4,}))\s*(?:\(?\s*(?:부가세\s*포함|VAT\s*포함|V\.?A\.?T\.?\s*포함)\s*\)?)",
                text,
                re.I,
            )
            if vat_inc_match:
                raw_amt = vat_inc_match.group(1)
                clean_digits = re.sub(r"[^\d]", "", raw_amt)
                if clean_digits:
                    document.total = Amount(int(clean_digits), _evidence(raw_amt, vat_inc_match.group(0).strip(), 0.88, True), "VAT포함 추론")
        if document.total.value is None:
            document.warnings.append("라벨 근거가 있는 합계 금액을 찾지 못했습니다. 최대 숫자를 합계로 확정하지 않았습니다.")
        if not document.items and kind == DocumentType.QUOTE:
            document.warnings.append("비교 가능한 품목 행을 찾지 못했습니다.")
    elif kind == DocumentType.ONLINE_STORE_CART:
        # 공식 온라인스토어 장바구니 전용 추출 로직
        # 1. 총액 탐색
        # (1) 애플: 장바구니 총액: ₩3,440,000 (OCR 인식 변형: 싸5042000, w5042000 등 지원)
        m_cart_total = re.search(r"장바구니\s*총액\s*[:：]?\s*[₩\\Ww싸]?\s*(\d{1,3}(?:,\d{3})+|\d{4,})", text)
        if m_cart_total:
            c_val = int(m_cart_total.group(1).replace(",", ""))
            document.total = Amount(c_val, _evidence(m_cart_total.group(1), m_cart_total.group(0).strip(), 0.95), "장바구니 총액")

        # (2) 삼성닷컴: 결제 예정 금액 3,057,900 원 (할인 적용 후 최종 결제선)
        if document.total.value is None:
            m_sched = re.search(r"결제\s*예정\s*금액\s*[:：]?\s*[₩\\Ww]?\s*(\d{1,3}(?:,\d{3})+|\d{4,})", text)
            if m_sched:
                c_val = int(m_sched.group(1).replace(",", ""))
                document.total = Amount(c_val, _evidence(m_sched.group(1), m_sched.group(0).strip(), 0.95), "결제 예정 금액")

        # (3) 총계 ₩2,990,000
        if document.total.value is None:
            m_total_table = re.search(r"(?:^|\n)\s*총계\s*[:：]?\s*[₩\\Ww]?\s*(\d{1,3}(?:,\d{3})+|\d{4,})", text)
            if m_total_table:
                c_val = int(m_total_table.group(1).replace(",", ""))
                document.total = Amount(c_val, _evidence(m_total_table.group(1), m_total_table.group(0).strip(), 0.92), "총계")

        # (4) 소계 탐색
        m_sub = re.search(r"(?:^|\n)\s*소계\s*[:：]?\s*[₩\\Ww]?\s*(\d{1,3}(?:,\d{3})+|\d{4,})", text)
        if m_sub:
            sub_val = int(m_sub.group(1).replace(",", ""))
            document.subtotal = Amount(sub_val, _evidence(m_sub.group(1), m_sub.group(0).strip(), 0.90), "소계")

        # (5) VAT 탐색
        m_vat = re.search(r"[₩\\Ww]?\s*(\d{1,3}(?:,\d{3})+|\d{4,})\s*(?:원)?의?\s*(?:vat|부가세|부가가치세)\s*포함", text, re.I)
        if not m_vat:
            m_vat = re.search(r"(\d{1,3}(?:,\d{3})+|\d{4,})\s*VAT", text, re.I)
        if m_vat:
            document.vat = Amount(int(m_vat.group(1).replace(",", "")), _evidence(m_vat.group(1), m_vat.group(0).strip(), 0.90), "VAT 포함")

        # 품목 추출
        document.items = _extract_online_cart_items(text)

        # 품목은 있는데 총액이 아직 없으면 품목 합산 또는 소계로 산출
        if document.total.value is None and document.items:
            valid_amts = [it.amount for it in document.items if it.amount]
            if valid_amts:
                sum_val = sum(valid_amts)
                document.total = Amount(sum_val, _evidence(str(sum_val), "장바구니 품목 합산", 0.88, True), "품목 합산")
        elif document.total.value is None and document.subtotal.value is not None:
            document.total = Amount(document.subtotal.value, _evidence(str(document.subtotal.value), "장바구니 소계 기준 총액", 0.85, True), "소계 기준 총액")

        if document.total.value is None:
            document.warnings.append("공식스토어 장바구니 화면에서 총액을 명확히 찾지 못했습니다.")
        if not document.items:
            document.warnings.append("공식스토어 장바구니 화면에서 품목을 찾지 못했습니다.")

    # 거래처 마스터 DB(캐시) 보완 연동
    if use_vendor_cache:
        if document.business_number.raw and is_valid_business_number(document.business_number.raw):
            from purchase_verifier.vendor_cache import get_vendor_by_biz_num
            cached_vendor = get_vendor_by_biz_num(document.business_number.raw, db_path=db_path)
            if cached_vendor:
                # 1. 상호명 누락 시 캐시된 정식 상호명 보완
                if not document.vendor_name.raw and cached_vendor.get("vendor_name"):
                    document.vendor_name = _evidence(
                        cached_vendor["vendor_name"],
                        f"[거래처 마스터 DB 참조] (사업자번호: {document.business_number.raw})",
                        0.95,
                        inferred=True,
                    )
                # 2. 대표자 누락 시 캐시된 대표자 보완
                if not document.representative.raw and cached_vendor.get("representative"):
                    document.representative = _evidence(
                        cached_vendor["representative"],
                        f"[거래처 마스터 DB 참조] (대표자: {cached_vendor['representative']})",
                        0.90,
                        inferred=True,
                    )
                # 3. 통장사본 계좌번호 누락 시 캐시된 계좌 보완
                if kind == DocumentType.BANK_COPY and not document.account_number.raw and cached_vendor.get("accounts"):
                    primary_acc = cached_vendor["accounts"][0]
                    document.account_number = _evidence(
                        primary_acc.get("account", ""),
                        f"[거래처 마스터 DB 참조] ({primary_acc.get('bank', '')} {primary_acc.get('account', '')})",
                        0.90,
                        inferred=True,
                    )
        elif document.vendor_name.raw and not document.business_number.raw:
            from purchase_verifier.vendor_cache import get_vendor_by_name
            cached_vendor = get_vendor_by_name(document.vendor_name.raw, db_path=db_path)
            if cached_vendor and cached_vendor.get("business_number"):
                document.business_number = _evidence(
                    cached_vendor["business_number"],
                    f"[거래처 마스터 DB 참조] (상호명: {document.vendor_name.raw})",
                    0.90,
                    inferred=True,
                )

    return document


def _value(field: Evidence) -> str:
    return _clean_spaces(field.raw)


def are_item_sets_comparable(items1: list[Item], items2: list[Item]) -> bool:
    if not items1 or not items2:
        return False
    if len(items1) != len(items2):
        return False

    def _get_pc_component_type(name: str) -> str:
        n = name.lower()
        if any(k in n for k in ("수냉", "수랭", "cpu쿨러", "수냉쿨러", "쿨러")):
            return "COOLER"
        if any(k in n for k in ("인텔", "amd", "라이젠", "ryzen", "285k", "cpu")) and not any(k in n for k in ("쿨러", "보드")):
            return "CPU"
        if any(k in n for k in ("메인보드", "보드", "z890", "z790", "b760", "b650", "x670", "x870", "wifi")) and not any(k in n for k in ("cpu", "vga", "쿨러")):
            return "MAINBOARD"
        if any(k in n for k in ("ddr5", "ddr4", "ram", "메모리")):
            return "RAM"
        if any(k in n for k in ("ssd", "nvme", "m.2", "m2", "evoplus", "990evo")):
            return "SSD"
        if any(k in n for k in ("rtx", "geforce", "지포스", "vga", "gpu", "그래픽")):
            return "GPU"
        if any(k in n for k in ("케이스", "case", "미들타워", "빅타워", "t3000")):
            return "CASE"
        if any(k in n for k in ("파워", "psu", "power", "hx-1500", "hx1500")):
            return "POWER"
        return ""

    def _item_model_codes(name: str) -> set[str]:
        return set(re.findall(r"[A-Za-z]{1,5}\d{2,6}[A-Za-z0-9]*|\b[A-Za-z0-9]+-[A-Za-z0-9]+\b|\b\d{3,4}[A-Za-z]+\b|\b\d+tb\b|\b\d+gb\b", name.lower()))

    used2 = set()
    for it1 in items1:
        m1 = _item_model_codes(it1.name)
        n1 = normalize_name(it1.name)
        c1 = _get_pc_component_type(it1.name)
        match_idx = -1
        for idx, it2 in enumerate(items2):
            if idx in used2:
                continue
            m2 = _item_model_codes(it2.name)
            n2 = normalize_name(it2.name)
            c2 = _get_pc_component_type(it2.name)

            # 1. 모델 번호 / 주요 식별 코드 일치 (예: FPC033, PN1550R5A1, RXM10AF, 285K, Z890, RTX5090 등)
            common = m1 & m2
            significant = {c for c in common if not c.endswith("gb") and not c.endswith("tb")}
            if significant or (m1 and m2 and (m1 & m2)):
                match_idx = idx
                break

            # 2. 유사도 또는 부분 일치
            sim = difflib.SequenceMatcher(None, n1, n2).ratio()
            if sim >= 0.60 or (len(n1) >= 4 and n1 in n2) or (len(n2) >= 4 and n2 in n1):
                match_idx = idx
                break

            # 3. 동일 컴퓨터 부품 카테고리 (수냉 쿨러 등 동급 대체품 매칭)
            if c1 and c2 and c1 == c2:
                match_idx = idx
                break

        if match_idx != -1:
            used2.add(match_idx)
        else:
            return False
    return len(used2) == len(items1)


def compare_documents(documents: list[PurchaseDocument]) -> list[CheckResult]:
    if not documents:
        return [CheckResult("documents", VerificationStatus.MISSING, "문서가 없습니다.")]
    quotes = [doc for doc in documents if doc.document_type in (DocumentType.QUOTE, DocumentType.ONLINE_STORE_CART)]
    results: list[CheckResult] = []
    if not quotes:
        results.append(CheckResult("견적서", VerificationStatus.MISSING, "견적서가 없습니다."))
        return results

    if len(quotes) > 1:
        vendor_names = []
        for d in quotes:
            v_name = _value(d.vendor_name) or d.filename
            if d.document_type == DocumentType.ONLINE_STORE_CART:
                vendor_names.append(f"{v_name}(공식스토어 장바구니)")
            else:
                vendor_names.append(v_name)
        unique_vendors = {normalize_name(v) for v in vendor_names if v}
        if len(unique_vendors) > 1:
            results.append(CheckResult(
                "비교견적 현황",
                VerificationStatus.PASS,
                f"서로 다른 업체의 비교견적 {len(quotes)}건 확인 ({', '.join(v for v in vendor_names if v)})"
            ))
        else:
            results.append(CheckResult(
                "비교견적 현황",
                VerificationStatus.REVIEW,
                f"동일 업체 견적서 {len(quotes)}건 확인 (수정 견적 여부 확인 필요)"
            ))

        all_have_items = all(len(q.items) > 0 for q in quotes)
        all_same_items = False
        if all_have_items and len(quotes) > 1:
            item_sets = [tuple(sorted(normalize_name(item.name) for item in q.items)) for q in quotes]
            if len(set(item_sets)) == 1:
                all_same_items = True
            else:
                base_items = quotes[0].items
                all_same_items = all(are_item_sets_comparable(base_items, q.items) for q in quotes[1:])

        if all_same_items:
            results.append(CheckResult(
                "품목 구성",
                VerificationStatus.PASS,
                f"견적서 간 동일 품목 구성 확인 (총 {len(quotes[0].items)}개 품목 일치)",
            ))
        else:
            q_items_summary = []
            for q in quotes:
                it_names = [it.name for it in q.items]
                v_name = _value(q.vendor_name) or q.filename
                summary_str = f"[{v_name}] " + (f"{len(it_names)}개 품목({', '.join(it_names[:3])}{'...' if len(it_names) > 3 else ''})" if it_names else "미추출")
                q_items_summary.append(summary_str)
            items_detail_str = " vs ".join(q_items_summary)
            results.append(CheckResult(
                "품목 구성",
                VerificationStatus.REVIEW,
                f"견적서 간 품목 구성이 다르거나 품목 추출이 불완전합니다. (실제 OCR 결과값: {items_detail_str})",
            ))
    else:
        q_doc = quotes[0]
        if q_doc.document_type == DocumentType.ONLINE_STORE_CART:
            results.append(CheckResult("공식스토어 장바구니", VerificationStatus.PASS, f"공식스토어 장바구니 견적대체 확인 ({q_doc.filename})"))
        else:
            results.append(CheckResult("견적서", VerificationStatus.PASS, f"단일 견적서 확인 ({q_doc.filename})"))
        if q_doc.items:
            results.append(CheckResult("품목 구성", VerificationStatus.PASS, f"{len(q_doc.items)}개 품목 추출 확인"))
        else:
            results.append(CheckResult("품목 구성", VerificationStatus.REVIEW, "품목 행을 명확히 추출하지 못했습니다. (실제 OCR 결과값: 미추출)"))

    return results


def find_lowest_quote(documents: list[PurchaseDocument]) -> LowestPriceResult:
    quotes = [d for d in documents if d.document_type in (DocumentType.QUOTE, DocumentType.ONLINE_STORE_CART) and d.total.value is not None]
    if not quotes:
        return LowestPriceResult(None, None, False, "확정 합계가 있는 견적서가 없습니다.")

    comparable = False
    if all(len(q.items) > 0 for q in quotes):
        item_sets = [tuple(sorted(normalize_name(item.name) for item in q.items)) for q in quotes]
        if len(set(item_sets)) == 1:
            comparable = True
        elif len(quotes) > 1:
            base_items = quotes[0].items
            comparable = all(are_item_sets_comparable(base_items, q.items) for q in quotes[1:])

    candidate = min(quotes, key=lambda q: q.total.value or 0)
    confirmed = comparable and all(not q.total.evidence.inferred for q in quotes)
    reason = "확정 합계와 동일 품목 구성" if confirmed else "후보만 표시: 합계 또는 품목 구성이 불확실합니다."
    return LowestPriceResult(candidate.filename, candidate.total.value, confirmed, reason)


def _lowest_quote_candidate(documents: list[PurchaseDocument]) -> Optional[PurchaseDocument]:
    quotes = [d for d in documents if d.document_type in (DocumentType.QUOTE, DocumentType.ONLINE_STORE_CART) and d.total.value is not None]
    if not quotes:
        return None
    return min(quotes, key=lambda q: q.total.value or 0)


def _compare_field(
    label: str,
    expected: str,
    actual: str,
    normalizer: Callable[[str], str],
    doc_name: str,
    policy: FieldPolicy = FieldPolicy.EXACT,
) -> Optional[CheckResult]:
    if not expected and not actual:
        return None
    if not expected or not actual:
        missing = "기준 견적" if not expected else doc_name
        exp_disp = f"'{expected}'" if expected else "미추출"
        act_disp = f"'{actual}'" if actual else "미추출"
        return CheckResult(
            label,
            VerificationStatus.REVIEW,
            f"{missing}에 값이 없어 {label} 대조를 완료하지 못했습니다. (실제 OCR 결과값: 기준 견적={exp_disp} vs {doc_name}={act_disp})",
            expected,
            actual,
        )
    
    # 마스킹(*) 문자 포함 시 수동 확인(REVIEW) 유도
    if "*" in expected or "*" in actual:
        return CheckResult(
            label,
            VerificationStatus.REVIEW,
            f"마스킹(*)된 정보가 포함되어 있어 {label} 수동 확인이 필요합니다. (실제 OCR 결과값: 기준 견적='{expected}' vs {doc_name}='{actual}')",
            expected,
            actual,
        )

    norm_expected = normalizer(expected)
    norm_actual = normalizer(actual)
    
    if norm_expected and norm_actual and norm_expected == norm_actual:
        return CheckResult(label, VerificationStatus.PASS, f"{label} 일치.", expected, actual)
    
    if policy == FieldPolicy.EXACT:
        return CheckResult(label, VerificationStatus.CONFLICT, f"{label} 불일치: 기준 견적='{expected}', {doc_name}='{actual}'.", expected, actual)

    # TEXT_REVIEW 정책 (상호, 대표자명 등 텍스트 전용)
    is_subset = (norm_expected in norm_actual) or (norm_actual in norm_expected)
    similarity = difflib.SequenceMatcher(None, norm_expected, norm_actual).ratio()
    
    if (similarity >= 0.75 or is_subset) and len(norm_expected) >= 2 and len(norm_actual) >= 2:
        match_type = f"유사도 {similarity*100:.1f}%" if not is_subset else "부분 일치(포함 관계)"
        detail_msg = (f"{match_type} 발견 (완전히 일치함을 의미하지 않습니다).\n"
                      f"실제 OCR 결과값: [기준 견적] '{expected}' / [{doc_name}] '{actual}'")
        return CheckResult(label, VerificationStatus.REVIEW, detail_msg, expected, actual)
    
    return CheckResult(label, VerificationStatus.CONFLICT, f"{label} 불일치: 기준 견적='{expected}', {doc_name}='{actual}'.", expected, actual)


def verify_documents(
    documents: list[PurchaseDocument],
    purchase_type: str = "물품",
    selected_filename: Optional[str] = None,
) -> list[CheckResult]:
    results = compare_documents(documents)
    license_doc = next((d for d in documents if d.document_type == DocumentType.BUSINESS_LICENSE), None)
    bank_doc = next((d for d in documents if d.document_type == DocumentType.BANK_COPY), None)
    confirm_doc = next((d for d in documents if d.document_type == DocumentType.CONSTRUCTION_CONFIRMATION), None)

    quotes = [d for d in documents if d.document_type in (DocumentType.QUOTE, DocumentType.ONLINE_STORE_CART)]
    standard_quotes = [d for d in documents if d.document_type == DocumentType.QUOTE]
    cart_quotes = [d for d in documents if d.document_type == DocumentType.ONLINE_STORE_CART]

    target: Optional[PurchaseDocument] = None
    if selected_filename:
        target = next((q for q in quotes if q.filename == selected_filename), None)
    if not target:
        target = _lowest_quote_candidate(documents)

    required = ((DocumentType.BUSINESS_LICENSE, "사업자등록증", license_doc), (DocumentType.BANK_COPY, "통장사본", bank_doc))
    if target and target.document_type == DocumentType.ONLINE_STORE_CART:
        results.append(CheckResult("공식스토어 견적대체", VerificationStatus.PASS, f"제조사 공식스토어({target.vendor_name.raw or '공식몰'}) 장바구니 대체 승인 (법인카드 결제 또는 수의계약 대상)"))
        for _kind, name, doc in required:
            if doc:
                results.append(CheckResult(name, VerificationStatus.PASS, "문서 확인"))
            else:
                results.append(CheckResult(name, VerificationStatus.PASS, f"제조사 공식스토어 직접 구매 건으로 {name} 제출 생략 가능 (법인카드 결제 또는 수의계약 사유서 갈음)"))
    else:
        for _kind, name, doc in required:
            results.append(CheckResult(name, VerificationStatus.PASS if doc else VerificationStatus.MISSING, "문서 확인" if doc else "필수 문서가 없습니다."))

    if purchase_type == "공사":
        if confirm_doc:
            results.append(CheckResult("공사 확인서", VerificationStatus.PASS, "시설공사/장비설치/공사 확인서 첨부 확인"))
            # 협력사명 일치 검증
            if target and _value(target.vendor_name) and _value(confirm_doc.vendor_name):
                c_vendor_check = _compare_field("공사 확인서 협력사명", _value(target.vendor_name), _value(confirm_doc.vendor_name), normalize_name, "공사 확인서", FieldPolicy.TEXT_REVIEW)
                if c_vendor_check:
                    results.append(c_vendor_check)
            # 공사예정금액 일치 검증
            if target and target.total.value is not None and confirm_doc.total.value is not None:
                amt_match = target.total.value == confirm_doc.total.value
                results.append(CheckResult(
                    "공사 확인서 금액",
                    VerificationStatus.PASS if amt_match else VerificationStatus.CONFLICT,
                    f"공사 확인서 예정금액({confirm_doc.total.value:,}원)과 견적서 합계({target.total.value:,}원) 일치." if amt_match else f"금액 불일치: 견적서={target.total.value:,}원 vs 공사확인서={confirm_doc.total.value:,}원",
                    f"{target.total.value:,}원",
                    f"{confirm_doc.total.value:,}원",
                ))
        else:
            results.append(CheckResult(
                "공사 확인서",
                VerificationStatus.MISSING,
                "반려 대상: 공사 선택 시 '시설공사' 또는 '장비설치 확인서' 첨부가 필수입니다."
            ))

    # 1. 첨부된 모든 일반 견적서(비교견적서 포함) 수신자 명의 전수 검사
    if standard_quotes:
        blocked_recs = []
        warning_recs = []
        for q in standard_quotes:
            q_rec = _value(q.recipient)
            if not q_rec:
                warning_recs.append(f"'{q.filename}': 수신자 미기재")
                continue
            norm_q = normalize_name(q_rec)
            if "산학협력단" in norm_q or "산단" in norm_q or bool(re.search(r"산\s*학\s*협\s*[력려련]\s*단?", norm_q)):
                continue
            elif "세종" in norm_q or "세종대" in norm_q:
                blocked_recs.append(f"'{q.filename}': 세종대학교(본교) 명의 ('{q_rec}')")
            else:
                warning_recs.append(f"'{q.filename}': 산단 외 수신자 ('{q_rec}')")

        if blocked_recs:
            results.append(CheckResult(
                "수신자 산학협력단",
                VerificationStatus.BLOCKED,
                f"반려 대상: 독립 법인인 세종대학교(본교) 명의로만 발행된 견적서가 있습니다. ('세종대학교 산학협력단'으로 재발행 필요)\n- " + "\n- ".join(blocked_recs),
                "세종대학교 산학협력단",
                ", ".join(blocked_recs),
            ))
        elif warning_recs:
            results.append(CheckResult(
                "수신자 산학협력단",
                VerificationStatus.REVIEW,
                f"확인 필요: '산학협력단' 명의가 누락되었거나 불명확한 견적서가 있습니다.\n- " + "\n- ".join(warning_recs),
                "세종대학교 산학협력단",
                ", ".join(warning_recs),
            ))
        else:
            results.append(CheckResult(
                "수신자 산학협력단",
                VerificationStatus.PASS,
                f"첨부된 모든 견적서(총 {len(standard_quotes)}건)의 수신자가 '세종대학교 산학협력단' 명의로 정상 확인되었습니다.",
                "세종대학교 산학협력단",
                "세종대학교 산학협력단",
            ))
    elif cart_quotes:
        results.append(CheckResult(
            "수신자 산학협력단",
            VerificationStatus.PASS,
            "제조사 공식스토어 장바구니 캡처본은 웹 화면 특성상 산학협력단 수신자 표기 생략 인정 (배송지 주소로 갈음)",
            "세종대학교 산학협력단",
            "공식스토어 배송지로 갈음",
        ))

    # 공식스토어 배송지 주소 확인
    if cart_quotes:
        for cq in cart_quotes:
            has_univ = any(k in cq.raw_text for k in ("세종대", "세종대학교", "능동로", "군자동"))
            if has_univ:
                results.append(CheckResult(
                    "공식스토어 배송지",
                    VerificationStatus.PASS,
                    f"장바구니 배송지: 세종대학교 교내 연구실 주소 확인 완료 ({cq.filename})",
                    "세종대학교",
                    "세종대학교 확인",
                ))
            else:
                results.append(CheckResult(
                    "공식스토어 배송지",
                    VerificationStatus.REVIEW,
                    f"장바구니 캡처본에 세종대학교 교내 배송지 표기 여부 수동 확인 요망 ({cq.filename})",
                    "세종대학교",
                    "미검출 또는 수동 확인",
                ))

    if target:
        if target.document_type == DocumentType.ONLINE_STORE_CART:
            results.append(CheckResult(
                "견적서 유효기간",
                VerificationStatus.PASS,
                "제조사 공식스토어 장바구니 실시간 캡처본 (주문 결제 시점 유효)",
                "실시간 캡처 유효",
                "주문 시점 유효",
            ))
        else:
            # 2. 견적서 유효기간 (5영업일 기준) 검증
            if target.expiry_date and target.expiry_date.raw:
                exp_date = _parse_date(target.expiry_date.raw)
                if exp_date:
                    today = date.today()
                    b_days = calculate_business_days(today, exp_date)
                    if b_days >= 5:
                        results.append(CheckResult(
                            "견적서 유효기간",
                            VerificationStatus.PASS,
                            f"유효기간 정상 확인 (만료일: {target.expiry_date.raw}, 잔여 {b_days}영업일)",
                            "5영업일 이상 잔여",
                            f"{b_days}영업일 잔여",
                        ))
                    elif b_days >= 0:
                        results.append(CheckResult(
                            "견적서 유효기간",
                            VerificationStatus.BLOCKED,
                            f"반려 대상: 견적서 유효기간이 5영업일 미만 남았습니다! (만료일: {target.expiry_date.raw}, 잔여 {b_days}영업일)",
                            "5영업일 이상 잔여",
                            f"{b_days}영업일 잔여",
                        ))
                    else:
                        results.append(CheckResult(
                            "견적서 유효기간",
                            VerificationStatus.BLOCKED,
                            f"반려 대상: 견적서 유효기간이 이미 지났습니다! (만료일: {target.expiry_date.raw}, {abs(b_days)}영업일 경과)",
                            "5영업일 이상 잔여",
                            f"{abs(b_days)}영업일 지남",
                        ))
                else:
                    results.append(CheckResult(
                        "견적서 유효기간",
                        VerificationStatus.REVIEW,
                        f"유효기간 파싱 불가. 수동 확인 권장. (실제 OCR 결과값: 유효기간 원문='{target.expiry_date.raw}')",
                        "유효기간 확인",
                        target.expiry_date.raw,
                    ))
            else:
                q_date_disp = target.quote_date.raw if target.quote_date.raw else "미기재"
                results.append(CheckResult(
                    "견적서 유효기간",
                    VerificationStatus.REVIEW,
                    f"견적서에 유효기간이 명시되지 않아 무기한으로 간주합니다. (확인 요망) (실제 OCR 결과값: 견적일자='{q_date_disp}', 유효기간=미기재)",
                    "유효기간 확인",
                    "미명시 (무기한 간주)",
                ))

        # 3. 사업자등록증 대조
        if license_doc and target.document_type != DocumentType.ONLINE_STORE_CART:
            name_check = _compare_field("사업자등록증 업체명", _value(target.vendor_name), _value(license_doc.vendor_name), normalize_name, "사업자등록증", FieldPolicy.TEXT_REVIEW)
            if name_check:
                target_bn = re.sub(r"\D", "", _value(target.business_number))
                lic_bn = re.sub(r"\D", "", _value(license_doc.business_number))
                if name_check.status == VerificationStatus.CONFLICT and target_bn and lic_bn and target_bn == lic_bn:
                    v_sim = difflib.SequenceMatcher(None, normalize_name(_value(target.vendor_name)), normalize_name(_value(license_doc.vendor_name))).ratio()
                    if v_sim >= 0.6:
                        name_check = CheckResult(
                            "사업자등록증 업체명",
                            VerificationStatus.REVIEW,
                            f"사업자등록번호는 일치하나 상호 OCR 유사도({v_sim*100:.1f}%)로 수동 확인 권장.\n실제 OCR 결과값: [기준 견적] '{_value(target.vendor_name)}' / [사업자등록증] '{_value(license_doc.vendor_name)}'",
                            _value(target.vendor_name),
                            _value(license_doc.vendor_name),
                        )
                results.append(name_check)
            number_check = _compare_field("사업자등록증 사업자번호", _value(target.business_number), _value(license_doc.business_number), lambda v: re.sub(r"\D", "", v), "사업자등록증", FieldPolicy.EXACT)
            if number_check:
                results.append(number_check)
            rep_check = _compare_field("사업자등록증 대표자명", _value(target.representative), _value(license_doc.representative), normalize_name, "사업자등록증", FieldPolicy.TEXT_REVIEW)
            if rep_check:
                results.append(rep_check)

        # 4. 통장사본 대조
        if bank_doc and target.document_type != DocumentType.ONLINE_STORE_CART:
            holder_check = _compare_field("통장사본 예금주", _value(target.vendor_name), _value(bank_doc.account_holder), normalize_name, "통장사본", FieldPolicy.TEXT_REVIEW)
            
            # 개인사업자 통장사본 예금주 형태 대조: "대표자(상호)" 또는 "상호(대표자)"
            holder_raw = _value(bank_doc.account_holder)
            target_rep = _value(target.representative) or (_value(license_doc.representative) if license_doc else "")
            target_vendor = _value(target.vendor_name)
            
            is_corp_bankbook = any(k in bank_doc.raw_text for k in ("기업자유예금", "기업통장", "사업자통장", "법인통장", "가맹점통장"))
            
            def _is_korean_name_match(n1: str, n2: str) -> bool:
                norm1 = normalize_name(n1)
                norm2 = normalize_name(n2)
                if not norm1 or not norm2:
                    return False
                if norm1 == norm2:
                    return True
                if len(norm1) == 3 and len(norm2) == 3:
                    if sum(1 for a, b in zip(norm1, norm2) if a == b) >= 2:
                        return True
                return difflib.SequenceMatcher(None, norm1, norm2).ratio() >= 0.65

            if holder_check and holder_check.status in (VerificationStatus.CONFLICT, VerificationStatus.REVIEW) and holder_raw:
                holder_parts = re.findall(r"[\w가-힣]+", holder_raw)
                if len(holder_parts) >= 2 and target_rep:
                    has_rep = any(_is_korean_name_match(target_rep, p) for p in holder_parts)
                    vendor_sims = [difflib.SequenceMatcher(None, normalize_name(target_vendor), normalize_name(p)).ratio() for p in holder_parts]
                    max_v_sim = max(vendor_sims) if vendor_sims else 0.0
                    
                    if has_rep and max_v_sim >= 0.6:
                        holder_check = CheckResult(
                            "통장사본 예금주",
                            VerificationStatus.PASS,
                            f"개인사업자 사업용 계좌 확인 (대표자 '{target_rep}' 및 상호 '{target_vendor}' 병기 일치, 예금주: '{holder_raw}')",
                            f"{target_vendor} ({target_rep})",
                            holder_raw,
                        )

            # 기업자유예금/기업통장인 경우 또는 대표자 단독 명의 판별
            if holder_check and holder_check.status in (VerificationStatus.CONFLICT, VerificationStatus.REVIEW) and (license_doc or target.representative.raw):
                rep_name_raw = _value(license_doc.representative) if license_doc else _value(target.representative)
                if rep_name_raw:
                    rep_candidates = re.split(r"/|,|외", rep_name_raw)
                    for rep in rep_candidates:
                        clean_rep = rep.strip()
                        if not clean_rep: 
                            continue
                        rep_matched = _is_korean_name_match(clean_rep, holder_raw)
                        vendor_matched = _is_korean_name_match(target_vendor, holder_raw)
                        
                        if is_corp_bankbook and (rep_matched or vendor_matched):
                            holder_check = CheckResult(
                                "통장사본 예금주",
                                VerificationStatus.PASS,
                                f"개인사업자 기업통장(기업자유예금) 계좌 확인 (예금주: '{holder_raw}', 대표자: '{clean_rep}')",
                                clean_rep,
                                holder_raw,
                            )
                            break
                        elif not is_corp_bankbook and rep_matched:
                            holder_check = CheckResult(
                                "통장사본 예금주", 
                                VerificationStatus.BLOCKED,
                                f"반려 대상: 상호명이 누락되고 대표자 단독 명의('{clean_rep}')로만 기재된 개인 통장입니다.\n"
                                f"실제 OCR 결과: [견적서 상호] {_value(target.vendor_name)} / [통장사본 예금주] {_value(bank_doc.account_holder)}",
                                _value(target.vendor_name),
                                _value(bank_doc.account_holder)
                            )
                            break
            if holder_check:
                results.append(holder_check)
            account_check = _compare_field("통장사본 계좌번호", _value(target.account_number), _value(bank_doc.account_number), lambda v: re.sub(r"\D", "", v), "통장사본", FieldPolicy.EXACT)
            if account_check:
                results.append(account_check)

        if target.document_type == DocumentType.ONLINE_STORE_CART:
            results.append(CheckResult("공식스토어 고객지원", VerificationStatus.PASS, f"제조사 공식 고객지원/스토어 결제 채널 이용 대상 ({target.vendor_name.raw})"))
        else:
            contact_fields = (
                ("담당자", target.contact_person),
                ("전화번호", target.phone),
                ("이메일", target.email),
            )
            for label, evidence in contact_fields:
                if not evidence.raw:
                    target_label = f"선택 견적 '{target.filename}'" if selected_filename else f"최저가 견적 '{target.filename}'"
                    results.append(CheckResult(
                        label,
                        VerificationStatus.REVIEW,
                        f"{target_label}에 {label} 정보가 없습니다. 수동 확인 권장. (실제 OCR 결과값: 미검출)",
                        "정보 기재",
                        "미검출",
                    ))
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
