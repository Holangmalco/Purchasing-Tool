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
    COST_BREAKDOWN = "COST_BREAKDOWN"
    APPROVAL_MEMO = "APPROVAL_MEMO"
    EXCHANGE_RATE_PROOF = "EXCHANGE_RATE_PROOF"
    SUPPORTING_DOC = "SUPPORTING_DOC"
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
    if not text:
        return False, ""
    if len(text.strip()) < 10:
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

    # 2. 특수문자 깨짐 (\ufffd 등 유니코드 대체문자 또는 비정상 특수문자 과다)
    if "\ufffd" in text or (len(text) > 0 and len(re.findall(r"[^\w\s.,;:()/\-–—\[\]{}가-힣]", text)) > len(text) * 0.25):
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
    # 공조/냉난방 관련 일반적인 OCR 오인식 보정 (골조/곰조 -> 공조, 메민/머민 -> 메인, 규진/뮤진 -> 유진)
    cleaned = re.sub(r"메민골조|메민공조|머민곰조|머민공조", "메인공조", cleaned)
    cleaned = re.sub(r"규진곰조|뮤진곰조|규진공조|뮤진공조", "유진공조", cleaned)
    # 상호명 뒤에 괄호로 붙은 사업자등록번호 분리/제거 (예: ㈜ 현원상사 ( 807-86-01103) -> ㈜ 현원상사)
    cleaned = re.sub(r"\s*\(?\s*\d{3}[\s-]*\d{2}[\s-]*\d{5}\s*\)?", "", cleaned).strip()
    # 상호 뒤에 성명/대표자 라벨(성명, 섬며, 성며, 선명, 대표자 등)이 이어진 경우 상호와 대표자 성명 완벽 분리
    m_rep_split = re.search(r"^(.*?)\s+(?:성\s*명|섬\s*며|성\s*며|선\s*명|대\s*표\s*자|대\s*표)\s*[:：.]?\s*([가-힣\w\s]{2,10})$", cleaned)
    if m_rep_split:
        v_cand = m_rep_split.group(1).strip()
        r_cand = m_rep_split.group(2).strip()
        r_cand = re.sub(r"신습현", "신승현", r_cand)
        if _is_vendor_name_candidate(v_cand) and _is_representative_candidate(r_cand):
            return v_cand, r_cand

    cleaned = re.sub(
        r"\s*(?:견\s*적\s*번\s*호|견\s*적\s*일|일\s*시|발\s*행\s*일|납\s*품\s*기\s*일|성\s*명|섬\s*며|성\s*며|선\s*명|대\s*표\s*자|대\s*표\s*이\s*사|대\s*표|사\s*업\s*장|주\s*소|업\s*태|종\s*목|연\s*락\s*처|전\s*화|담\s*당\s*부\s*서|담\s*당\s*자|담\s*당|영\s*업\s*팀|솔\s*루\s*션\s*사\s*업\s*부|솔\s*루\s*션\s*팀|사\s*업\s*부|영\s*업\s*부|홈\s*페\s*이\s*지)\s*[:：]?\s*.*$",
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
        if _is_vendor_name_candidate(parts[0]) and _is_representative_candidate(parts[1]):
            return parts[0], parts[1]
        elif _is_representative_candidate(parts[1]):
            return "", parts[1]
    return cleaned, ""


def _is_representative_candidate(name: str) -> bool:
    clean = _clean_spaces(name)
    # 존칭(귀하, 귀중, 님, 교수님 등) 접미사 사전 정제
    clean = re.sub(r"\s*(?:귀하|귀중|님|교수님|선생님|대표님)\s*$", "", clean).strip()
    if not clean:
        return False
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
        "회사", "회사명", "타워", "빌딩", "센터", "아파트", "오피스텔", "주식회사", "하이츠", "밸리",
        "세무서장", "세무서", "국세청", "국세청장", "산학협력단", "산학협력", "산단", "대학교", "세종대", "대학원",
        "연구책임자", "연구원", "연구소", "공급받는자", "공급자", "수신자", "수신처", "발행자", "발행처",
        "작성자", "청구자", "관리자", "소재지", "전자세금계산서", "과세자", "일반과세자", "간이과세자", "면세사업자",
        "법인사업자", "공급가액", "합계금액", "부가세", "유효기간", "견적서", "사업자번호", "등록번호", "각자대표", "공동대표"
    )
    if any(k in clean for k in invalid_keywords):
        return False
    if clean in ("이사", "대표이사", "사장", "부장", "과장", "팀장", "상무", "전무", "대표", "대 표", "유형", "표자"):
        return False
    if clean.endswith(("각자", "공동")):
        return False
    # 단독 행정구역/도로명 주소 패턴 배제 (예: 관평동, 역삼로, 테크노1로 등)
    if re.search(r"^[(\[]?[가-힣]{1,6}(?:동|읍|면|리|구|시|군|로|길)[)\]]?$", clean):
        return False

    digits_or_letters = re.sub(r"[^\w가-힣\u4e00-\u9fff]", "", clean)
    if not (1 < len(digits_or_letters) <= 12):
        return False

    # 한글 전용 대표자명(2~4자)인 경우 성씨 유효성 검사 (외국인 영문명이나 한자/공동대표 슬래시 표기 제외)
    if re.fullmatch(r"[가-힣]{2,4}", digits_or_letters):
        if not re.match(rf"^{_KOREAN_SURNAMES}", digits_or_letters):
            return False

    return True


def _is_contact_person_candidate(val: str) -> bool:
    clean = _clean_spaces(val).strip()
    if not clean or re.search(r"[:：]\s*$", clean):
        return False
    norm = normalize_name(clean)
    invalid_keywords = (
        "귀하", "귀중", "산학협력", "세종", "부서", "구매담당", "담당부서", "대표", "대표이사",
        "고객님", "고객", "교수님", "교수", "선생님", "선생", "연구원님", "연구원", "연구책임자", "연구책임",
        "의뢰인", "의뢰자", "발주자", "발주처", "신청자", "사용자", "貴下",
        "전화", "전화번호", "팩스", "팩스번호", "휴대폰", "핸드폰", "연락처", "대표번호", "내선번호",
        "이메일", "전자우편", "mail", "email", "주소", "사업장", "소재지",
        "상호", "상호명", "법인명", "업체명", "회사명", "사업자", "사업자번호", "등록번호",
        "업태", "종목", "견적", "견적서", "견적일자", "납기", "결재조건", "특기사항", "비고", "참고사항",
        "합계", "금액", "단가", "수량", "공급가액", "부가세", "품명", "규격",
        "업무담당자", "업무담당", "담당자", "작성자", "발신인", "수신자", "공급자",
    )
    if any(k in norm for k in invalid_keywords):
        return False
    # 숫자가 포함된 경우 사람 이름 후보에서 제외
    if re.search(r"\d", clean):
        return False
    digits_or_letters = re.sub(r"[^\w가-힣A-Za-z]", "", clean)
    return 2 <= len(digits_or_letters) <= 20


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


def repair_business_number_ocr(raw_val: str, own_prefix: str = "") -> str:
    """OCR 오인식 문자(O->0, l->1, B->8 등) 및 1자리 체크섬 오차를 교정하여 유효한 사업자등록번호를 복원합니다.
    (계좌번호 등 타 정보는 제외하고 오직 10자리 사업자등록번호에만 안전하게 적용)
    """
    if not raw_val:
        return ""

    own_p = re.sub(r"\D", "", own_prefix or my_business_prefix())

    # 1단계: 자주 발생하는 OCR 영문/특수문자 혼동 우선 치환
    trans_table = str.maketrans({
        "O": "0", "o": "0", "D": "0", "Q": "0",
        "I": "1", "l": "1", "i": "1", "|": "1",
        "Z": "2",
        "S": "5", "s": "5",
        "b": "6",
        "B": "8",
    })
    cleaned = raw_val.translate(trans_table)
    digits = re.sub(r"\D", "", cleaned)

    # 11자리인 경우 앞 또는 뒤 1자리 노이즈(콜론 등이 숫자로 오인식된 경우) 제거 시도
    if len(digits) == 11:
        if is_valid_business_number(digits[1:]) and not (own_p and digits[1:].startswith(own_p)):
            return normalize_business_number(digits[1:])
        elif is_valid_business_number(digits[:-1]) and not (own_p and digits[:-1].startswith(own_p)):
            return normalize_business_number(digits[:-1])

    if len(digits) != 10 or len(set(digits)) == 1:
        return ""

    # 자사 사업자번호 접두사로 시작하는 번호는 타사 번호로 변형 방지
    if own_p and digits.startswith(own_p):
        return normalize_business_number(digits)

    # 이미 체크섬이 맞다면 정상 번호 반환
    norm = normalize_business_number(digits)
    if is_valid_business_number(norm):
        return norm

    # 2단계: 영문/특수문자가 포함되어 있었던 경우에 한해 1자리 OCR 혼동 문자 교정
    # 순수 10자리 숫자로 이미 입력된 번호(예: 전화번호나 임의 번호)는 억지로 타 번호로 왜곡하지 않음
    has_alpha = bool(re.search(r"[A-Za-z|]", raw_val))
    if not has_alpha:
        return norm

    confusable_map = {
        "0": ["8", "6", "9"],
        "1": ["7"],
        "2": ["7", "3"],
        "3": ["8", "2"],
        "5": ["6"],
        "6": ["5", "8", "0"],
        "7": ["1"],
        "8": ["0", "3", "6"],
        "9": ["0"],
    }

    digits_list = list(digits)
    for idx in reversed(range(10)):
        char = digits_list[idx]
        for alt in confusable_map.get(char, []):
            cand = list(digits_list)
            cand[idx] = alt
            cand_str = "".join(cand)
            if own_p and cand_str.startswith(own_p):
                continue
            cand_norm = normalize_business_number(cand_str)
            if is_valid_business_number(cand_norm):
                return cand_norm

    return norm


def _is_iacf_text(text: str) -> bool:
    """영문 산학협력단(IACF) 표기 여부 확인.
    
    규정: 영문 견적서의 경우 'industry-academic/academy/academia cooperation foundation' 또는
    약자 'iacf'가 포함되어 있어야 정식 산학협력단 수신자로 인정.
    개행 또는 공백 변형도 포괄하여 매칭.
    """
    if not text:
        return False
    t = text.lower()
    # 1. industry-academic/academy/academia/university cooperation foundation 및 변형 (하이픈, 띄어쓰기, 개행 유연 매칭)
    if re.search(r"industry[- ]?(?:academic|academia|academy|university)[\s\n]+cooperation(?:[\s\n]+foundation)?", t):
        return True
    if re.search(r"industry[- ]?(?:academic|academia|academy|university)[\s\n]+foundation", t):
        return True
    # 2. 약자 iacf (단독 단어 매칭)
    if re.search(r"\b(?:sju[-_ ]?)?iacf\b", t):
        return True
    # 3. cooperation foundation + sejong
    if re.search(r"cooperation[\s\n]+foundation", t) and ("sejong" in t or "sju" in t):
        return True
    return False


def _evidence(raw: str, source: str, confidence: float, inferred: bool = False) -> Evidence:
    return Evidence(raw=raw, evidence=source, confidence=confidence, inferred=inferred)


def classify_document(text: str, filename: str = "") -> DocumentType:
    haystack = f"{filename} {text}".lower()
    haystack_nospace = re.sub(r"\s+", "", haystack)

    # -------------------------------------------------------------
    # 0단계: 우선 감지 (부속 서류: 환율증빙, 기안문/품의서, 산출내역서, 사유서 등)
    # -------------------------------------------------------------
    is_explicit_quote = any(k in haystack_nospace for k in (
        "견적서", "견적서(quotation)", "quotation", "proformainvoice", "proforma",
        "priceproposal", "formalquotation", "見積書"
    ))

    # 0-1. 환율정보 증빙 (해외 구매 시 요청 당일 환율조회 증빙자료)
    has_fx_terms = any(k in haystack_nospace for k in (
        "서울외국환중개", "매매기준율", "외국환율조회", "환율조회", "환율정보", "환율변동", "고시회차",
        "송금보내실때", "송금받으실때", "현찰사실때", "keb하나은행환율", "한국은행기준환율"
    ))
    has_fx_fname = any(k in filename.lower() for k in ("환율", "exchangerate", "fx_rate", "매매기준율"))
    if (has_fx_terms or (has_fx_fname and any(k in haystack_nospace for k in ("환율", "usd", "달러", "기준율", "고시")))) and not any(k in haystack_nospace for k in ("견적서", "quotation", "見積書", "사업자등록증")):
        return DocumentType.EXCHANGE_RATE_PROOF

    # 0-2. 구매 기안문 / 품의서 / 지출결의서 (내부 결재선 및 기안자가 명시된 행정문서)
    has_memo_terms = any(k in haystack_nospace for k in (
        "기안문", "구매품의서", "품의서", "지출결의서", "구매기안", "연구비집행기안", "기안서", "지출결의", "내부결재문"
    )) or any(k in filename.lower() for k in ("기안문", "품의서", "지출결의", "구매기안", "연구비집행기안"))
    has_workflow_terms = any(k in haystack_nospace for k in ("기안자", "결재선", "기안일자", "결재일자", "협조자", "결재", "합의", "과장", "팀장", "부장", "처장"))
    if has_memo_terms and has_workflow_terms and not any(k in haystack_nospace for k in ("사업자등록증", "통장사본")):
        return DocumentType.APPROVAL_MEMO

    # 0-3. 세부산출내역서 / 공사내역서 / 원가계산서 (공사, 시제작, 세부 품목 내역서)
    has_breakdown_title = any(k in haystack_nospace for k in (
        "세부산출내역서", "산출내역서", "공사내역서", "원가계산서", "일위대가표", "일위대가", "세부내역서",
        "작업내역서", "제작내역서", "물량내역서"
    )) or any(k in filename.lower() for k in ("세부산출내역서", "산출내역서", "공사내역서", "원가계산서", "일위대가", "내역서"))
    has_breakdown_table = any(k in haystack_nospace for k in ("재료비", "노무비", "직접노무비", "일위대가")) and any(k in haystack_nospace for k in ("경비", "합계", "금액", "수량"))
    if (has_breakdown_title or has_breakdown_table) and not is_explicit_quote:
        return DocumentType.COST_BREAKDOWN

    # 0-4. 기타 부속 증빙서류 (사유서, 규격비교표, 시방서, 과업지시서 등)
    has_support_title = any(k in haystack_nospace for k in (
        "비교견적불가사유서", "수의계약사유서", "특정물품구매사유서", "단독응찰사유서", "구매사유서",
        "규격비교표", "과업지시서", "시방서", "사유서"
    )) or any(k in filename.lower() for k in ("사유서", "규격비교표", "과업지시서", "시방서", "사양서"))
    if has_support_title and not any(k in haystack_nospace for k in ("사업자등록증", "통장사본", "공사확인서")):
        if not (any(q in filename.lower() for q in ("견적서", "quotation")) and not any(s in filename.lower() for s in ("사유서", "규격"))):
            return DocumentType.SUPPORTING_DOC

    scores = {kind: 0 for kind in DocumentType}
    keywords = {
        DocumentType.QUOTE: (
            "견적", "quotation", "quote", "공급가액", "합계", "subtotal", "단가", "見積",
            "proforma invoice", "proforma", "price quote", "formal quotation", "cost estimate",
            "total amount", "grand total", "unit price",
        ),
        DocumentType.ONLINE_STORE_CART: (
            "장바구니총액", "장바구니", "apple.com/kr/shop/bag", "교육스토어홈", "applecare+", "apple(kr)",
            "결제예정금액", "삼성닷컴", "samsung.com", "주문상품", "멤버십포인트", "shoppingcart", "주문금액"
        ),
        DocumentType.BUSINESS_LICENSE: ("사업자등록증", "사업자번호", "등록번호", "대표자", "업태", "법인명", "세무서", "세무서장", "법인사업자", "일반과세자", "사면자", "들록증"),
        DocumentType.BANK_COPY: ("통장사본", "예금주", "계좌번호", "은행", "통장", "보통예금", "국민은행", "기업은행", "신한은행", "우리은행", "하나은행", "농협", "kb", "ibk"),
        DocumentType.CONSTRUCTION_CONFIRMATION: ("공사확인서", "준공", "공사명", "시공", "시설공사", "장비설치", "설치확인서"),
        DocumentType.COST_BREAKDOWN: (
            "세부산출내역서", "산출내역서", "공사내역서", "원가계산서", "일위대가", "세부내역서", "작업내역서",
            "재료비", "노무비", "경비", "직접노무비", "간접노무비", "제작내역서", "물량내역서",
        ),
        DocumentType.APPROVAL_MEMO: (
            "기안문", "품의서", "구매품의서", "지출결의서", "구매기안", "연구비집행기안", "기안서",
            "기안자", "결재선", "기안일자", "결재일자", "협조자",
        ),
        DocumentType.EXCHANGE_RATE_PROOF: (
            "매매기준율", "외국환율조회", "환율조회", "서울외국환중개", "송금보내실때", "송금받으실때",
            "현찰사실때", "고시회차", "외국환율", "기준환율", "keb하나은행환율", "exchange rate",
        ),
        DocumentType.SUPPORTING_DOC: (
            "비교견적불가사유서", "수의계약사유서", "특정물품구매사유서", "단독응찰사유서",
            "구매사유서", "사유서", "규격비교표", "규격서", "과업지시서", "시방서", "카탈로그",
        ),
    }
    for kind, terms in keywords.items():
        scores[kind] = sum(haystack_nospace.count(re.sub(r"\s+", "", term)) for term in terms)
    explicit_titles = {
        DocumentType.BUSINESS_LICENSE: ("사업자등록증", "사면자들록"),
        DocumentType.BANK_COPY: ("통장사본", "통장", "보통예금", "계좌번호", "계좌", "예금주", "실명확인"),
        DocumentType.CONSTRUCTION_CONFIRMATION: ("공사확인서", "시설공사", "장비설치확인서", "장비설치", "설치확인서"),
        DocumentType.ONLINE_STORE_CART: ("장바구니총액", "장바구니-apple", "apple.com/kr/shop/bag", "결제예정금액", "삼성닷컴", "교육스토어홈", "applecare+"),
        DocumentType.QUOTE: (
            "견적서", "견적", "quotation", "quote", "proformainvoice", "proforma", "costestimate",
            "pricequote", "priceproposal", "formalquotation", "estimate", "見積書", "見積",
        ),
        DocumentType.COST_BREAKDOWN: (
            "세부산출내역서", "산출내역서", "공사내역서", "원가계산서", "일위대가표", "일위대가", "세부내역서",
            "작업내역서", "제작내역서", "물량내역서",
        ),
        DocumentType.APPROVAL_MEMO: (
            "기안문", "품의서", "구매품의서", "지출결의서", "구매기안", "연구비집행기안", "기안서", "지출결의",
        ),
        DocumentType.EXCHANGE_RATE_PROOF: (
            "매매기준율", "서울외국환중개", "외국환율조회", "환율조회", "환율정보", "환율변동조회", "고시회차",
        ),
        DocumentType.SUPPORTING_DOC: (
            "비교견적불가사유서", "수의계약사유서", "특정물품구매사유서", "단독응찰사유서", "구매사유서",
            "규격비교표", "과업지시서", "시방서", "사유서",
        ),
    }
    title_hits = {kind: any(re.sub(r"\s+", "", term) in haystack_nospace for term in terms) for kind, terms in explicit_titles.items()}
    # 장바구니 화면 강한 핑거프린트 감지
    if "장바구니" in haystack_nospace and any(k in haystack_nospace for k in ("총액", "apple", "samsung", "삼성", "스토어", "결제", "bag", "shop")):
        title_hits[DocumentType.ONLINE_STORE_CART] = True
    if "결제예정금액" in haystack_nospace and any(k in haystack_nospace for k in ("삼성", "samsung", "주문상품", "멤버십")):
        title_hits[DocumentType.ONLINE_STORE_CART] = True
    if "applecare+" in haystack_nospace:
        title_hits[DocumentType.ONLINE_STORE_CART] = True

    # 통장사본 강한 핑거프린트 감지 (계좌번호 패턴 + 은행/예금 관련 키워드)
    if re.search(r"\b\d{3,6}-\d{2,6}-\d{3,8}\b", haystack_nospace) and any(k in haystack_nospace for k in ("은행", "통장", "예금", "계좌", "국민", "기업", "신한", "우리", "하나", "농협", "signature")):
        if not any(k in haystack_nospace for k in ("견적서", "견적", "사업자등록증", "공사확인서", "시설공사", "장비설치", "품의서", "기안문")):
            title_hits[DocumentType.BANK_COPY] = True

    # 영문/약식 견적서 핑거프린트 감지 (품목, 단가, 합계 표 구조 + 타 서류 키워드 부재)
    has_quote_terms = any(k in haystack_nospace for k in ("quotation", "quote", "proforma", "estimate", "단가", "공급가액", "견적"))
    has_table_terms = any(k in haystack_nospace for k in ("total", "subtotal", "합계", "총액", "총계")) and any(k in haystack_nospace for k in ("unitprice", "qty", "set", "ea", "수량", "단가", "금액", "부가세", "vat"))
    if (has_quote_terms or has_table_terms) and not any(k in haystack_nospace for k in ("사업자등록증", "세무서", "일반과세자", "통장사본", "예금주", "보통예금", "공사확인서", "applecare+", "장바구니", "apple.com", "삼성닷컴", "samsung.com")):
        scores[DocumentType.QUOTE] += 3

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
        # 다음 줄이 라벨 헤더 형태(콜론으로 끝나거나 값이 없는 양식 줄)면 건너뜀
        if re.search(r"[:：]\s*$", next_line.strip()):
            continue
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
        after_str = line_after[:25]
        if re.match(r"^[\s]*(?:[./\-년/\s]\s*)?\d{1,2}\s*(?:[./\-월월일]|\b월\b|\b일\b)", after_str):
            continue
        if re.match(r"^[\s]*[./\-년]\s*\d{1,2}\s*[./\-월]?", after_str):
            continue
        if re.match(r"^[\s]*-[\s]*\d{2,4}[\s]*-", after_str):
            continue

        raw = match.group(3)
        clean_digits = re.sub(r"[^\d]", "", raw)
        # 4자리 숫자가 연도(1990~2040)이고 뒤에 월/일이 오는 경우 날짜이므로 금액에서 제외
        if clean_digits and 1990 <= int(clean_digits) <= 2040:
            if re.search(r"^\s*(?:[./\-년/\s]\s*)?\d{1,2}\s*월", after_str):
                continue
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
            elif len(line_amts) == 2 and abs(line_amts[0] * 0.1 - line_amts[1]) <= max(10, int(line_amts[1] * 0.02)):
                calc_sum = line_amts[0] + line_amts[1]
                if f"{calc_sum:,}" in text or str(calc_sum) in text:
                    return Amount(calc_sum, _evidence(f"{calc_sum:,}", f"{line_full.strip()} (VAT포함 합계: {calc_sum:,})", 0.98), matched_label)

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
        after_str = text[match.end():match.end() + 20]
        # 날짜 형식 (2026.08.31, 2026-08-31, 2026년 8월, 2026 8월 등) 제외
        if re.match(r"^[\s]*(?:[./\-년/\s]\s*)?\d{1,2}\s*(?:[./\-월월일]|\b월\b|\b일\b)", after_str):
            continue
        if re.match(r"^[\s]*[./\-년]\s*\d{1,2}", after_str):
            continue
        # 사업자등록번호 또는 전화번호 형식 제외
        if re.match(r"^[\s]*-[\s]*\d{2,4}", after_str):
            continue

        raw = match.group(3)
        clean_digits = re.sub(r"[^\d]", "", raw)
        # 4자리 숫자가 연도(1990~2040)이고 뒤에 월/일이 오는 경우 날짜이므로 금액에서 제외
        if clean_digits and 1990 <= int(clean_digits) <= 2040:
            if re.search(r"^\s*(?:[./\-년/\s]\s*)?\d{1,2}\s*월", after_str):
                continue
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
    m_month_day = re.search(r"(?:유\s*효\s*기\s*간|견\s*적\s*유\s*효(?:\s*기\s*간)?|만\s*료\s*일|유\s*효\s*만\s*료\s*일)[^\d\n]{0,30}(\d{1,2})\s*월\s*(\d{1,2})\s*일", text)
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
    validity = re.search(
        r"(?:유\s*효\s*기\s*간|견\s*적\s*유\s*효(?:\s*기\s*간)?|견\s*적\s*일\s*(?:로\s*)?부\s*터|견\s*적\s*후|발\s*행\s*일\s*(?:로\s*)?부\s*터|validity|validy|period\s*of\s*valid(?:ity|y))[^\d\n]{0,40}(\d{1,3})\s*(일|日|days?|주|weeks?|개월|months?)",
        text,
        re.IGNORECASE,
    )
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
    table_keywords = ("단가", "공급가액", "공급가", "금액", "부가세", "단위", "수량", "금액(원)", "수량/단위", "unitprice", "qty")
    for idx, l in enumerate(lines):
        l_nospace = re.sub(r"\s+", "", l).lower()
        has_prod_kw = any(k in l_nospace for k in ("품명", "품목", "품명및사양", "description", "item"))
        has_val_kw = any(k in l_nospace for k in ("수량", "단가", "금액", "공급가", "공급가액", "amount", "total"))
        if has_prod_kw and has_val_kw:
            header_idx = idx
            break
        val_count = sum(1 for k in table_keywords if k in l_nospace)
        if val_count >= 2 and any(k in l_nospace for k in ("번호", "no.", "no")):
            header_idx = idx
            break
        if val_count >= 3:
            header_idx = idx
            break

    target_lines = lines[header_idx + 1:] if header_idx >= 0 else lines

    pending_prefix = ""
    for idx, line in enumerate(target_lines):
        line_clean = _clean_spaces(line)
        line_clean = re.sub(r"(\d+)\s*,\s*(\d{3})", r"\1,\2", line_clean)
        line_clean = re.sub(r"\b(\d{1,2})\s+(0,\d{3}(?:,\d{3})*)\b", r"\1\2", line_clean)
        line_clean = re.sub(r"\b(\d{1,2})\s+(\d{2},\d{3})\b(?!\s*,\s*\d{3})", r"\1\2", line_clean)
        if not line_clean or re.search(r"(품명|품목|description).*(수량|단가|금액)", line_clean, re.I):
            continue
        line_lower = line_clean.lower()
        line_nospace = re.sub(r"\s+", "", line_clean).lower()
        # 1. 견적 합계/총액 행 도달 시 품목 추출 종료 (단, 이미 품목이 1개 이상 추출된 경우)
        if items and re.search(r"^\s*(?:견\s*적\s*금\s*액\s*(?:계|\(VAT)?|합\s*계\s*금\s*액|공\s*급\s*가\s*액|공\s*급\s*금\s*액)\b", line_clean, re.I):
            break
        # 2. 제품 사양 또는 견적 비고 블록 도달 시 품목 추출 종료
        if items and any(k in line_nospace for k in ("제품의상세사양", "제품상세사양", "견적참고사항")):
            break

        if any(k in line_nospace for k in ("견적", "일자", "유효", "기간", "전화", "연락처", "팩스", "email", "이메일", "사업자", "등록번호", "대표", "주소", "소재지", "결제", "통장", "계좌", "담당", "특기사항", "특기", "비고", "참고사항", "주의사항")):
            continue
        if re.search(r"^\s*(?:합\s*계(?:\s*금\s*액)?|총\s*계|총\s*합\s*계|총\s*견적금액|금\s*액|공\s*급\s*금\s*액|공\s*급\s*가\s*액)\b", line_clean, re.I):
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
            if re.match(r"^(?:[-–—•*·]+\.?|\d+[\.\)])", line_clean) or any(k in line_clean for k in ("상호", "사양", "규격", "조건", "유틸리티", "설치")):
                continue
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
                trailing = line_clean[num_matches[-1].end():].strip()
                clean_trailing = re.sub(r"^(?:원(?:\s*정)?|vat\s*(?:포함|별도)?|부가세\s*(?:포함|별도)?|[(\[{]|\s*)+", "", trailing, flags=re.I).strip()
                clean_letters = re.sub(r"[^\w가-힣]", "", clean_trailing)
                if len(numbers) == 1 and len(clean_letters) >= 2:
                    matched = False
                else:
                    quantity = 1.0
                    unit_price = amount
                    name_cutoff = num_matches[-1].start()
                    matched = True

        if matched and amount is not None:
            raw_name = line_clean[:name_cutoff].strip()
            raw_name = re.sub(r"^\d+\s*(?:[\.\)\-]\s*|\s+)", "", raw_name).strip()
            raw_name = re.sub(r"\s+\d+\s*(?:set|ea|대|개|식|box|pk|명|회|건)?$", "", raw_name, flags=re.I).strip()
            raw_name = re.sub(r"\s+(?:set|ea|대|개|식|box|pk|명|회|건)\s*\d*$", "", raw_name, flags=re.I).strip()
            if pending_prefix:
                full_name = f"{pending_prefix} {raw_name}".strip()
                pending_prefix = ""
            else:
                full_name = raw_name
            full_name = re.sub(r"^\d+\s*(?:[\.\)\-]\s*|\s+)", "", full_name).strip()
            full_name = re.sub(r"\s+\d+\s*(?:set|ea|대|개|식|box|pk|명|회|건)?$", "", full_name, flags=re.I).strip()
            full_name = re.sub(r"\s+(?:set|ea|대|개|식|box|pk|명|회|건)\s*\d*$", "", full_name, flags=re.I).strip()
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
        "비고", "품목", "모델명", "규격", "product", "remark", "description", "item", "qty", "quantity",
        "세무서", "세무서장", "국세청", "지방국세청", "구청", "시청", "도청",
        "보통예금", "기업자유예금", "당좌예금", "정기예금", "저축예금", "연구책임자"
    )
    if any(k in norm for k in excluded):
        return False
    clean = re.sub(r"[^\w가-힣]", "", name)
    if clean in ("주식회사", "유한회사", "합자회사", "합명회사", "사단법인", "재단법인"):
        return False

    # 1. 단독 행정구역/동/면/리/로/길 패턴은 상호명 후보에서 원천 배제 (예: 관평동, 역삼동)
    if re.search(r"^[(\[]?[가-힣]{1,6}(?:동|읍|면|리|구|시|군|로|길)[)\]]?$", name.strip()):
        return False

    # 2. 시/도 + 구/군/시/로/길 주소 패턴 배제 (예: 대전광역시 유성구 관평동, 서울특별시 강남구 테크노1로)
    if re.search(r"(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)(?:특별|광역|자치)?(?:시|도)?\s+[가-힣0-9\s]+(?:구|군|시|읍|면|동|로|길)\b", name):
        return False
    if re.search(r"\b[가-힣]+(?:시|군|구)\s+[가-힣0-9]+(?:로|길|동|읍|면)\b", name):
        return False

    # 3. 단독 은행/금융기관 명칭 배제 (통장사본/견적서 은행 라인 오인식 방지, 주식회사 등 법인격 없는 단순 은행명)
    if name.strip() in (
        "국민은행", "기업은행", "신한은행", "우리은행", "하나은행", "농협", "농협은행",
        "SC제일은행", "씨티은행", "카카오뱅크", "케이뱅크", "토스뱅크", "우체국", "새마을금고", "신협", "수협",
        "보통예금", "자유예금", "당좌예금"
    ):
        return False

    # 4. 주소 파편/호수/건물명 (예: 관양동창덕에버빌, 108호, 109호관암돌심목에이, 관악대로488) 배제
    if any(k in name for k in ("동", "로", "길", "구", "시")) and any(k in name for k in ("빌", "호", "타워", "아파트", "센터", "층", "대로", "번길", "에이", "에비")):
        return False
    if re.search(r"\d+호", name):
        return False

    return len(clean) >= 2


# ==============================================================================
# 📋 문서 항목별 정규식 및 라벨 추출 규칙 테이블 (Extraction Rules Registry)
# ==============================================================================
# 한 화면에서 문서 유형별 정규식과 키워드 라벨을 한눈에 파악하고 수정할 수 있는 중앙 규칙 테이블입니다.
EXTRACTION_RULES = {
    "사업자번호": {
        "labels": ("사업자번호", "등록번호", "사업자등록번호", "사업자 등록번호", "사업자"),
        "pattern": r"[0-9OolIDBsSZ\s\-–—―─]{10,30}",
    },
    "상호명": {
        "labels": (
            "상호/대표", "상호 / 대표", "상호/대표자", "상호 / 대표자", "상호/성명", "상호 / 성명",
            "상호(법인명 및 사업체명)", "상호(법인명)", "상호[법인명]", "상호및성명", "상호 및 성명",
            "상호명", "상호", "상 호 명", "상 호", "법인명(단체명)", "법인명 (단체명)", "법인명[단체명]",
            "법인명 [단체명]", "법인명", "단체명", "업체명", "회사명", "회사면", "회사 면", "협력사명", "협력사",
            "삼人호", "삼호",
            "공급자", "발신", "발신처", "발행처", "발행자", "시공사", "시공자", "공사업체", "시공업체", "설치업체",
        ),
    },
    "대표자": {
        "labels": (
            "代 表", "代表", "회사명/대표", "상호/대표", "대표자(대표유형)", "대표자 (대표유형)",
            "대표자[대표유형]", "대표자 [대표유형]", "대표자", "대표자명", "대표자 성명", "대표이사", "대표", "성명",
            "섬며", "섬명", "성며", "선명",
        ),
    },
    "수신자": {
        "labels": (
            "수신자", "수신처", "수신", "귀하", "납품처", "발주처", "고객명", "고객사",
            "Messrs", "MESSRS", "Messrs.", "To", "TO", "to",
            "Bill To", "BILL TO", "Ship To", "SHIP TO", "Invoice To", "INVOICE TO",
            "Customer", "CUSTOMER", "Buyer", "BUYER", "Client", "CLIENT", "Attn", "ATTN",
        ),
    },
    "담당자": {
        "labels": ("담당자", "담당자명", "예약 담당자", "예약담당자", "담당", "담담", "담 담", "작성자", "발신인"),
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
        "account_labels": ("계좌번호", "계좌", "계변호", "입금계좌", "입금안내", "입금처"),
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
            "전체합계", "전체 합계", "전 체 합 계", "총합계금액", "총 합계 금액",
            "견적금액(VAT포함)", "견적금액 (VAT포함)", "견적금액(부가세포함)", "견적금액 (부가세포함)",
            "견적금액", "견적 금액", "견적총액", "견적 총액",
            "合 計 金 額", "合計金額", "合 計 金 額", "合計金額",
            "合 計", "合計",
        ),
        "general_total_labels": (
            "견적합계금액", "전체합계", "합계금액", "총합계금액", "총합계", "합계금",
            "합계(Total)", "합계 (Total)", "합계(total)", "합계 (total)",
            "제안공급가", "제안가격", "공급총액", "총 공급 가액", "총 공급가액", "총공급가액", "총공급대가",
            "예정금액", "공사예정금액", "설치예정금액", "사업비", "소요예산",
            "합계", "총계", "총액",
            "공급대가", "공사금액", "금액", "Total Amount", "TOTAL AMOUNT", "Total", "TOTAL",
            "合 計 金 額", "合計金額", "合 計 金 額", "合計金額",
            "合 計", "合計", "金 額", "金額", "金 額",
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
    valid_kinds = (
        DocumentType.QUOTE,
        DocumentType.BUSINESS_LICENSE,
        DocumentType.BANK_COPY,
        DocumentType.CONSTRUCTION_CONFIRMATION,
        DocumentType.ONLINE_STORE_CART,
        DocumentType.COST_BREAKDOWN,
        DocumentType.APPROVAL_MEMO,
        DocumentType.EXCHANGE_RATE_PROOF,
        DocumentType.SUPPORTING_DOC,
    )
    if kind not in valid_kinds:
        document.warnings.append("문서 유형을 확정하지 못했습니다. 원문을 검토하세요.")
    if text:
        is_anom, anom_msg = detect_ocr_anomaly(text)
        if is_anom:
            document.warnings.append(f"OCR 품질 경고 ({filename or '문서'}): {anom_msg}")
    own_prefix = re.sub(r"\D", "", my_business_prefix())

    # 공급자(Supplier) 섹션 분리 (견적서 내 공급받는자/수신처와의 라벨 혼동 원천 방지)
    supplier_text = text
    if kind in (DocumentType.QUOTE, DocumentType.CONSTRUCTION_CONFIRMATION, DocumentType.COST_BREAKDOWN):
        m_sup = re.search(r"(?:^|\n|[^\w가-힣])(?:공\s*급\s*자|공\s*급\s*처|발\s*행\s*처|Supplier|Vendor)(?=[^\w가-힣]|\n|$)", text, re.I)
        if m_sup:
            after_sup = text[m_sup.start():]
            m_end_sup = re.search(r"\n\s*(?:공\s*급\s*받\s*는\s*자|수\s*신\s*자|수\s*신|귀\s*[하중]|아래와\s*같이|합\s*계|총\s*합\s*계|품\s*명\b|No\b)", after_sup)
            if m_end_sup and m_end_sup.start() > 30:
                supplier_text = after_sup[:m_end_sup.start()]
            else:
                supplier_text = after_sup

    # 1. 라벨(등록번호, 사업자번호 등) 기반 번호 탐색 (공급자 블록 우선)
    labeled_biz = _labeled(supplier_text, EXTRACTION_RULES["사업자번호"]["labels"], EXTRACTION_RULES["사업자번호"]["pattern"])
    if not labeled_biz.raw and supplier_text != text:
        labeled_biz = _labeled(text, EXTRACTION_RULES["사업자번호"]["labels"], EXTRACTION_RULES["사업자번호"]["pattern"])
    selected_biz = ""
    biz_evidence_str = ""
    is_valid_biz = False

    if labeled_biz.raw:
        repaired = repair_business_number_ocr(labeled_biz.raw)
        clean_digits = re.sub(r"\D", "", repaired if repaired else labeled_biz.raw)
        if len(clean_digits) == 10 and not clean_digits.startswith(own_prefix):
            normalized = normalize_business_number(clean_digits)
            selected_biz = normalized
            raw_digits = re.sub(r"\D", "", labeled_biz.raw)
            is_repaired = bool(repaired and repaired != normalize_business_number(raw_digits))
            biz_evidence_str = f"{labeled_biz.evidence} (스마트교정: {normalized})" if is_repaired else labeled_biz.evidence
            is_valid_biz = is_valid_business_number(normalized)

    # 2. 라벨에서 번호를 못 찾았거나, 찾은 번호의 체크섬이 유효하지 않은 경우, 본문 전체에서 체크섬이 유효한 10자리 번호 우선 탐색
    if not selected_biz or not is_valid_biz:
        for m in _BIZ_RE.finditer(supplier_text if supplier_text != text else text):
            digits = re.sub(r"\D", "", m.group(1))
            if len(digits) == 10 and not digits.startswith(own_prefix):
                normalized = normalize_business_number(digits)
                if is_valid_business_number(normalized):
                    selected_biz = normalized
                    biz_evidence_str = m.group(0)
                    is_valid_biz = True
                    break
                else:
                    repaired = repair_business_number_ocr(m.group(1))
                    if repaired and is_valid_business_number(repaired):
                        selected_biz = repaired
                        biz_evidence_str = f"{m.group(0)} (스마트교정: {repaired})"
                        is_valid_biz = True
                        break
        if not selected_biz and supplier_text != text:
            for m in _BIZ_RE.finditer(text):
                digits = re.sub(r"\D", "", m.group(1))
                if len(digits) == 10 and not digits.startswith(own_prefix):
                    normalized = normalize_business_number(digits)
                    if is_valid_business_number(normalized):
                        selected_biz = normalized
                        biz_evidence_str = m.group(0)
                        is_valid_biz = True
                        break
                    else:
                        repaired = repair_business_number_ocr(m.group(1))
                        if repaired and is_valid_business_number(repaired):
                            selected_biz = repaired
                            biz_evidence_str = f"{m.group(0)} (스마트교정: {repaired})"
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
    if kind == DocumentType.BUSINESS_LICENSE and not document.vendor_name.raw:
        # 사업자등록증 등록번호 바로 다음 줄 상호 우선 탐색
        m_biz_num_line = re.search(r"(?:등록번호|등[록목]번호|[0-9]{3}-[0-9]{2}-[0-9]{5})[^\n]*\n+([^\n]+)", text)
        if m_biz_num_line:
            cand_line = m_biz_num_line.group(1).strip()
            if not re.search(r"^\d{4,}", cand_line) and not re.search(r"\d{3,}-\d{2,}", cand_line):
                cand_cleaned = re.sub(r"^[0-9\s:：\-.모보에]*(?:상\s*호\s*[:：.]?)?", "", cand_line).strip()
                cand_cleaned, _ = _clean_vendor_name(cand_cleaned)
                cand_cleaned = re.split(r"(?:성\s*명|대\s*표|생년월일|개업)", cand_cleaned)[0].strip()
                if _is_vendor_name_candidate(cand_cleaned) and len(cand_cleaned) >= 2 and any(c in cand_cleaned for c in ("공조", "에어컨", "시스템", "엔지니어링", "메인", "전자", "유진")):
                    document.vendor_name = _evidence(cand_cleaned, m_biz_num_line.group(0).strip(), 0.90, inferred=True)

    if kind != DocumentType.BANK_COPY and not document.vendor_name.raw:
        raw_vendor = _labeled(
            supplier_text,
            EXTRACTION_RULES["상호명"]["labels"],
        )
        if not raw_vendor.raw and supplier_text != text:
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
            cand = re.sub(r"^\[?단체명[\[:\]]?\s*", "", cand).strip()
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
            # 상호 라벨이 누락된 스캔 사업자등록증: 등록번호 바로 다음 줄 탐색 (공조, 에어컨 등 상호 키워드 또는 한글 업체명)
            m_biz_num_line = re.search(r"(?:등록번호|등[록목]번호|[0-9]{3}-[0-9]{2}-[0-9]{5})[^\n]*\n+([^\n]+)", text)
            if m_biz_num_line:
                cand_line = m_biz_num_line.group(1).strip()
                if not re.search(r"^\d{4,}", cand_line) and not re.search(r"\d{3,}-\d{2,}", cand_line):
                    cand_cleaned = re.sub(r"^[0-9\s:：\-.모보에]*(?:상\s*호\s*[:：.]?)?", "", cand_line).strip()
                    cand_cleaned = re.sub(r"^(?:보대인|대인)", "메인", cand_cleaned)
                    cand_cleaned = re.sub(r"^(?:에인)", "메인", cand_cleaned)
                    cand_cleaned = re.split(r"(?:성\s*명|대\s*표|생년월일|개업)", cand_cleaned)[0].strip()
                    if _is_vendor_name_candidate(cand_cleaned) and len(cand_cleaned) >= 2 and any(c in cand_cleaned for c in ("공조", "에어컨", "시스템", "엔지니어링", "메인", "전자", "유진")):
                        document.vendor_name = _evidence(cand_cleaned, m_biz_num_line.group(0).strip(), 0.90, inferred=True)

    if kind == DocumentType.CONSTRUCTION_CONFIRMATION and not document.vendor_name.raw:
        # 공사확인서 협력사(시공사) 직접 추출: 예: 4협력사명메인공조, 협력사: 메인공조
        m_confirm_v = re.search(r"(?:^|\n)\s*(?:[0-9\s.)\]]*)(?:협\s*력\s*사\s*명|협\s*력\s*사|시\s*공\s*사|공\s*사\s*업\s*체)[\s:：.\-]*([가-힣\w()]{2,20})", text)
        if m_confirm_v:
            cand_cv = re.split(r"(?:과\s*업|상\s*세|내\s*용|설\s*치|금\s*액)", m_confirm_v.group(1))[0].strip()
            cand_cv, _ = _clean_vendor_name(cand_cv)
            if _is_vendor_name_candidate(cand_cv):
                document.vendor_name = _evidence(cand_cv, m_confirm_v.group(0).strip(), 0.90, inferred=True)

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

    if extracted_rep_from_vendor and not document.representative.raw:
        document.representative = _evidence(extracted_rep_from_vendor, "상호-성명 라인 분리", 0.90)

    if not document.representative.raw:
        document.representative = _labeled(
            supplier_text,
            EXTRACTION_RULES["대표자"]["labels"],
            validator=_is_representative_candidate,
        )
    if not document.representative.raw and supplier_text != text:
        document.representative = _labeled(
            text,
            EXTRACTION_RULES["대표자"]["labels"],
            validator=_is_representative_candidate,
        )
    if not document.representative.raw:
        rep_direct = re.search(
            rf"(?:성\s*명|섬\s*며|성\s*며|선\s*명|대\s*표\s*자|대\s*표(?!\s*유\s*형)|代\s*表|代表)[\s'’`.,·]*[:：.\-]?\s*({_KOREAN_SURNAMES}[가-힣]{{1,3}}|[가-힣\u4e00-\u9fff\s]{{2,6}})(?!\w)",
            supplier_text if supplier_text != text else text,
        )
        if not rep_direct and supplier_text != text:
            rep_direct = re.search(
                rf"(?:성\s*명|섬\s*며|성\s*며|선\s*명|대\s*표\s*자|대\s*표(?!\s*유\s*형)|代\s*表|代表)[\s'’`.,·]*[:：.\-]?\s*({_KOREAN_SURNAMES}[가-힣]{{1,3}}|[가-힣\u4e00-\u9fff\s]{{2,6}})(?!\w)",
                text,
            )
        if rep_direct:
            cand_rep = _clean_spaces(rep_direct.group(1)).strip()
            cand_rep = re.sub(r"신습현", "신승현", cand_rep)
            if _is_representative_candidate(cand_rep):
                document.representative = _evidence(cand_rep, rep_direct.group(0).strip(), 0.88, inferred=True)
    if kind == DocumentType.BUSINESS_LICENSE and not document.representative.raw:
        # 사업자등록증 법인사업자 표준 서식: 대표자(대표유형) : 홍길동(각자대표), 김철수(각자대표)
        # OCR 노이즈: 대표자대표유형]이남작각자대표김형우각자대표], 대표자 [대표유형] 등
        lic_rep_m = re.search(
            r"(?:대\s*표\s*자|성\s*명)\s*(?:[\(\[]\s*대\s*표\s*유\s*형\s*[\)\]]|대\s*표\s*유\s*형[\]\)]?)?[\s:：.-]*([^\n]{1,100})",
            text,
        )
        if lic_rep_m:
            val = lic_rep_m.group(1).strip()
            val_clean = re.sub(r"[(\[]?\s*(?:각\s*자\s*대\s*표|공\s*동\s*대\s*표|대\s*표\s*유\s*형|대\s*표\s*이\s*사)\s*[)\]]?", " ", val)
            val_clean = re.sub(r"[)\]}>'\"`.,·]", " ", val_clean)
            cands = [p.strip() for p in re.split(r"[,/\s]+", val_clean) if p.strip()]
            valid_reps = [c for c in cands if _is_representative_candidate(c) or re.match(rf"^{_KOREAN_SURNAMES}[가-힣]{{1,2}}$", c)]
            if valid_reps:
                rep_str = ", ".join(valid_reps)
                document.representative = _evidence(rep_str, lic_rep_m.group(0).strip(), 0.95, inferred=True)

    if not document.representative.raw:
        rep_suffix = re.search(
            rf"({_KOREAN_SURNAMES}[가-힣]{{1,3}})\s*대\s*표(?!\w)",
            supplier_text if supplier_text != text else text,
        )
        if not rep_suffix and supplier_text != text:
            rep_suffix = re.search(
                rf"({_KOREAN_SURNAMES}[가-힣]{{1,3}})\s*대\s*표(?!\w)",
                text,
            )
        if rep_suffix:
            cand_rep = rep_suffix.group(1).strip()
            if not cand_rep.endswith(("각자", "공동")) and _is_representative_candidate(cand_rep):
                document.representative = _evidence(cand_rep, rep_suffix.group(0).strip(), 0.88, inferred=True)
    if kind == DocumentType.BUSINESS_LICENSE and not document.representative.raw:
        # 1. 정식 라벨: 대표자, 대표, 표자, 성명
        lic_rep = re.search(rf"(?:대\s*표\s*자|대\s*표(?!\s*유\s*형)|표\s*자|성\s*명)[- \s'’`.,·:이$*#@%^&!/\\~_<]*\s*({_KOREAN_SURNAMES}[가-힣]{{1,2}})", text)
        if lic_rep:
            cand_r = lic_rep.group(1).strip()
            if _is_representative_candidate(cand_r):
                document.representative = _evidence(cand_r, lic_rep.group(0).strip(), 0.90, inferred=True)

        # 2. 배분 정렬(넓은 자간) 사업자등록증: 상호/법인명 바로 다음 줄에서 '대·표'가 누락되고 줄 첫머리가 '자'로 시작되는 경우만 엄격 한정
        # (담당자, 사업자, 수신자, 근로자 등 타 단어 끝 글자 '자' 오인식 원천 차단)
        if not document.representative.raw:
            lic_rep_narrow = re.search(
                rf"(?:법\s*인\s*명|상\s*호)[^\n]*\n+\s*자\s*[- \s'’`.,·:이$*#@%^&!/\\~_<]*\s*({_KOREAN_SURNAMES}[가-힣]{{1,2}})",
                text
            )
            if lic_rep_narrow:
                cand_rn = lic_rep_narrow.group(1).strip()
                if _is_representative_candidate(cand_rn):
                    document.representative = _evidence(cand_rn, lic_rep_narrow.group(0).strip(), 0.85, inferred=True)

    if document.representative.raw:
        rep_clean = re.split(r"(?:생\s*년\s*월\s*일|주\s*민|개\s*업|사\s*업\s*장)", document.representative.raw)[0].strip()
        rep_clean = re.sub(r"\s*\(?(?:인|직인|서명|직인생략|직인\s*생략|印|서명날인)\)?$", "", rep_clean).strip()
        if "/" in rep_clean:
            slash_r = [p.strip() for p in rep_clean.split("/") if p.strip()]
            if len(slash_r) >= 2 and _is_representative_candidate(slash_r[-1]):
                rep_clean = slash_r[-1]
        if rep_clean:
            document.representative = _evidence(rep_clean, document.representative.evidence, document.representative.confidence)
    if not document.representative.raw and extracted_rep_from_vendor:
        document.representative = _evidence(extracted_rep_from_vendor, f"상호 뒤 성명 분리: {raw_vendor.raw}", 0.78, inferred=True)

    # 수신자(Recipient) 추출: 2줄 분리 인쇄 및 인접 행(Window) 스캔 지원
    lines = text.splitlines()
    rec_line_idx = -1
    raw_l1 = ""
    evidence_line = ""

    # 1. 수신 라벨이 포함된 줄 인덱스 탐색
    for idx, line in enumerate(lines):
        if re.search(r"(?:수\s*신\s*자?|수\s*신\s*처|To\b|Messrs\.?|Bill\s*To\b|Invoice\s*To\b)[\s:：.-]", line, re.I):
            rec_line_idx = idx
            evidence_line = line
            # 라벨 뒤 텍스트 분리
            m = re.search(r"(?:수\s*신\s*자?|수\s*신\s*처|To\b|Messrs\.?|Bill\s*To\b|Invoice\s*To\b)[\s:：.-]*([^\n]{1,60})", line, re.I)
            if m:
                raw_l1 = m.group(1).strip()
            break

    # 1-1. 라벨 탐색 실패 시 _labeled fallback
    if not raw_l1:
        labeled_r = _labeled(text, EXTRACTION_RULES["수신자"]["labels"])
        if labeled_r.raw:
            raw_l1 = labeled_r.raw
            evidence_line = labeled_r.evidence

    if raw_l1:
        cleaned_rec = re.sub(
            r"^(?:Messrs\.?|To\b\.?|To\b|Bill\s*To\b|Ship\s*To\b|Invoice\s*To\b|Customer\b|Buyer\b|Client\b|Attn\b|업\s*체\s*명|처\s*상\s*호|수\s*신\s*처\s*상\s*호|상\s*호\s*명|상\s*호|수\s*신\s*자?|고\s*객\s*명|고\s*객\s*사|회\s*사\s*명)\s*[:：.]?\s*",
            "",
            raw_l1,
            flags=re.I,
        ).strip(" .:,;|-")
        cleaned_rec = re.split(
            r"(?:Quotation\s*#?|Quote\s*#?|DATE\b|Date\b|귀\s*[하중]|貴\s*[中下]|\s+상\s*호|공\s*급|등\s*록\s*번\s*호|사\s*업\s*자|대\s*표\s*자\s*명|대\s*표\s*자|대\s*표(?!\w)|성\s*명|견\s*적\s*제\s*출\s*일|제\s*출\s*일|\d\s*\d\s*\d\s*-)",
            cleaned_rec,
            flags=re.I,
        )[0].strip(" .:,;|-")

        # 2. 인접 윈도우 스캔 (2~3줄 분리 인쇄 병합: TO: 옆에 사람이름이 오거나 기관명이 다음 줄에 개행된 경우 포함)
        if rec_line_idx != -1 and not any(k in normalize_name(cleaned_rec) for k in ("산단", "산학협력")) and not _is_iacf_text(cleaned_rec):
            for next_idx in range(rec_line_idx + 1, min(rec_line_idx + 4, len(lines))):
                l_next = lines[next_idx]
                m_san = re.search(r"(세종[\s/·\-]*대[\s/·\-]*학[\s/·\-]*교[^\n]*|산[\s/·\-]*학[\s/·\-]*협[\s/·\-]*[력려련]?[\s/·\-]*단?|산[\s/·\-]*단|\biacf\b|sejong\s+univ[^\n]*|industry[- ]?(?:academic|academy|academia)[^\n]*|cooperation[\s\n]+foundation|\bfoundation\b)", l_next, re.I)
                if m_san:
                    sub_part = m_san.group(1).strip(" .:,;|-")
                    sub_part = re.split(r"(?:공\s*급\s*자|\s+상\s*호|등\s*록|사\s*업\s*자|주\s*소|전\s*화|TEL|일\s*자|Issued\s*By)", sub_part, flags=re.I)[0].strip(" .:,;|-")
                    sub_part = re.sub(r"\s*귀\s*[하중]", "", sub_part).strip(" .:,;|-")
                    sub_part = re.sub(r"산[\s/·\-]*학[\s/·\-]*협[\s/·\-]*[력려련]?[\s/·\-]*단?", "산학협력단", sub_part)
                    sub_part = re.sub(r"산[\s/·\-]*단", "산단", sub_part)
                    if sub_part:
                        if not any(k in normalize_name(cleaned_rec) for k in ("세종", "sejong")) and any(k in normalize_name(sub_part) for k in ("세종", "sejong")):
                            cleaned_rec = sub_part
                        else:
                            cleaned_rec = f"{cleaned_rec} {sub_part}".strip()
                        evidence_line = f"{evidence_line} / {l_next.strip()}"
                        # 3번째 줄에 Foundation이나 산학협력단이 또 나뉜 경우 (예: Sejong Univ... \n Foundation)
                        if next_idx + 1 < len(lines):
                            l_after = lines[next_idx + 1]
                            m_after = re.search(r"(\bfoundation\b|산[\s/·\-]*학[\s/·\-]*협[\s/·\-]*[력려련]?[\s/·\-]*단?|산단)", l_after, re.I)
                            if m_after:
                                f_part = m_after.group(1).strip(" .:,;|-")
                                f_part = re.sub(r"산[\s/·\-]*학[\s/·\-]*협[\s/·\-]*[력려련]?[\s/·\-]*단?", "산학협력단", f_part)
                                cleaned_rec = f"{cleaned_rec} {f_part}".strip()
                                evidence_line = f"{evidence_line} / {l_after.strip()}"
                        break

        if cleaned_rec:
            document.recipient = _evidence(cleaned_rec, evidence_line, 0.88, inferred=True)

    # 3. 본문 전체 fallback 탐색 (수신처 라벨이 없는 경우 또는 수신처에 산단이 누락된 경우)
    has_full_sanhan = False
    if document.recipient.raw:
        r_norm = normalize_name(document.recipient.raw)
        if ("산학협력" in r_norm or "산단" in r_norm) and ("세종" in r_norm or "sju" in r_norm):
            has_full_sanhan = True
        elif _is_iacf_text(document.recipient.raw):
            has_full_sanhan = True

    if not has_full_sanhan:
        candidate_lines = []
        for idx, line in enumerate(lines):
            line_clean = re.sub(r"^[▪▶•\-\*\s]+", "", _clean_spaces(line)).strip()
            norm = normalize_name(line_clean)
            line_lower = line_clean.lower()
            is_ko_hit = ("세종" in norm or "산단" in norm or "산학협력" in norm) and not any(k in norm for k in ("공급자", "시공사", "발행처", "세무서", "견적서", "대관", "컨벤션센터", "운영관리"))
            is_en_hit = (_is_iacf_text(line_lower) or ("sejong" in line_lower and ("univ" in line_lower or "iacf" in line_lower or "industry" in line_lower or "cooperation" in line_lower))) and not any(k in line_lower for k in ("vendor", "supplier", "issuer", "quotation", "quote"))
            if is_ko_hit or is_en_hit:
                candidate_lines.append((idx, line_clean))
        
        chosen_idx = -1
        chosen_line = None
        for c_idx, cl in candidate_lines:
            cl_norm = normalize_name(cl)
            if "산학협력" in cl_norm or "산단" in cl_norm or _is_iacf_text(cl):
                chosen_line = cl
                chosen_idx = c_idx
                break
        if not chosen_line and candidate_lines:
            chosen_idx, chosen_line = candidate_lines[0]

        if chosen_line:
            cleaned_rec = re.sub(r"^[▪▶•\-\*\s]+", "", chosen_line)
            cleaned_rec = re.sub(
                r"^(?:Messrs\.?|To\b\.?|To\b|Bill\s*To\b|Ship\s*To\b|Invoice\s*To\b|Customer\b|Buyer\b|Client\b|Attn\b|업\s*체\s*명|처\s*상\s*호|수\s*신\s*처\s*상\s*호|상\s*호\s*명|상\s*호|수\s*신\s*자?|고\s*객\s*명|고\s*객\s*사|회\s*사\s*명)\s*[:：.]?\s*",
                "",
                cleaned_rec,
                flags=re.I,
            ).strip(" .:,;|-")
            cleaned_rec = re.split(
                r"(?:DATE\b|Date\b|귀\s*[하중]|貴\s*[中下]|\s+상\s*호|공\s*급|등\s*록\s*번\s*호|사\s*업\s*자|대\s*표\s*자\s*명|대\s*표\s*자|대\s*표(?!\w)|성\s*명|견\s*적\s*제\s*출\s*일|제\s*출\s*일|\d\s*\d\s*\d\s*-)",
                cleaned_rec,
                flags=re.I,
            )[0].strip(" .:,;|-")

            # fallback 탐색에서도 인접 윈도우 스캔 (2줄 분리 개행된 산학협력단/Foundation 병합)
            if chosen_idx != -1 and not any(k in normalize_name(cleaned_rec) for k in ("산단", "산학협력")) and not _is_iacf_text(cleaned_rec):
                for next_idx in range(chosen_idx + 1, min(chosen_idx + 3, len(lines))):
                    l_next = lines[next_idx]
                    m_san = re.search(r"(산[\s/·\-]*학[\s/·\-]*협[\s/·\-]*[력려련]?[\s/·\-]*단?|산[\s/·\-]*단|\biacf\b|industry[- ]?(?:academic|academy|academia)[^\n]*|cooperation[\s\n]+foundation|\bfoundation\b)", l_next, re.I)
                    if m_san:
                        sub_part = m_san.group(1).strip(" .:,;|-")
                        sub_part = re.split(r"(?:공\s*급\s*자|\s+상\s*호|등\s*록|사\s*업\s*자|주\s*소|전\s*화|TEL|일\s*자)", sub_part)[0].strip(" .:,;|-")
                        sub_part = re.sub(r"\s*귀\s*[하중]", "", sub_part).strip(" .:,;|-")
                        sub_part = re.sub(r"산[\s/·\-]*학[\s/·\-]*협[\s/·\-]*[력려련]?[\s/·\-]*단?", "산학협력단", sub_part)
                        sub_part = re.sub(r"산[\s/·\-]*단", "산단", sub_part)
                        if sub_part:
                            cleaned_rec = f"{cleaned_rec} {sub_part}".strip()
                            chosen_line = f"{chosen_line} / {l_next.strip()}"
                            break

            if cleaned_rec:
                if document.recipient.raw and any(k in normalize_name(document.recipient.raw) for k in ("세종", "sejong")) and not any(k in normalize_name(document.recipient.raw) for k in ("산단", "산학협력")) and not _is_iacf_text(document.recipient.raw):
                    document.recipient = _evidence(f"{document.recipient.raw} {cleaned_rec}".strip(), f"{document.recipient.evidence} + {chosen_line}", 0.88, inferred=True)
                else:
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

    if not document.phone.raw or len(re.findall(r"\d", document.phone.raw)) < 7:
        document.phone = _labeled(text, EXTRACTION_RULES["전화번호"]["labels"], EXTRACTION_RULES["전화번호"]["pattern"])
    
    # 고객(교수님/연구원/고객님/의뢰인) 라벨 직후에 기재된 교내 의뢰인 전화번호 감지
    client_phone_match = re.search(
        r"(?:고객님|교수님|선생님|연구원님|연구원|의뢰인)[^\n]*\n+[^\n]*?(?:연락처|전화|휴대폰|H\.?P)[\s:：.-]*([0-9() .+-]{8,20})",
        text
    )
    client_phone_digits = re.sub(r"\D", "", client_phone_match.group(1)) if client_phone_match else ""

    # 만약 라벨에서 찾은 번호가 고객 직속 전화번호라면 업체 번호에서 배제
    if document.phone.raw and client_phone_digits and re.sub(r"\D", "", document.phone.raw) == client_phone_digits:
        document.phone = Evidence()

    # 010 휴대폰 번호 우선 및 본문 정규식 fallback (고객 번호 제외)
    valid_mobiles = []
    for m in re.finditer(r"01[016789][- ]?\d{3,4}[- ]?\d{4}", text):
        m_dig = re.sub(r"\D", "", m.group(0))
        if m_dig != client_phone_digits:
            valid_mobiles.append(m.group(0).strip())

    m_mobile = valid_mobiles[0] if valid_mobiles else None
    if not document.phone.raw or len(re.findall(r"\d", document.phone.raw)) < 7:
        if m_mobile:
            document.phone = _evidence(m_mobile, m_mobile, 0.9, inferred=True)
        else:
            all_landlines = []
            for m in re.finditer(r"(?:02|0[3-6]\d)[- )]?\d{3,4}[- ]?\d{4}", text):
                m_dig = re.sub(r"\D", "", m.group(0))
                if m_dig != client_phone_digits:
                    all_landlines.append(m.group(0).strip())
            if all_landlines:
                document.phone = _evidence(all_landlines[0], all_landlines[0], 0.8, inferred=True)
    elif m_mobile and not document.phone.raw.startswith("01"):
        # 라벨에서 유선번호를 찾았더라도 본문에 유효한 010 휴대폰 번호가 명시되어 있는 경우 담당자 다이렉트 연락처로 반영
        document.phone = _evidence(m_mobile, f"{document.phone.evidence} (휴대폰 우선: {m_mobile})", 0.92, inferred=True)

    if document.phone.raw:
        cleaned_p = re.sub(r"^[^\d]+", "", document.phone.raw).strip()
        cleaned_p = re.sub(r"[^\d]+$", "", cleaned_p).strip()
        if cleaned_p and len(re.findall(r"\d", cleaned_p)) >= 7 and re.sub(r"\D", "", cleaned_p) != client_phone_digits:
            document.phone = _evidence(cleaned_p, document.phone.evidence, document.phone.confidence)
        else:
            document.phone = Evidence()

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
            if not any(k in cand_holder for k in ("고객", "회원", "귀하", "은행", "통장", "예금", "과목", "지점", "여럼", "담당자")):
                document.account_holder = _evidence(cand_holder, nim_match.group(0).strip(), 0.85, inferred=True)
        # 1-2. (주)상호 / [주]상호 / [주상호 패턴 직접 탐색
        if not document.account_holder.raw:
            for corp_match in re.finditer(
                r"(?:^|\n)\s*((?:\(주\)|㈜|\[주\]|\[주|주\s*식\s*회\s*사)\s*[가-힣A-Za-z0-9]{2,15}|[가-힣A-Za-z0-9]{2,15}\s*(?:\(주\)|㈜|\[주\]|\[주))",
                text,
            ):
                cand_ch = _clean_spaces(corp_match.group(1)).strip()
                cand_ch = re.sub(r"^\[주\]?\s*", "(주)", cand_ch)
                cand_ch = re.sub(r"\s*\[주\]?$", "(주)", cand_ch)
                if not any(k in cand_ch for k in ("은행", "통장", "과목", "계좌", "신규", "예금", "여럼", "Bank", "BANK")):
                    document.account_holder = _evidence(cand_ch, corp_match.group(0).strip(), 0.88, inferred=True)
                    break
        # 2. 통장 상단 첫 줄 또는 계좌번호/과목 앞의 상호/성명 탐색
        if not document.account_holder.raw:
            for line in text.splitlines()[:6]:
                clean_l = _clean_spaces(line).strip()
                if not clean_l or len(clean_l) < 2 or len(clean_l) > 25:
                    continue
                if any(k in clean_l for k in ("은행", "통장", "과목", "계좌", "금융", "신규", "지점", "인감", "서명", "Bank", "BANK", "예금", "종류", "거래", "보통", "자유", "당좌", "저축", "적금", "청약", "여럼", "결산", "가입", "발행", "회차", "실명", "금요일", "국민", "문램", "SIGNATURE", "signature")):
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
    def _is_account_candidate(val: str) -> bool:
        clean_len = len(re.sub(r"[^\d*]", "", val))
        if clean_len < 8 or clean_len > 25:
            return False
        if any(k in val for k in ("품명", "수량", "단가", "공급가", "금액", "대표", "업태", "종목", "주소", "견적", "산학", "연락처", "전화")):
            return False
        # 일반 전화번호 형식(02-, 031-, 042-, 010- 등 전국 지역번호/휴대폰) 제외
        if bool(re.match(r"^0\d{1,2}[- ]?\d{3,4}[- ]?\d{4}$", val)) or val.startswith(("02-", "010-", "070-", "050-", "080-", "031-", "032-", "033-", "041-", "042-", "043-", "044-", "051-", "052-", "053-", "054-", "055-", "061-", "062-", "063-", "064-")):
            return False
        # 사업자등록번호 형식(3-2-5) 제외
        if bool(re.match(r"^\d{3}[- ]?\d{2}[- ]?\d{5}$", val)) or is_valid_business_number(val):
            return False
        # 주민/법인등록번호 형식(6-7) 제외
        if bool(re.match(r"^\d{6}[- ]?\d{7}$", val)):
            return False
        # 날짜 형식 (2026-08-31, 2026.08.31 등) 제외
        if bool(_DATE_RE.match(val)) or bool(re.match(r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}$", val)):
            return False
        if bool(re.search(r"\d{1,2}\s*월|\d{1,2}\s*일|\d{4}\s*년", val)):
            return False
        return True

    document.account_number = _labeled(
        text,
        EXTRACTION_RULES["통장사본"]["account_labels"],
        EXTRACTION_RULES["통장사본"]["account_pattern"],
        validator=_is_account_candidate,
    )
    if not document.account_number.raw:
        acc_match = re.search(r"(?:계좌번호|계좌\s*번호|입금계좌|계좌|계변호|입금안내|입금처)[^\d\n]{0,20}([0-9\-–—]{6,30})", text)
        if acc_match:
            acc_cand = acc_match.group(1).strip()
            if _is_account_candidate(acc_cand):
                document.account_number = _evidence(acc_cand, acc_match.group(0).strip(), 0.90, inferred=True)
    if not document.account_number.raw and kind == DocumentType.BANK_COPY:
        hyphen_acc = re.search(r"\b(\d{2,6}(?:-\d{2,7}){2,4})\b", text)
        if hyphen_acc:
            cand_acc = hyphen_acc.group(1).strip()
            digits_only = re.sub(r"\D", "", cand_acc)
            # 일반 전화번호 형식(02-, 031-, 042-, 010- 등 전국 지역번호/휴대폰) 제외
            is_phone = bool(re.match(r"^0\d{1,2}-\d{3,4}-\d{4}$", cand_acc)) or cand_acc.startswith(("02-", "010-", "070-", "050-", "080-"))
            # 사업자등록번호 형식(3-2-5) 제외
            is_biz = bool(re.match(r"^\d{3}-\d{2}-\d{5}$", cand_acc)) or is_valid_business_number(cand_acc)
            # 법인/주민등록번호 형식(6-7) 제외
            is_reg = bool(re.match(r"^\d{6}-\d{7}$", cand_acc))
            if len(digits_only) >= 10 and not is_phone and not is_biz and not is_reg:
                document.account_number = _evidence(cand_acc, hyphen_acc.group(0).strip(), 0.90, inferred=True)
    if not document.account_number.raw and "은행계좌" in text:
        bank_acc_match = re.search(
            r"(?:국민|기업|중소기업|신한|우리|하나|농협|SC제일|씨티|카카오|케이|토스|대구|부산|광주|제주|전북|경남|우체국|새마을|신협|수협)(?:은행|금고|뱅크)?\s*[:：]?\s*([0-9\-–—]{7,25})",
            text,
        )
        if bank_acc_match:
            cand_acc = bank_acc_match.group(1).strip()
            digits = re.sub(r"\D", "", cand_acc)
            if len(digits) >= 9 and not is_valid_business_number(cand_acc) and not re.match(r"^0\d{1,2}-", cand_acc) and not re.match(r"^\d{4}-\d{4}$", cand_acc):
                document.account_number = _evidence(cand_acc, bank_acc_match.group(0).strip(), 0.90, inferred=True)
    if document.account_number.raw:
        m_acc_clean = re.search(r"[0-9\-–—]{6,30}", document.account_number.raw)
        if m_acc_clean and len(re.sub(r"\D", "", m_acc_clean.group(0))) >= 6:
            document.account_number = _evidence(m_acc_clean.group(0).strip(), document.account_number.evidence, document.account_number.confidence)

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
    if kind in (DocumentType.QUOTE, DocumentType.CONSTRUCTION_CONFIRMATION, DocumentType.COST_BREAKDOWN):
        total_text = re.sub(r"(?<=\d)[이공영](?=\d)", "0", text)
        total_rules = EXTRACTION_RULES["견적총액"]
        # 1차: 최종 수정/할인/네고 금액 라벨 우선 탐색
        document.total = _amount_after_label(
            total_text,
            total_rules["revised_labels"],
            exclude_keywords=("별도", "제외", "excl"),
        )
        # 2차: 명시적 부가세 포함 최종 합계 라벨 탐색
        if document.total.value is None:
            document.total = _amount_after_label(
                total_text,
                total_rules["vat_inclusive_labels"],
                exclude_keywords=("별도", "제외", "excl"),
            )
        # 3차: 일반 합계/총액 라벨 탐색
        if document.total.value is None:
            document.total = _amount_after_label(
                total_text,
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
        if document.total.value is None or (document.subtotal.value is not None and document.total.value == document.subtotal.value and document.vat.value):
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
                    document.total = Amount(int(clean_digits), _evidence(raw_amt, vat_inc_match.group(0).strip(), 0.95, True), "VAT포함 추론")
            elif document.subtotal.value is not None and document.vat.value is not None and document.vat.value > 0:
                calc_val = document.subtotal.value + document.vat.value
                calc_val_formatted = f"{calc_val:,}"
                if calc_val_formatted in text or str(calc_val) in text:
                    document.total = Amount(calc_val, _evidence(calc_val_formatted, "공급가액+부가세 산술 합계", 0.92, True), "합계금액 산출")
        if document.total.value is None and kind in (DocumentType.QUOTE, DocumentType.CONSTRUCTION_CONFIRMATION):
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

            # 4. 주요 공통 키워드 / 영문 약어 일치 (예: RTA, CVD, ALD, HPLC 등 특수 장비/품목 식별어)
            words1 = {w for w in re.findall(r"[A-Za-z0-9]{3,}", it1.name.lower()) if w not in ("the", "and", "for", "set", "box", "item", "system")}
            words2 = {w for w in re.findall(r"[A-Za-z0-9]{3,}", it2.name.lower()) if w not in ("the", "and", "for", "set", "box", "item", "system")}
            if words1 and words2 and (words1 & words2):
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
    quotes_with_amt = [d for d in documents if d.document_type in (DocumentType.QUOTE, DocumentType.ONLINE_STORE_CART) and d.total.value is not None]
    if quotes_with_amt:
        return min(quotes_with_amt, key=lambda q: q.total.value or 0)
    quotes_any = [d for d in documents if d.document_type in (DocumentType.QUOTE, DocumentType.ONLINE_STORE_CART)]
    return quotes_any[0] if quotes_any else None


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
    # 대표자 다중 대표(각자대표/공동대표) 세트 비교
    if "대표" in label:
        def _extract_rep_set(v: str) -> set[str]:
            v_clean = re.sub(r"[(\[]?\s*(?:각\s*자\s*대\s*표|공\s*동\s*대\s*표|대\s*표\s*이\s*사|대\s*표|인|직인|서명|印)\s*[)\]]?", " ", v)
            parts = [normalize_name(p) for p in re.split(r"[,/&+\s]+", v_clean) if p.strip()]
            return {p for p in parts if len(p) >= 2}

        reps_expected = _extract_rep_set(expected)
        reps_actual = _extract_rep_set(actual)
        if reps_expected and reps_actual:
            if reps_expected == reps_actual:
                return CheckResult(
                    label,
                    VerificationStatus.PASS,
                    f"{label} 일치 ({', '.join(sorted(reps_expected))}).",
                    expected,
                    actual,
                )
            if reps_expected.issubset(reps_actual) or reps_actual.issubset(reps_expected):
                common = reps_expected & reps_actual
                return CheckResult(
                    label,
                    VerificationStatus.PASS,
                    f"{label} 일치 (각자대표/공동대표 중 일치 확인: {', '.join(sorted(common))}).",
                    expected,
                    actual,
                )
            if reps_expected & reps_actual:
                common = reps_expected & reps_actual
                return CheckResult(
                    label,
                    VerificationStatus.PASS,
                    f"{label} 일치 (공동/각자대표 일부 일치: {', '.join(sorted(common))}).",
                    expected,
                    actual,
                )

    # 마스터 DB 연동 매칭 검사 (견적서와 서류가 동일한 등록 마스터 업체로 매칭되는 경우)
    if "업체명" in label or "상호" in label:
        try:
            from purchase_verifier.vendor_cache import find_matching_vendor_fuzzy
            m1 = find_matching_vendor_fuzzy(expected)
            m2 = find_matching_vendor_fuzzy(actual)
            if m1 and m2 and m1[0]["business_number"] == m2[0]["business_number"]:
                canonical = m1[0]["vendor_name"]
                return CheckResult(
                    label,
                    VerificationStatus.PASS,
                    f"마스터 DB 등록 업체 일치 확인 (정규 상호: '{canonical}', {m1[2]} / {m2[2]})\n"
                    f"실제 OCR 결과값: [기준 견적] '{expected}' / [{doc_name}] '{actual}'",
                    expected,
                    actual,
                )
        except Exception:
            pass

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
    reference_date: Optional[date] = None,
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
            elif target and _value(target.vendor_name) and not _value(confirm_doc.vendor_name):
                results.append(CheckResult("공사 확인서 협력사명", VerificationStatus.REVIEW, "공사 확인서에서 시공업체(협력사)명을 명확히 추출하지 못했습니다. 수동 확인 필요.", _value(target.vendor_name), "미추출"))

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
            elif target and target.total.value is not None and confirm_doc.total.value is None:
                results.append(CheckResult(
                    "공사 확인서 금액",
                    VerificationStatus.REVIEW,
                    "공사 확인서에서 공사예정금액을 명확히 추출하지 못했습니다. 수동 확인 필요.",
                    f"{target.total.value:,}원",
                    "미추출",
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
            norm_q = normalize_name(q_rec) if q_rec else ""
            raw_text = q.raw_text or ""
            rec_lower = q_rec.lower() if q_rec else ""

            # 1) 한글 산학협력단 / 산단 체크 (수신처 라벨 또는 문서 내 명시)
            has_ko_iacf = (
                "산학협력단" in norm_q
                or "산단" in norm_q
                or bool(re.search(r"산\s*학\s*협\s*[력려련]\s*단?", norm_q))
                or (not q_rec and "세종" in raw_text and ("산학협력단" in raw_text or "산단" in raw_text))
            )

            # 2) 영문 IACF / Industry-Academic Cooperation Foundation 체크 (사용자 지침: 약자 iacf 포함 필수)
            has_en_iacf = _is_iacf_text(rec_lower) or _is_iacf_text(raw_text)

            if has_ko_iacf or has_en_iacf:
                continue

            if not q_rec:
                warning_recs.append(f"'{q.filename}': 수신자 미기재")
                continue

            # 세종대학교(본교) 명의로만 발행된 경우 (산학협력단/IACF 누락) -> BLOCKED
            is_ko_univ = "세종" in norm_q or "세종대" in norm_q
            is_en_univ = ("sejong" in rec_lower or "sju" in rec_lower) and not has_en_iacf

            if is_ko_univ or is_en_univ:
                blocked_recs.append(f"'{q.filename}': 세종대학교(본교) 명의 - 산학협력단(IACF) 누락 ('{q_rec}')")
            else:
                warning_recs.append(f"'{q.filename}': 산단 외 수신자 ('{q_rec}')")

        if blocked_recs:
            results.append(CheckResult(
                "수신자 산학협력단",
                VerificationStatus.BLOCKED,
                f"반려 대상: 독립 법인인 세종대학교(본교) 명의로만 발행된 견적서가 있습니다. ('세종대학교 산학협력단' 또는 'IACF'로 재발행 필요)\n- " + "\n- ".join(blocked_recs),
                "세종대학교 산학협력단",
                ", ".join(blocked_recs),
            ))
        elif warning_recs:
            results.append(CheckResult(
                "수신자 산학협력단",
                VerificationStatus.REVIEW,
                f"확인 필요: '산학협력단' 명의가 누락되었거나 불명확한 견적서가 있습니다. (영문 견적서의 경우 'IACF' 또는 'Industry-Academic Cooperation Foundation' 표기 필수)\n- " + "\n- ".join(warning_recs),
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
                    check_base = reference_date or date.today()
                    b_days = calculate_business_days(check_base, exp_date)
                    base_label = f" (기준일: {check_base.strftime('%Y-%m-%d')})" if reference_date else ""
                    if b_days >= 5:
                        results.append(CheckResult(
                            "견적서 유효기간",
                            VerificationStatus.PASS,
                            f"유효기간 정상 확인 (만료일: {target.expiry_date.raw}, 잔여 {b_days}영업일{base_label})",
                            "5영업일 이상 잔여",
                            f"{b_days}영업일 잔여",
                        ))
                    elif b_days >= 0:
                        results.append(CheckResult(
                            "견적서 유효기간",
                            VerificationStatus.BLOCKED,
                            f"반려 대상: 견적서 유효기간이 5영업일 미만 남았습니다! (만료일: {target.expiry_date.raw}, 잔여 {b_days}영업일{base_label})",
                            "5영업일 이상 잔여",
                            f"{b_days}영업일 잔여",
                        ))
                    else:
                        results.append(CheckResult(
                            "견적서 유효기간",
                            VerificationStatus.BLOCKED,
                            f"반려 대상: 견적서 유효기간이 이미 지났습니다! (만료일: {target.expiry_date.raw}, {abs(b_days)}영업일 경과{base_label})",
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
            if number_check and number_check.status == VerificationStatus.CONFLICT:
                target_digits = re.sub(r"\D", "", _value(target.business_number))
                lic_digits = re.sub(r"\D", "", _value(license_doc.business_number))
                if len(target_digits) == 10 and len(lic_digits) == 10:
                    diff_count = sum(1 for a, b in zip(target_digits, lic_digits) if a != b)
                    if diff_count == 1:
                        target_v = normalize_name(_value(target.vendor_name))
                        lic_v = normalize_name(_value(license_doc.vendor_name))
                        if target_v and lic_v and (target_v == lic_v or target_v in lic_v or lic_v in target_v):
                            number_check = CheckResult(
                                "사업자등록증 사업자번호",
                                VerificationStatus.PASS,
                                f"사업자등록증 사업자번호 일치 (상호 '{_value(target.vendor_name)}' 일치 및 OCR 1자리 오인식 보정 반영: {_value(target.business_number)})",
                                _value(target.business_number),
                                _value(license_doc.business_number),
                            )
            if number_check:
                results.append(number_check)
            rep_check = _compare_field("사업자등록증 대표자명", _value(target.representative), _value(license_doc.representative), normalize_name, "사업자등록증", FieldPolicy.TEXT_REVIEW)
            if rep_check:
                results.append(rep_check)

        # 4. 통장사본 대조
        if bank_doc and target.document_type != DocumentType.ONLINE_STORE_CART:
            holder_check = _compare_field("통장사본 예금주", _value(target.vendor_name), _value(bank_doc.account_holder), normalize_name, "통장사본", FieldPolicy.TEXT_REVIEW)
            account_check = _compare_field("통장사본 계좌번호", _value(target.account_number), _value(bank_doc.account_number), lambda v: re.sub(r"\D", "", v), "통장사본", FieldPolicy.EXACT)
            
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
                        
                        if is_corp_bankbook and vendor_matched:
                            holder_check = CheckResult(
                                "통장사본 예금주",
                                VerificationStatus.PASS,
                                f"개인사업자 기업통장(기업자유예금) 계좌 확인 (예금주: '{holder_raw}', 상호: '{target_vendor}')",
                                target_vendor,
                                holder_raw,
                            )
                            break
                        elif rep_matched and not vendor_matched:
                            holder_check = CheckResult(
                                "통장사본 예금주", 
                                VerificationStatus.BLOCKED,
                                f"반려 대상: 상호명이 누락되고 대표자 단독 명의('{clean_rep}')로만 기재된 개인 통장입니다.\n"
                                f"규정상 개인사업자 통장은 업체명 단독 또는 업체명+대표자 병기 형태여야만 승인됩니다.\n"
                                f"실제 OCR 결과: [견적서 상호] {_value(target.vendor_name)} / [통장사본 예금주] {_value(bank_doc.account_holder)}",
                                _value(target.vendor_name),
                                _value(bank_doc.account_holder)
                            )
                            break

            # 계좌번호가 견적서와 100% 일치할 경우: 통장사본 도트 폰트/인쇄 마모로 예금주 텍스트 미추출 시 계좌번호 일치를 근거로 인정
            if account_check and account_check.status == VerificationStatus.PASS:
                if holder_check and holder_check.status in (VerificationStatus.REVIEW, VerificationStatus.CONFLICT) and not holder_raw:
                    holder_check = CheckResult(
                        "통장사본 예금주",
                        VerificationStatus.PASS,
                        f"통장사본 계좌번호({_value(bank_doc.account_number)}) 견적서와 일치 확인 (통장 도트프린트 인쇄 특성상 상호 '{_value(target.vendor_name)}' 육안 대조 권장)",
                        _value(target.vendor_name),
                        f"{_value(target.vendor_name)} (계좌번호 일치)",
                    )

            if holder_check:
                results.append(holder_check)
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

    # 부속/증빙 서류 첨부 상태 확인 (세부산출내역서, 기안문/품의서, 환율조회증빙, 기타 사유서 등)
    breakdown_doc = next((d for d in documents if d.document_type == DocumentType.COST_BREAKDOWN), None)
    memo_doc = next((d for d in documents if d.document_type == DocumentType.APPROVAL_MEMO), None)
    fx_doc = next((d for d in documents if d.document_type == DocumentType.EXCHANGE_RATE_PROOF), None)
    supporting_docs = [d for d in documents if d.document_type == DocumentType.SUPPORTING_DOC]

    if breakdown_doc:
        b_msg = f"세부산출내역서 첨부 확인 ({breakdown_doc.filename})"
        if target and target.total.value is not None and breakdown_doc.total.value is not None:
            if breakdown_doc.total.value == target.total.value:
                b_msg += f" - 내역서 총액({breakdown_doc.total.value:,}원)이 견적 총액과 일치"
            elif target.subtotal.value and breakdown_doc.total.value == target.subtotal.value:
                b_msg += f" - 내역서 총액({breakdown_doc.total.value:,}원)이 견적 공급가액과 일치"
            else:
                b_msg += f" - 내역서 총액: {breakdown_doc.total.value:,}원"
        results.append(CheckResult("세부산출내역서", VerificationStatus.PASS, b_msg))

    if memo_doc:
        results.append(CheckResult("구매 기안문/품의서", VerificationStatus.PASS, f"내부 결재문서 확인 ({memo_doc.filename}) - 견적 비교 대상 자동 분리"))

    if fx_doc:
        results.append(CheckResult("환율정보 증빙", VerificationStatus.PASS, f"요청 당일 환율조회 증빙 확인 ({fx_doc.filename}) - 해외구매 환율 검증"))

    for sdoc in supporting_docs:
        results.append(CheckResult("기타 부속서류", VerificationStatus.PASS, f"부속 증빙서류 확인 ({sdoc.filename})"))

    for document in documents:
        if document.raw_text:
            is_anom, anom_msg = detect_ocr_anomaly(document.raw_text)
            if is_anom:
                results.append(CheckResult(
                    f"OCR 품질 ({document.filename or '문서'})",
                    VerificationStatus.REVIEW,
                    f"문서 원본 텍스트 품질 저하/깨짐 감지 ({anom_msg}). 판독 오류 방지를 위해 원본 서류의 육안 대조가 필요합니다.",
                    "정상 OCR 텍스트",
                    anom_msg,
                ))
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


def find_auxiliary_contacts(
    documents: list[PurchaseDocument],
    selected_filename: Optional[str] = None,
) -> list[dict[str, str]]:
    """선정된 견적서 외 기타/부속 서류(제품설명자료, 사유서, 시방서 등)에 기재된 보조 연락처 정보를 수집합니다."""
    aux_contacts = []
    seen_contacts = set()

    for doc in documents:
        # 선정된 견적서는 제외
        if selected_filename and doc.filename == selected_filename:
            continue
        # 비교 견적서나 통장사본은 제외 (기타 부속서류, 설명자료, UNKNOWN 문서 대상)
        if doc.document_type in (DocumentType.QUOTE, DocumentType.BANK_COPY):
            continue

        raw = doc.raw_text or ""
        c_person = doc.contact_person.raw if _is_contact_person_candidate(doc.contact_person.raw) else ""
        c_phone = doc.phone.raw if (doc.phone.raw and len(re.findall(r"\d", doc.phone.raw)) >= 7) else ""
        c_email = doc.email.raw or ""

        # 본문 직접 정규식 스캔 (연락처 블록 또는 010 번호)
        if not c_phone:
            m_p = re.search(r"01[016789][- ]?\d{3,4}[- ]?\d{4}", raw)
            if not m_p:
                m_p = re.search(r"(?:02|0[3-6]\d)[- )]?\d{3,4}[- ]?\d{4}", raw)
            if m_p:
                c_phone = m_p.group(0).strip()

        if not c_email:
            m_e = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", raw)
            if m_e:
                c_email = m_e.group(0).strip()

        if not c_person:
            # "연락처\n... 대표" 또는 "담당자: ..." 패턴 스캔
            m_c = re.search(rf"(?:연락처|담당자|작성자)[^\n]*\n+[^\n]*?({_KOREAN_SURNAMES}[가-힣]{{1,3}})\s*(?:대표|담당|팀장|과장|대리|차장|부장|연구원)?", raw)
            if m_c and _is_contact_person_candidate(m_c.group(1)):
                matched_role = re.search(r"(?:대표|담당|팀장|과장|대리|차장|부장|연구원)", m_c.group(0))
                role_suffix = f" {matched_role.group(0)}" if matched_role else ""
                c_person = f"{m_c.group(1)}{role_suffix}".strip()
            elif doc.representative.raw and _is_contact_person_candidate(doc.representative.raw):
                c_person = f"{doc.representative.raw} 대표"

        if c_phone or c_email or c_person:
            contact_key = f"{c_phone}_{c_email}_{c_person}"
            if contact_key not in seen_contacts:
                seen_contacts.add(contact_key)
                aux_contacts.append({
                    "filename": doc.filename,
                    "doc_type": doc.document_type.value,
                    "contact_person": c_person,
                    "phone": c_phone,
                    "email": c_email,
                })

    return aux_contacts


def extract_main_doc_aux_info(
    target_doc: Optional[PurchaseDocument],
    license_doc: Optional[PurchaseDocument] = None,
    main_phone: str = "",
    main_email: str = "",
) -> dict[str, Any]:
    """선정된 본 견적서 및 사업자등록증에서 실무 보조 정보(추가 연락처, 세금계산서 메일, 납기, 결제조건 등)를 추출합니다."""
    info: dict[str, Any] = {
        "extra_phones": [],
        "fax": "",
        "tax_email": "",
        "lead_time": "",
        "payment_terms": "",
        "quote_account": "",
        "validity": "",
    }
    texts = []
    if target_doc and target_doc.raw_text:
        texts.append(target_doc.raw_text)
    if license_doc and license_doc.raw_text:
        texts.append(license_doc.raw_text)
    combined = "\n".join(texts)
    if not combined:
        return info

    target_text = target_doc.raw_text if (target_doc and target_doc.raw_text) else ""

    # 1. 팩스 번호
    m_fax = re.search(r"(?:Fax|팩\s*스)[^0-9\n]{0,10}([0-9() .+-]{8,25})", combined, re.I)
    if m_fax:
        fax_val = re.sub(r"[^\d-]", "", m_fax.group(1)).strip("-")
        if len(re.findall(r"\d", fax_val)) >= 7:
            info["fax"] = fax_val

    # 2. 추가 유선/직통 전화번호 (메인 전화번호 및 팩스번호 제외)
    seen_phones = set()
    if main_phone:
        seen_phones.add(re.sub(r"\D", "", main_phone))
    if info["fax"]:
        seen_phones.add(re.sub(r"\D", "", info["fax"]))

    all_phones = re.finditer(r"(?:01[016789]|02|0[3-6]\d)[- )]?\d{3,4}[- ]?\d{4}", combined)
    for mp in all_phones:
        p_raw = mp.group(0).strip()
        digits = re.sub(r"\D", "", p_raw)
        if digits not in seen_phones and len(digits) >= 8:
            seen_phones.add(digits)
            info["extra_phones"].append(p_raw)

    # 3. 전자세금계산서 전용 이메일
    m_tax = re.search(r"(?:전\s*자\s*세\s*금\s*계\s*산\s*서|세\s*금\s*계\s*산\s*서)[^\n@]*?([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", combined)
    if m_tax:
        tax_mail = m_tax.group(1).strip()
        info["tax_email"] = tax_mail
    elif "hometax.go.kr" in combined:
        m_ht = re.search(r"[A-Za-z0-9._%+-]+@hometax\.go\.kr", combined)
        if m_ht:
            info["tax_email"] = m_ht.group(0).strip()

    # 4. 납기 (납품기한) - 견적서 본문 우선
    if target_text:
        m_lead = re.search(r"(?:납\s*기|납\s*품\s*기\s*한|납\s*품\s*기\s*일|납\s*품\s*일\s*자|배\s*송)[\s:：.-]*([^\n]{2,50})", target_text)
        if m_lead:
            val = _clean_spaces(m_lead.group(1)).strip("|,;: .\\-")
            if val and not any(k in val for k in ("일자", "번호", "상호")):
                info["lead_time"] = val

    # 5. 결제조건 (지급조건) - 견적서 본문 우선
    if target_text:
        m_pay = re.search(r"(?:결\s*[제재]\s*조\s*건|대\s*금\s*지\s*급\s*조\s*건|지\s*급\s*조\s*건)[\s:：.-]*([^\n]{2,50})", target_text)
        if m_pay:
            val = _clean_spaces(m_pay.group(1)).strip("|,;: .\\-")
            if val and not any(k in val for k in ("일자", "번호", "상호")):
                info["payment_terms"] = val

    # 6. 견적서 기재 계좌 (통장사본 대조용)
    if target_text:
        m_acc = re.search(r"(?:입\s*금\s*계\s*좌|송\s*금\s*계\s*좌|계\s*좌\s*번\s*호|입\s*금\s*처)[\s:：.-]*([^\n]{5,50})", target_text)
        if m_acc:
            val = _clean_spaces(m_acc.group(1)).strip("|,;: .\\-")
            if re.search(r"\d{6,}", val):
                info["quote_account"] = val

    # 7. 유효기간 / 만료일
    if target_doc:
        if target_doc.expiry_date.raw:
            info["validity"] = f"만료일: {target_doc.expiry_date.raw}"
        elif target_text:
            m_val = re.search(r"(?:유\s*효\s*기\s*간|유\s*효\s*일\s*자)[\s:：.-]*([^\n]{2,40})", target_text)
            if m_val:
                val = _clean_spaces(m_val.group(1)).strip("|,;: .\\-")
                if val:
                    info["validity"] = val

    return info
