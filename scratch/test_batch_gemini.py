# -*- coding: utf-8 -*-
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv('C:/Purchasing Tool/.env')

from google import genai
from google.genai import types

from purchase_verifier.core import (
    PurchaseDocument, DocumentType, Evidence, Amount, Item,
    verify_documents, find_lowest_quote
)

folder = Path(r"C:\Users\SANDAN\.gemini\antigravity-ide\brain\1836eed3-a34b-4802-989b-e23f289ef0d7\.user_uploaded")
files = sorted(folder.glob("media_1791865224*.pdf"))
print(f"Loading {len(files)} files...")

contents = []
for f in files:
    contents.append(f"--- 파일명: {f.name} ---")
    contents.append(types.Part.from_bytes(data=f.read_bytes(), mime_type="application/pdf"))

prompt = """당신은 대학교 산학협력단 및 공공기관의 구매/회계 감사 전문 인공지능입니다.
첨부된 서류들을 정밀 분석하여 각 파일별로 아래 JSON 배열(Array of Objects) 형식으로 응답하세요:
[
  {
    "filename": "첨부된 파일명 정확히 일치",
    "document_type": "QUOTE(견적서) | BUSINESS_LICENSE(사업자등록증) | BANK_COPY(통장사본) | CONSTRUCTION_CONFIRMATION(시설공사/장비설치확인서) | SPECIFICATION(사유서/시방서) | UNKNOWN(기타)",
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
        "quantity": 수량(숫자),
        "unit_price": 단가(숫자),
        "amount": 금액(숫자)
      }
    ]
  }
]
반드시 위 JSON 형식으로만 응답하세요."""

contents.append(prompt)

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
response = client.models.generate_content(
    model='gemini-3.8-flash',
    contents=contents,
    config=types.GenerateContentConfig(response_mime_type='application/json')
)

with open('scratch/batch_gemini_res.json', 'w', encoding='utf-8') as f:
    f.write(response.text)

print("=== Gemini Batch Extraction Completed! ===")
import json
data = json.loads(response.text)
for item in data:
    print(f"[{item.get('document_type')}] {item.get('filename')}")
    print(f"  상호: {item.get('vendor_name')} / 대표: {item.get('representative')} / 사업자: {item.get('biz_number')}")
    print(f"  총액: {item.get('total_amount')} / 수신자: {item.get('recipient')}")
    print(f"  계좌: {item.get('bank_name')} {item.get('account_number')} ({item.get('account_holder')})")
    print(f"  연락처: {item.get('phone')} / {item.get('email')}")
    print(f"  품목수: {len(item.get('items', []))}")
    print("-" * 50)
