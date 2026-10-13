# -*- coding: utf-8 -*-
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from purchase_verifier.core import extract_document, verify_documents, find_lowest_quote, DocumentType
from purchase_verifier.ocr import extract_text_from_pdf_data

folder = Path(r"C:\Users\SANDAN\.gemini\antigravity-ide\brain\1836eed3-a34b-4802-989b-e23f289ef0d7\.user_uploaded")
recent_files = sorted(folder.glob("media_1791865224*.pdf"))

docs = []
for f in recent_files:
    text = extract_text_from_pdf_data(f.read_bytes())
    doc = extract_document(text, f.name)
    docs.append(doc)
    print(f"=== [{doc.document_type.value}] {f.name} ===")
    print(f"상호: {doc.vendor_name.raw}")
    print(f"사업자번호: {doc.business_number.raw}")
    print(f"대표자: {doc.representative.raw}")
    print(f"총액: {doc.total.value}")
    print(f"수신자: {doc.recipient.raw}")
    print(f"견적일자: {doc.quote_date.raw}")
    print(f"유효기간: {doc.expiry_date.raw}")
    print(f"계좌번호: {doc.account_number.raw}")
    print(f"담당자: {doc.contact_person.raw}")
    print(f"연락처: {doc.phone.raw}")
    print(f"이메일: {doc.email.raw}")
    print(f"경고: {doc.warnings}")
    print("-" * 50)

lowest = find_lowest_quote(docs)
print(f"\n최저가 견적서: {lowest.candidate_filename} / {lowest.amount:,}원 / 확정여부: {lowest.confirmed} ({lowest.reason})\n")

results = verify_documents(docs, purchase_type="공사")
print("=== 종합 검증 결과 (시설공사 기준) ===")
for r in results:
    print(f"[{r.status.value}] {r.name}: {r.detail}")
