# -*- coding: utf-8 -*-
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from purchase_verifier.ocr import extract_text_from_pdf_data
from purchase_verifier.core import extract_document

folder = Path(r"C:\Users\SANDAN\.gemini\antigravity-ide\brain\1836eed3-a34b-4802-989b-e23f289ef0d7\.user_uploaded")
recent_files = sorted(folder.glob("media_1791865224*.pdf"))

out_path = Path("scratch/raw_texts_utf8.txt")
with open(out_path, "w", encoding="utf-8") as out:
    for f in recent_files:
        text = extract_text_from_pdf_data(f.read_bytes())
        doc = extract_document(text, f.name)
        out.write(f"==================== {f.name} ({doc.document_type.value}) ====================\n")
        out.write(f"[파싱 결과]\n")
        out.write(f"- 상호: {doc.vendor_name.raw}\n")
        out.write(f"- 사업자번호: {doc.business_number.raw}\n")
        out.write(f"- 대표자: {doc.representative.raw}\n")
        out.write(f"- 금액: {doc.total.value}\n")
        out.write(f"- 수신자: {doc.recipient.raw}\n")
        out.write(f"- 견적일자: {doc.quote_date.raw}\n")
        out.write(f"- 계좌번호: {doc.account_number.raw}\n")
        out.write(f"- 담당자: {doc.contact_person.raw}\n")
        out.write(f"- 연락처: {doc.phone.raw}\n")
        out.write(f"- 이메일: {doc.email.raw}\n")
        out.write(f"[OCR 텍스트 원문]\n{text}\n\n")

print("Done writing to scratch/raw_texts_utf8.txt")
