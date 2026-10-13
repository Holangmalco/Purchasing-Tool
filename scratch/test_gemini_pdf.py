# -*- coding: utf-8 -*-
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv('C:/Purchasing Tool/.env')

from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

pdf_path = r'C:\Users\SANDAN\.gemini\antigravity-ide\brain\1836eed3-a34b-4802-989b-e23f289ef0d7\.user_uploaded\media_1791865224113.pdf'
with open(pdf_path, 'rb') as f:
    pdf_bytes = f.read()

prompt = """이 문서는 한국 구매 견적서입니다. 아래 JSON 형식으로만 정확하게 파싱하여 출력하세요:
{
  "document_type": "QUOTE",
  "vendor_name": "공급자 상호명",
  "representative": "대표자 성명",
  "biz_number": "사업자등록번호 (000-00-00000)",
  "total_amount": 000,
  "recipient": "수신처",
  "quote_date": "YYYY-MM-DD",
  "items": [{"name": "품목명", "amount": 000}]
}"""

response = client.models.generate_content(
    model='gemini-3.8-flash',
    contents=[
        types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
        prompt
    ],
    config=types.GenerateContentConfig(response_mime_type='application/json')
)

with open('scratch/gemini_extracted_quote.json', 'w', encoding='utf-8') as out_f:
    out_f.write(response.text)

print('=== Gemini 3.8 Flash Extraction Result ===')
print(response.text)
