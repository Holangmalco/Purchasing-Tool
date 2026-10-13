# -*- coding: utf-8 -*-
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv('C:/Purchasing Tool/.env')

from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

pdf_path = r'C:\Users\SANDAN\.gemini\antigravity-ide\brain\1836eed3-a34b-4802-989b-e23f289ef0d7\.user_uploaded\media_1791865224125.pdf'
with open(pdf_path, 'rb') as f:
    pdf_bytes = f.read()

prompt = """이 문서는 한국 사업자등록증입니다. 아래 JSON 형식으로만 정확하게 파싱하여 출력하세요:
{
  "document_type": "BUSINESS_LICENSE",
  "vendor_name": "상호(법인명)",
  "representative": "성명(대표자)",
  "biz_number": "등록번호 (000-00-00000)",
  "open_date": "개업연월일",
  "address": "사업장소재지",
  "tax_email": "전자세금계산서 전용 메일"
}"""

response = client.models.generate_content(
    model='gemini-3.8-flash',
    contents=[
        types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
        prompt
    ],
    config=types.GenerateContentConfig(response_mime_type='application/json')
)

print('=== Gemini Biz License Extraction Result ===')
print(response.text)
