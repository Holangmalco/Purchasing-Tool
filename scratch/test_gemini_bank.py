# -*- coding: utf-8 -*-
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv('C:/Purchasing Tool/.env')

from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

pdf_path = r'C:\Users\SANDAN\.gemini\antigravity-ide\brain\1836eed3-a34b-4802-989b-e23f289ef0d7\.user_uploaded\media_1791865224130.pdf'
with open(pdf_path, 'rb') as f:
    pdf_bytes = f.read()

prompt = """이 문서는 한국 통장사본입니다. 아래 JSON 형식으로만 정확하게 파싱하여 출력하세요:
{
  "document_type": "BANK_COPY",
  "bank_name": "은행명",
  "account_holder": "예금주명",
  "account_number": "계좌번호"
}"""

response = client.models.generate_content(
    model='gemini-3.8-flash',
    contents=[
        types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
        prompt
    ],
    config=types.GenerateContentConfig(response_mime_type='application/json')
)

print('=== Gemini Bank Copy Extraction Result ===')
print(response.text)
