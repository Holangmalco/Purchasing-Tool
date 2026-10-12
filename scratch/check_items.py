import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from purchase_verifier.ocr import extract_text_from_pdf_data
from purchase_verifier.core import extract_document, verify_documents

pdf_map = {
    'media_1791275416465.pdf': '흥진정밀_사업자등록증.pdf',
    'media_1791275416603.pdf': '흥진정밀_통장사본.pdf',
    'media_1791275416631.pdf': '견적서_흥진정밀.pdf',
    'media_1791275416665.pdf': '비교견적서_우주측기.pdf',
}
base_dir = r'C:\Users\SANDAN\.gemini\antigravity-ide\brain\9522c3d4-d761-48ce-8814-477b1416214e\.user_uploaded'

docs = []
for pfile, real_name in pdf_map.items():
    full_p = os.path.join(base_dir, pfile)
    with open(full_p, 'rb') as f:
        data = f.read()
    txt = extract_text_from_pdf_data(data)
    doc = extract_document(txt, filename=real_name)
    docs.append(doc)
    print(f'=== {real_name} ===')
    print(f'  Type: {doc.document_type.name}')
    print(f'  Vendor: {doc.vendor_name.raw}')
    print(f'  BizNum: {doc.business_number.raw}')
    print(f'  Rep: {doc.representative.raw}')
    print(f'  Holder: {doc.account_holder.raw}')
    print(f'  Account: {doc.account_number.raw}')
    print(f'  Total: {doc.total.value}')
    print(f'  Items ({len(doc.items)}):')
    for it in doc.items:
        print(f'    - {it.name!r} qty={it.quantity} price={it.unit_price} amt={it.amount}')

results = verify_documents(docs)
print('\n=== Verification Results ===')
for r in results:
    print(f'[{r.status.name}] {r.name}: {r.detail}')
