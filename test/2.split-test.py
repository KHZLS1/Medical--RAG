from app.data_loader import load_medical_documents
from app.text_split import split_documents

docs = load_medical_documents(limit_per_file=1000)
chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
print(f"原始 {len(docs)} 条 → 切分后 {len(chunks)} 条")