"""测试后端 lifespan 中的关键操作是否正常"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("1. Importing app.database...")
from app.database import Base, engine
print("   OK")

print("2. Creating tables (Base.metadata.create_all)...")
Base.metadata.create_all(bind=engine)
print("   OK")

print("3. Importing app.rag_chain...")
from app.rag_chain import stream_answer
print("   OK")

print("4. Importing app.vectorstore...")
from app.vectorstore import get_milvus_client, get_retriever
from app.config import settings

client = get_milvus_client()
stats = client.get_collection_stats(settings.milvus_collection)
print(f"   OK (collection={settings.milvus_collection}, rows={int(stats.get('row_count', 0)) if stats else 0})")

print("5. Importing app.llm...")
from app.llm import get_llm
print("   OK")

print("6. Testing DeepSeek LLM connection...")
llm = get_llm()
resp = llm.invoke("say hi")
print(f"   LLM response: {resp.content[:50]}")

print("\nALL TESTS PASSED")
