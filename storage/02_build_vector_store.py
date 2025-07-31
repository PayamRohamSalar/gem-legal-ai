# scripts/02_build_vector_store.py
import sys
import os
import json
from tqdm import tqdm

# افزودن مسیر ریشه پروژه به sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.vector_db_handler import VectorDBHandler

# تعریف مسیرهای اصلی
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_JSONL_PATH = os.path.join(ROOT_DIR, 'data', 'processed_data.jsonl')
VECTOR_STORE_PATH = os.path.join(ROOT_DIR, 'data', 'vector_store')
COLLECTION_NAME = "legal_documents_v1" # یک نام برای مجموعه خود انتخاب کنید

def main():
    """
    تابع اصلی برای اجرای کامل فاز ۲:
    1. خواندن داده‌های پردازش‌شده از فایل jsonl.
    2. آماده‌سازی داده‌ها برای ورود به پایگاه داده برداری.
    3. ایجاد پایگاه داده و افزودن داده‌ها به آن.
    """
    print("🚀 Starting Phase 2: Building Vector Store...")

    # 1. خواندن داده‌های پردازش‌شده
    print(f"Reading processed data from {INPUT_JSONL_PATH}...")
    try:
        with open(INPUT_JSONL_PATH, 'r', encoding='utf-8') as f:
            all_docs = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_JSONL_PATH}")
        print("Please make sure you have successfully completed Phase 1.")
        return

    # 2. آماده‌سازی داده‌ها برای ChromaDB
    # ما به لیست‌هایی از متون، متادیتا و شناسه‌ها نیاز داریم
    all_chunks_texts = []
    all_chunks_metadatas = []
    all_chunks_ids = []

    for doc in all_docs:
        for chunk in doc['chunks']:
            all_chunks_texts.append(chunk['text'])
            all_chunks_ids.append(chunk['chunk_id'])
            # متادیتای هر chunk را برای فیلتر کردن در آینده ذخیره می‌کنیم
            all_chunks_metadatas.append({
                "document_id": doc['document_id'],
                "document_title": doc['title'],
                "article_number": chunk.get('number', 'N/A'),
                "chunk_type": chunk.get('type', 'N/A')
            })
            
    print(f"Prepared a total of {len(all_chunks_ids)} chunks for indexing.")

    # 3. ایجاد و پر کردن پایگاه داده
    db_handler = VectorDBHandler(db_path=VECTOR_STORE_PATH)
    collection = db_handler.get_or_create_collection(name=COLLECTION_NAME)

    # افزودن داده‌ها به صورت دسته‌ای (batch) برای کارایی بهتر
    batch_size = 100
    print(f"Adding chunks to vector store in batches of {batch_size}...")
    for i in tqdm(range(0, len(all_chunks_ids), batch_size)):
        batch_ids = all_chunks_ids[i:i+batch_size]
        batch_texts = all_chunks_texts[i:i+batch_size]
        batch_metadatas = all_chunks_metadatas[i:i+batch_size]
        
        db_handler.add_batch(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadatas
        )

    print("\nVector store has been built and populated successfully! ✅")
    print(f"Total items in collection: {collection.count()}")

if __name__ == "__main__":
    main()