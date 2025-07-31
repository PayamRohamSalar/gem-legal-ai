# scripts/00_run_data_pipeline.py
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.document_extractor import LegalDocumentExtractor
from src.data_processing.smart_chunker import LegalTextChunker
from src.retrieval.vector_database import LegalVectorDatabase

# --- تنظیمات ---
RAW_DOCS_PATH = "data/raw_documents"
PROCESSED_TEXT_PATH = "data/processed_chunks"
CHUNKS_OUTPUT_PATH = "data/chunks"
DB_PATH = "data/vector_db"
COLLECTION_NAME = "legal_hybrid_v1" # نام جدید برای دیتابیس هیبرید

def main():
    print("🚀=============== START: HYBRID DATA PIPELINE ===============🚀")

    # === بخش ۱: استخراج و پاک‌سازی ===
    print("\n--- STAGE 1: Extracting and Cleaning Documents ---")
    extractor = LegalDocumentExtractor(input_dir=RAW_DOCS_PATH, output_dir=PROCESSED_TEXT_PATH)
    processed_docs_info = extractor.process_all_documents()

    if not processed_docs_info:
        print("❌ Pipeline stopped. No documents processed.")
        return

    # === بخش ۲: تقسیم‌بندی هوشمند ===
    print("\n--- STAGE 2: Performing Smart Chunking ---")
    chunker = LegalTextChunker(chunk_size=400, chunk_overlap=80, min_chunk_size=100)
    
    # این دستور تمام chunk ها را از فایل‌های json ذخیره شده در مرحله قبل، بارگذاری می‌کند
    chunker.save_chunks_from_processed_docs(PROCESSED_TEXT_PATH, CHUNKS_OUTPUT_PATH)
    
    # === بخش ۳: ساخت پایگاه داده برداری ===
    print("\n--- STAGE 3: Building Vector Database ---")
    db = LegalVectorDatabase(db_path=DB_PATH, collection_name=COLLECTION_NAME)
    db.reset_database() # پاک کردن دیتابیس قبلی برای اطمینان
    
    # اضافه کردن تمام chunk های ذخیره شده به دیتابیس
    success = db.add_chunks_from_files(chunks_dir=CHUNKS_OUTPUT_PATH)

    if success:
        print("\n✅ Vector Database built successfully!")
        stats = db.get_statistics()
        print(f"📊 Final Stats: {stats.get('total_documents', 0)} chunks indexed.")
    else:
        print("\n❌ Failed to build Vector Database.")
    
    print("\n🚀=============== END: HYBRID DATA PIPELINE ===============🚀")

# یک تابع کمکی به کلاس چانکر اضافه می‌کنیم تا خواندن فایل‌ها راحت‌تر باشد
def save_chunks_from_processed_docs(self, processed_dir, output_dir):
    text_files = [f for f in os.listdir(processed_dir) if f.endswith('_cleaned.txt')]
    for test_file in text_files:
        document_name = test_file.replace('_cleaned.txt', '')
        with open(os.path.join(processed_dir, test_file), 'r', encoding='utf-8') as f:
            text = f.read()
        metadata_file = os.path.join("data/metadata", f"{document_name}_metadata.json")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            doc_metadata = json.load(f)
        chunks = self.chunk_document(text, doc_metadata['metadata'])
        self.save_chunks(chunks, output_dir, document_name)

# این متد را به کلاس LegalTextChunker اضافه می‌کنیم
LegalTextChunker.save_chunks_from_processed_docs = save_chunks_from_processed_docs

if __name__ == "__main__":
    main()