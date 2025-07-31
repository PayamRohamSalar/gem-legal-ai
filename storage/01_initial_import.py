# scripts/01_initial_import.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from src.data_processing.document_loader import load_docx_documents
from src.data_processing.text_cleaner import clean_text
from src.data_processing.chunker import chunk_document_by_article

# تعریف مسیرهای اصلی پروژه
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DOCS_PATH = os.path.join(ROOT_DIR, 'data', 'raw_documents')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data', 'processed_data.jsonl')

def main():
    """
    تابع اصلی برای اجرای کامل فاز ۱:
    1. بارگذاری اسناد خام
    2. پاکسازی متن هر سند
    3. تقسیم‌بندی هر سند به chunk
    4. ذخیره نتیجه نهایی در فایل jsonl
    """
    print("🚀 Starting Phase 1: Data Processing Pipeline...")

    # 1. بارگذاری اسناد
    raw_documents = load_docx_documents(RAW_DOCS_PATH)
    if not raw_documents:
        print("No documents found. Please place .docx files in data/raw_documents/")
        return

    all_processed_docs = []

    # 2. پردازش هر سند
    for doc in raw_documents:
        print(f"\nProcessing document: {doc['filename']}...")
        
        # استخراج شناسه سند از نام فایل
        doc_id = os.path.splitext(doc['filename'])[0]
        
        # 3. پاکسازی متن
        cleaned_content = clean_text(doc['content'])
        
        # 4. تقسیم‌بندی به chunk
        chunks = chunk_document_by_article(doc_id, cleaned_content)
        
        # ساختار نهایی JSON برای این سند
        processed_doc = {
            "document_id": doc_id,
            "title": doc_id.replace('_', ' '), # یک عنوان ساده از نام فایل
            "source_filename": doc['filename'],
            "chunks": chunks
        }
        all_processed_docs.append(processed_doc)

    # 5. ذخیره خروجی
    print(f"\nWriting {len(all_processed_docs)} processed documents to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for doc in all_processed_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
    print("✅ Phase 1 completed successfully!")


if __name__ == "__main__":
    # اطمینان از وجود پوشه داده
    if not os.path.exists(RAW_DOCS_PATH):
        print(f"Error: Directory not found at {RAW_DOCS_PATH}")
        print("Please run '00_create_project_structure.py' first.")
    else:
        main()