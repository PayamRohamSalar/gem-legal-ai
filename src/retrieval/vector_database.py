# src/retrieval/vector_database.py

import os
import json
import uuid
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions

class LegalVectorDatabase:
    """
    مدیریت Vector Database برای اسناد حقوقی
    
    این کلاس مسئول ذخیره، جستجو و مدیریت embeddings در ChromaDB است
    """
    
    def __init__(self, 
                 db_path: str = "data/vector_db",
                 collection_name: str = "legal_documents",
                 embedding_model_name: str = None):
        
        self.db_path = db_path
        self.collection_name = collection_name
        
        # تعیین نام صحیح مدل embedding
        if embedding_model_name is None:
            # استفاده از نام کامل صحیح پیش‌فرض
            self.embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        else:
            # تبدیل نام کلید به نام کامل مدل
            model_mapping = {
                'multilingual-mpnet': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                'multilingual-minilm': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'bert-fa-base': 'HooshvareLab/bert-fa-base-uncased',
                'all-minilm': 'sentence-transformers/all-MiniLM-L6-v2'
            }
            
            # اگر نام کلید است، تبدیل کن، وگرنه همان نام را استفاده کن
            self.embedding_model_name = model_mapping.get(embedding_model_name, embedding_model_name)
        
        # ایجاد فولدر دیتابیس
        os.makedirs(db_path, exist_ok=True)
        
        # راه‌اندازی ChromaDB
        self.client = None
        self.collection = None
        self.embedding_function = None
        
        self._initialize_database()
    
    def _initialize_database(self):
        """راه‌اندازی اولیه دیتابیس"""
        try:
            print(f"🔄 راه‌اندازی ChromaDB در: {self.db_path}")
            print(f"📋 مدل embedding: {self.embedding_model_name}")
            
            # ایجاد client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # تنظیم embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
            
            # ایجاد یا دریافت collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                print(f"✅ Collection موجود بارگذاری شد: {self.collection_name}")
                
            except Exception:
                # اگر collection وجود نداشت، ایجاد کن
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"description": "Legal documents collection for RAG system"}
                )
                print(f"✅ Collection جدید ایجاد شد: {self.collection_name}")
            
            # نمایش آمار
            count = self.collection.count()
            print(f"📊 تعداد documents در collection: {count}")
            
        except Exception as e:
            print(f"❌ خطا در راه‌اندازی دیتابیس: {str(e)}")
            raise
    
    def reset_database(self):
        """پاک کردن کامل دیتابیس"""
        try:
            print("⚠️ در حال پاک کردن کامل دیتابیس...")
            
            if self.client and self.collection:
                self.client.delete_collection(self.collection_name)
                print(f"✅ Collection {self.collection_name} پاک شد")
            
            # ایجاد مجدد
            self._initialize_database()
            
        except Exception as e:
            print(f"❌ خطا در reset دیتابیس: {str(e)}")
    
    def add_chunks_from_files(self, chunks_dir: str) -> bool:
        """
        اضافه کردن chunks از فایل‌های JSON
        
        Args:
            chunks_dir: مسیر فولدر حاوی فایل‌های chunk
            
        Returns:
            bool: موفقیت در اضافه کردن
        """
        if not os.path.exists(chunks_dir):
            print(f"❌ فولدر {chunks_dir} وجود ندارد!")
            return False
        
        print(f"🔄 شروع اضافه کردن chunks از: {chunks_dir}")
        
        all_chunks = []
        processed_files = 0
        
        # پیمایش تمام فولدرهای chunks
        for item in os.listdir(chunks_dir):
            item_path = os.path.join(chunks_dir, item)
            
            if os.path.isdir(item_path) and item.endswith('_chunks'):
                print(f"📁 پردازش فولدر: {item}")
                
                # خواندن تمام فایل‌های JSON در فولدر
                chunk_files = [f for f in os.listdir(item_path) if f.endswith('.json')]
                
                for chunk_file in tqdm(chunk_files, desc=f"بارگذاری {item}"):
                    chunk_path = os.path.join(item_path, chunk_file)
                    
                    try:
                        with open(chunk_path, 'r', encoding='utf-8') as f:
                            chunk_data = json.load(f)
                            all_chunks.append(chunk_data)
                    
                    except Exception as e:
                        print(f"❌ خطا در خواندن {chunk_file}: {str(e)}")
                        continue
                
                processed_files += 1
        
        if not all_chunks:
            print("❌ هیچ chunk معتبری یافت نشد!")
            return False
        
        print(f"📊 مجموع chunks یافت شده: {len(all_chunks)} از {processed_files} فایل")
        
        # اضافه کردن به دیتابیس
        return self.add_chunks(all_chunks)
    
    def add_chunks(self, chunks: List[Dict]) -> bool:
        """
        اضافه کردن لیستی از chunks به دیتابیس
        
        Args:
            chunks: لیست chunks برای اضافه کردن
            
        Returns:
            bool: موفقیت در اضافه کردن
        """
        if not chunks:
            print("❌ لیست chunks خالی است!")
            return False
        
        try:
            print(f"🔄 اضافه کردن {len(chunks)} chunk به دیتابیس...")
            
            # آماده‌سازی داده‌ها برای ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for chunk in tqdm(chunks, desc="آماده‌سازی chunks"):
                # تولید ID یکتا
                chunk_id = chunk.get('chunk_id', str(uuid.uuid4()))
                
                # متن chunk
                document_text = chunk.get('text', '')
                if not document_text.strip():
                    continue
                
                # metadata
                metadata = {
                    'chunk_index': chunk.get('chunk_index', 0),
                    'word_count': chunk.get('word_count', 0),
                    'char_count': chunk.get('char_count', 0),
                    'quality_score': chunk.get('quality_score', 0.0),
                    'document_title': chunk.get('document_title', ''),
                    'document_type': chunk.get('document_type', ''),
                    'authority': chunk.get('authority', ''),
                    'approval_date': chunk.get('approval_date', ''),
                    'structures': json.dumps(chunk.get('structures', []), ensure_ascii=False),
                    'keywords': json.dumps(chunk.get('keywords', []), ensure_ascii=False),
                    'legal_entities': json.dumps(chunk.get('legal_entities', []), ensure_ascii=False)
                }
                
                documents.append(document_text)
                metadatas.append(metadata)
                ids.append(chunk_id)
            
            if not documents:
                print("❌ هیچ document معتبری برای اضافه کردن یافت نشد!")
                return False
            
            # اضافه کردن به ChromaDB (در batches)
            batch_size = 100
            total_added = 0
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                try:
                    self.collection.add(
                        documents=batch_docs,
                        metadatas=batch_metas,
                        ids=batch_ids
                    )
                    total_added += len(batch_docs)
                    
                except Exception as e:
                    print(f"❌ خطا در اضافه کردن batch {i//batch_size + 1}: {str(e)}")
                    continue
            
            print(f"✅ {total_added} chunk با موفقیت اضافه شد")
            print(f"📊 مجموع documents در دیتابیس: {self.collection.count()}")
            
            return total_added > 0
            
        except Exception as e:
            print(f"❌ خطا در اضافه کردن chunks: {str(e)}")
            return False
    
    def search_semantic(self, 
                       query: str, 
                       n_results: int = 10,
                       filters: Dict = None) -> List[Dict]:
        """
        جستجوی معنایی در دیتابیس
        
        Args:
            query: متن جستجو
            n_results: تعداد نتایج
            filters: فیلترهای metadata
            
        Returns:
            list: نتایج جستجو
        """
        if not query.strip():
            return []
        
        try:
            print(f"🔍 جستجوی معنایی برای: '{query[:50]}...'")
            
            # تبدیل فیلترها به فرمت ChromaDB
            where_clause = None
            if filters:
                where_clause = self._build_where_clause(filters)
            
            # جستجو در ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # پردازش نتایج
            processed_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # تبدیل distance به similarity
                    
                    # Parse JSON fields
                    try:
                        metadata['structures'] = json.loads(metadata.get('structures', '[]'))
                        metadata['keywords'] = json.loads(metadata.get('keywords', '[]'))
                        metadata['legal_entities'] = json.loads(metadata.get('legal_entities', '[]'))
                    except:
                        pass
                    
                    result_item = {
                        'text': doc,
                        'metadata': metadata,
                        'similarity_score': float(similarity),
                        'distance': float(distance),
                        'rank': i + 1
                    }
                    
                    processed_results.append(result_item)
            
            print(f"✅ {len(processed_results)} نتیجه یافت شد")
            
            return processed_results
            
        except Exception as e:
            print(f"❌ خطا در جستجوی معنایی: {str(e)}")
            return []
    
    def _build_where_clause(self, filters: Dict) -> Dict:
        """تبدیل فیلترها به فرمت ChromaDB"""
        where_clause = {}
        
        for key, value in filters.items():
            if key == 'document_type' and value:
                where_clause['document_type'] = {"$eq": value}
            elif key == 'authority' and value:
                where_clause['authority'] = {"$eq": value}
            elif key == 'min_quality_score' and value:
                where_clause['quality_score'] = {"$gte": float(value)}
            elif key == 'min_word_count' and value:
                where_clause['word_count'] = {"$gte": int(value)}
        
        return where_clause
    
    def get_statistics(self) -> Dict:
        """دریافت آمار دیتابیس"""
        try:
            total_count = self.collection.count()
            
            if total_count == 0:
                return {
                    'total_documents': 0,
                    'message': 'دیتابیس خالی است'
                }
            
            # نمونه‌گیری برای آمارگیری
            sample_size = min(100, total_count)
            sample_results = self.collection.get(limit=sample_size, include=['metadatas'])
            
            if not sample_results['metadatas']:
                return {'total_documents': total_count}
            
            # محاسبه آمار
            doc_types = {}
            authorities = {}
            quality_scores = []
            word_counts = []
            
            for metadata in sample_results['metadatas']:
                # نوع سند
                doc_type = metadata.get('document_type', 'نامشخص')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                # مرجع
                authority = metadata.get('authority', 'نامشخص')
                authorities[authority] = authorities.get(authority, 0) + 1
                
                # کیفیت
                quality = metadata.get('quality_score', 0)
                if isinstance(quality, (int, float)):
                    quality_scores.append(float(quality))
                
                # تعداد کلمات
                word_count = metadata.get('word_count', 0)
                if isinstance(word_count, int):
                    word_counts.append(word_count)
            
            stats = {
                'total_documents': total_count,
                'sample_size': len(sample_results['metadatas']),
                'document_types': doc_types,
                'authorities': authorities,
                'quality_stats': {
                    'average': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                    'min': min(quality_scores) if quality_scores else 0,
                    'max': max(quality_scores) if quality_scores else 0
                },
                'word_count_stats': {
                    'average': sum(word_counts) / len(word_counts) if word_counts else 0,
                    'min': min(word_counts) if word_counts else 0,
                    'max': max(word_counts) if word_counts else 0
                }
            }
            
            return stats
            
        except Exception as e:
            return {'error': f'خطا در دریافت آمار: {str(e)}'}
    
    def search_by_filters(self, filters: Dict, limit: int = 50) -> List[Dict]:
        """جستجو بر اساس فیلترهای metadata"""
        try:
            where_clause = self._build_where_clause(filters)
            
            results = self.collection.get(
                where=where_clause,
                limit=limit,
                include=['documents', 'metadatas']
            )
            
            processed_results = []
            
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    metadata = results['metadatas'][i]
                    
                    # Parse JSON fields
                    try:
                        metadata['structures'] = json.loads(metadata.get('structures', '[]'))
                        metadata['keywords'] = json.loads(metadata.get('keywords', '[]'))
                        metadata['legal_entities'] = json.loads(metadata.get('legal_entities', '[]'))
                    except:
                        pass
                    
                    result_item = {
                        'text': doc,
                        'metadata': metadata,
                        'rank': i + 1
                    }
                    
                    processed_results.append(result_item)
            
            return processed_results
            
        except Exception as e:
            print(f"❌ خطا در جستجوی فیلتری: {str(e)}")
            return []
    
    def delete_collection(self):
        """حذف collection"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"✅ Collection {self.collection_name} حذف شد")
        except Exception as e:
            print(f"❌ خطا در حذف collection: {str(e)}")
    
    def backup_database(self, backup_path: str):
        """پشتیبان‌گیری از دیتابیس"""
        try:
            import shutil
            shutil.copytree(self.db_path, backup_path)
            print(f"✅ پشتیبان‌گیری در {backup_path} ذخیره شد")
        except Exception as e:
            print(f"❌ خطا در پشتیبان‌گیری: {str(e)}")


def main():
    """تابع اصلی برای تست"""
    print("🔧 Legal AI Assistant - Vector Database Manager")
    print("=" * 50)
    
    # ایجاد دیتابیس
    db = LegalVectorDatabase()
    
    # نمایش آمار فعلی
    stats = db.get_statistics()
    print(f"📊 آمار دیتابیس:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # اگر دیتابیس خالی است، chunks را بارگذاری کن
    if stats.get('total_documents', 0) == 0:
        print(f"\n🔄 دیتابیس خالی است. بارگذاری chunks...")
        chunks_dir = "data/chunks"
        
        if os.path.exists(chunks_dir):
            success = db.add_chunks_from_files(chunks_dir)
            if success:
                print(f"✅ chunks با موفقیت بارگذاری شدند")
                
                # نمایش آمار جدید
                new_stats = db.get_statistics()
                print(f"\n📊 آمار جدید:")
                for key, value in new_stats.items():
                    print(f"   {key}: {value}")
            else:
                print(f"❌ خطا در بارگذاری chunks")
        else:
            print(f"❌ فولدر chunks یافت نشد: {chunks_dir}")
            print(f"ابتدا فاز 1 را اجرا کنید: python run_phase1.py")
    
    # تست جستجوی نمونه
    if stats.get('total_documents', 0) > 0:
        print(f"\n🔍 تست جستجوی نمونه...")
        test_query = "قانون مقررات انتظامی هیئت علمی"
        results = db.search_semantic(test_query, n_results=3)
        
        print(f"نتایج جستجو برای '{test_query}':")
        for i, result in enumerate(results, 1):
            print(f"{i}. امتیاز: {result['similarity_score']:.3f}")
            print(f"   متن: {result['text'][:100]}...")
            print(f"   نوع: {result['metadata'].get('document_type', 'نامشخص')}")
            print()


if __name__ == "__main__":
    main()