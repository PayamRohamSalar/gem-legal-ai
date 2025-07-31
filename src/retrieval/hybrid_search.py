# src/retrieval/hybrid_search.py

import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
from datetime import datetime

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hazm

# اگر relative import کار نکرد، از absolute import استفاده کنید
try:
    from .vector_database import LegalVectorDatabase
    from .embedding_manager import EmbeddingModelManager
except ImportError:
    from src.retrieval.vector_database import LegalVectorDatabase
    from src.retrieval.embedding_manager import EmbeddingModelManager

class PersianTextProcessor:
    """پردازش‌گر متن فارسی برای جستجو"""
    
    def __init__(self):
        self.normalizer = hazm.Normalizer()
        self.stemmer = hazm.Stemmer()
        self.lemmatizer = hazm.Lemmatizer()
        
        # کلمات stop فارسی
        self.stop_words = {
            'و', 'در', 'به', 'از', 'که', 'این', 'آن', 'با', 'برای', 'تا', 'بر', 'را',
            'است', 'بود', 'می', 'خواهد', 'باید', 'شده', 'شود', 'کرد', 'نمود', 'گرفت',
            'های', 'ها', 'ان', 'یا', 'اگر', 'چون', 'چرا', 'کجا', 'کی', 'چه', 'هر',
            'همه', 'همچنین', 'ولی', 'اما', 'یعنی', 'مثل', 'مانند', 'نیز', 'هم'
        }
        
        # کلمات مهم حقوقی که نباید حذف شوند
        self.legal_keywords = {
            'قانون', 'ماده', 'بند', 'تبصره', 'فصل', 'مقرر', 'موضوع', 'مصوب',
            'مجلس', 'هیئت', 'وزیران', 'شورای', 'عالی', 'وزارت', 'مؤسسه',
            'دانشگاه', 'پژوهش', 'فناوری', 'تحقیقات', 'علوم', 'آموزش',
            'انتظامی', 'تخلف', 'مجازات', 'تعهد', 'مسئولیت', 'حق', 'وظیفه'
        }
    
    def process_text(self, text: str, for_search: bool = True) -> List[str]:
        """
        پردازش متن فارسی برای جستجو
        
        Args:
            text: متن ورودی
            for_search: آیا برای جستجو است یا نه
            
        Returns:
            list: لیست کلمات پردازش شده
        """
        if not text:
            return []
        
        # نرمال‌سازی
        text = self.normalizer.normalize(text)
        
        # حذف علائم نگارشی (جز نقطه و کاما برای حفظ ساختار)
        text = re.sub(r'[^\w\s\.\,]', ' ', text)
        
        # تقسیم به کلمات
        words = text.split()
        
        processed_words = []
        
        for word in words:
            word = word.strip()
            if not word:
                continue
            
            # حذف اعداد خالص (مگر اینکه جزء عبارت حقوقی باشند)
            if word.isdigit() and len(word) > 4:
                continue
            
            # کلمات خیلی کوتاه
            if len(word) < 2:
                continue
            
            # حذف stop words (مگر کلمات حقوقی مهم)
            if for_search and word in self.stop_words and word not in self.legal_keywords:
                continue
            
            # stemming برای جستجو
            if for_search and word not in self.legal_keywords:
                try:
                    word = self.stemmer.stem(word)
                except:
                    pass  # اگر stemming ناموفق بود، کلمه اصلی را نگه دار
            
            processed_words.append(word)
        
        return processed_words
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """استخراج عبارات کلیدی از متن"""
        phrases = []
        
        # پترن‌های حقوقی
        legal_patterns = [
            r'ماده\s+\d+',
            r'بند\s+[الف-ی]',
            r'تبصره\s+\d*',
            r'فصل\s+\d+',
            r'قانون\s+[\w\s]{5,50}',
            r'آیین‌نامه\s+[\w\s]{5,50}',
            r'مصوب\s+\d{2,4}/\d{1,2}/\d{1,2}'
        ]
        
        for pattern in legal_patterns:
            matches = re.findall(pattern, text)
            phrases.extend(matches)
        
        return phrases


class HybridSearchEngine:
    """
    موتور جستجوی ترکیبی (Semantic + Keyword)
    
    این کلاس جستجوی معنایی و کلیدواژه‌ای را ترکیب می‌کند
    """
    
    def __init__(self, 
                 vector_db: LegalVectorDatabase,
                 embedding_manager: EmbeddingModelManager):
        
        self.vector_db = vector_db
        self.embedding_manager = embedding_manager
        self.text_processor = PersianTextProcessor()
        
        # BM25 index
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_metadata = []
        
        # TF-IDF vectorizer
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        self._build_keyword_indices()
    
    
    # در فایل src/retrieval/hybrid_search.py

    def _build_keyword_indices(self):
        """ساخت indices برای جستجوی کلیدواژه‌ای"""
        print("🔄 ساخت indices برای جستجوی کلیدواژه‌ای...")
        
        try:
            # دریافت تمام documents از vector database - نسخه اصلاح شده
            # به جای استفاده از فیلتر، مستقیماً از collection.get استفاده می‌کنیم
            if self.vector_db.collection.count() == 0:
                print("❌ هیچ document در vector database یافت نشد!")
                return
            
            # دریافت تمام اسناد بدون فیلتر
            all_results_raw = self.vector_db.collection.get(
                limit=self.vector_db.collection.count(), 
                include=['documents', 'metadatas']
            )

            print(f"📊 در حال پردازش {len(all_results_raw['ids'])} document...")
            
            # بقیه کد بدون تغییر باقی می‌ماند...
            processed_texts = []
            raw_texts = []
            metadatas = []

            for i in range(len(all_results_raw['ids'])):
                text = all_results_raw['documents'][i]
                metadata = all_results_raw['metadatas'][i]
                
                processed_words = self.text_processor.process_text(text, for_search=True)
                processed_texts.append(processed_words)
                
                raw_texts.append(text)
                metadatas.append(metadata)

            # بقیه متد نیز بدون تغییر است
            if processed_texts:
                self.bm25_index = BM25Okapi(processed_texts)
                self.bm25_documents = raw_texts
                self.bm25_metadata = metadatas
                print(f"✅ BM25 index ساخته شد با {len(processed_texts)} document")
            
            if raw_texts:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words=list(self.text_processor.stop_words),
                    ngram_range=(1, 2),
                    min_df=2
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(raw_texts)
                print(f"✅ TF-IDF index ساخته شد با {self.tfidf_matrix.shape[1]} feature")
                
        except Exception as e:
            print(f"❌ خطا در ساخت keyword indices: {str(e)}")
    
    
    def search_bm25(self, query: str, top_k: int = 20) -> List[Dict]:
        """جستجوی BM25"""
        if not self.bm25_index:
            return []
        
        try:
            # پردازش query
            processed_query = self.text_processor.process_text(query, for_search=True)
            
            if not processed_query:
                return []
            
            # محاسبه امتیازات BM25
            scores = self.bm25_index.get_scores(processed_query)
            
            # مرتب‌سازی و انتخاب top results
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for rank, idx in enumerate(top_indices):
                if scores[idx] > 0:  # فقط نتایج با امتیاز مثبت
                    results.append({
                        'text': self.bm25_documents[idx],
                        'metadata': self.bm25_metadata[idx],
                        'bm25_score': float(scores[idx]),
                        'rank': rank + 1,
                        'search_type': 'bm25'
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ خطا در جستجوی BM25: {str(e)}")
            return []
    
    def search_tfidf(self, query: str, top_k: int = 20) -> List[Dict]:
        """جستجوی TF-IDF"""
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            return []
        
        try:
            # تبدیل query به بردار TF-IDF
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # محاسبه شباهت cosine
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # مرتب‌سازی و انتخاب top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for rank, idx in enumerate(top_indices):
                if similarities[idx] > 0.01:  # threshold کمینه
                    results.append({
                        'text': self.bm25_documents[idx],
                        'metadata': self.bm25_metadata[idx],
                        'tfidf_score': float(similarities[idx]),
                        'rank': rank + 1,
                        'search_type': 'tfidf'
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ خطا در جستجوی TF-IDF: {str(e)}")
            return []
    
    def search_semantic(self, query: str, top_k: int = 20, filters: Dict = None) -> List[Dict]:
        """جستجوی معنایی"""
        try:
            results = self.vector_db.search_semantic(query, n_results=top_k, filters=filters)
            
            # اضافه کردن نوع جستجو
            for result in results:
                result['search_type'] = 'semantic'
            
            return results
            
        except Exception as e:
            print(f"❌ خطا در جستجوی معنایی: {str(e)}")
            return []
    
    def hybrid_search(self, 
                     query: str,
                     top_k: int = 10,
                     semantic_weight: float = 0.6,
                     bm25_weight: float = 0.3,
                     tfidf_weight: float = 0.1,
                     filters: Dict = None) -> List[Dict]:
        """
        جستجوی ترکیبی (هیبرید)
        
        Args:
            query: متن جستجو
            top_k: تعداد نتایج نهایی
            semantic_weight: وزن جستجوی معنایی
            bm25_weight: وزن BM25
            tfidf_weight: وزن TF-IDF
            filters: فیلترهای metadata
            
        Returns:
            list: نتایج ترکیب شده و رتبه‌بندی شده
        """
        print(f"🔍 جستجوی هیبرید برای: '{query[:50]}...'")
        
        # انجام جستجوهای مختلف
        semantic_results = self.search_semantic(query, top_k=top_k*2, filters=filters)
        bm25_results = self.search_bm25(query, top_k=top_k*2)
        tfidf_results = self.search_tfidf(query, top_k=top_k*2)
        
        print(f"📊 نتایج خام: {len(semantic_results)} semantic, {len(bm25_results)} BM25, {len(tfidf_results)} TF-IDF")
        
        # ترکیب نتایج
        combined_results = self._combine_results(
            semantic_results, bm25_results, tfidf_results,
            semantic_weight, bm25_weight, tfidf_weight
        )
        
        # Re-ranking و فیلتر کردن
        final_results = self._rerank_results(combined_results, query)[:top_k]
        
        print(f"✅ {len(final_results)} نتیجه نهایی")
        
        return final_results
    
    def _combine_results(self, 
                        semantic_results: List[Dict],
                        bm25_results: List[Dict], 
                        tfidf_results: List[Dict],
                        semantic_weight: float,
                        bm25_weight: float,
                        tfidf_weight: float) -> List[Dict]:
        """ترکیب نتایج از منابع مختلف"""
        
        # ایجاد dictionary برای ترکیب امتیازات
        combined_scores = defaultdict(lambda: {
            'text': '', 'metadata': {}, 'scores': {}, 'total_score': 0.0
        })
        
        # نرمال‌سازی و اضافه کردن امتیازات semantic
        if semantic_results:
            max_semantic = max(r['similarity_score'] for r in semantic_results)
            for result in semantic_results:
                text = result['text']
                normalized_score = result['similarity_score'] / max_semantic if max_semantic > 0 else 0
                
                combined_scores[text]['text'] = text
                combined_scores[text]['metadata'] = result['metadata']
                combined_scores[text]['scores']['semantic'] = normalized_score
                combined_scores[text]['total_score'] += normalized_score * semantic_weight
        
        # اضافه کردن امتیازات BM25
        if bm25_results:
            max_bm25 = max(r['bm25_score'] for r in bm25_results)
            for result in bm25_results:
                text = result['text']
                normalized_score = result['bm25_score'] / max_bm25 if max_bm25 > 0 else 0
                
                if text not in combined_scores:
                    combined_scores[text]['text'] = text
                    combined_scores[text]['metadata'] = result['metadata']
                
                combined_scores[text]['scores']['bm25'] = normalized_score
                combined_scores[text]['total_score'] += normalized_score * bm25_weight
        
        # اضافه کردن امتیازات TF-IDF
        if tfidf_results:
            max_tfidf = max(r['tfidf_score'] for r in tfidf_results)
            for result in tfidf_results:
                text = result['text']
                normalized_score = result['tfidf_score'] / max_tfidf if max_tfidf > 0 else 0
                
                if text not in combined_scores:
                    combined_scores[text]['text'] = text
                    combined_scores[text]['metadata'] = result['metadata']
                
                combined_scores[text]['scores']['tfidf'] = normalized_score
                combined_scores[text]['total_score'] += normalized_score * tfidf_weight
        
        # تبدیل به لیست و مرتب‌سازی
        final_results = []
        for text, data in combined_scores.items():
            final_results.append({
                'text': data['text'],
                'metadata': data['metadata'],
                'hybrid_score': data['total_score'],
                'component_scores': data['scores'],
                'search_type': 'hybrid'
            })
        
        # مرتب‌سازی بر اساس امتیاز کل
        final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return final_results
    
    def _rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """re-ranking نتایج بر اساس معیارهای اضافی"""
        
        # استخراج عبارات کلیدی از query
        query_phrases = self.text_processor.extract_key_phrases(query)
        query_words = set(self.text_processor.process_text(query, for_search=True))
        
        for i, result in enumerate(results):
            text = result['text']
            metadata = result['metadata']
            
            # امتیاز اضافی برای تطابق عبارات حقوقی
            phrase_bonus = 0.0
            for phrase in query_phrases:
                if phrase.lower() in text.lower():
                    phrase_bonus += 0.1
            
            # امتیاز اضافی برای کیفیت chunk
            quality_bonus = metadata.get('quality_score', 0) / 10.0 * 0.05
            
            # امتیاز اضافی برای تطابق نوع سند
            doc_type_bonus = 0.0
            if any(word in ['قانون', 'ماده', 'بند'] for word in query_words):
                if metadata.get('document_type') == 'قانون':
                    doc_type_bonus = 0.02
            
            # امتیاز کل جدید
            total_bonus = phrase_bonus + quality_bonus + doc_type_bonus
            result['final_score'] = result['hybrid_score'] + total_bonus
            result['rerank_bonus'] = total_bonus
            result['rank'] = i + 1
        
        # مرتب‌سازی مجدد
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # به‌روزرسانی rank
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return results
    
    def get_search_stats(self) -> Dict:
        """آمار سیستم جستجو"""
        stats = {
            'vector_db_stats': self.vector_db.get_statistics(),
            'bm25_ready': self.bm25_index is not None,
            'tfidf_ready': self.tfidf_vectorizer is not None,
            'total_indexed_documents': len(self.bm25_documents),
            'embedding_model': self.embedding_manager.current_model_name
        }
        
        if self.tfidf_vectorizer:
            stats['tfidf_features'] = self.tfidf_matrix.shape[1]
        
        return stats


def main():
    """تابع اصلی برای تست"""
    print("🔧 Legal AI Assistant - Hybrid Search Engine")
    print("=" * 50)
    
    # راه‌اندازی components
    print("🔄 راه‌اندازی اجزای سیستم...")
    
    # Vector Database
    vector_db = LegalVectorDatabase()
    
    # Embedding Manager
    embedding_manager = EmbeddingModelManager()
    recommended_model = embedding_manager.get_recommended_model()
    
    if not embedding_manager.load_model(recommended_model):
        print("❌ خطا در بارگذاری embedding model!")
        return
    
    # Hybrid Search Engine
    search_engine = HybridSearchEngine(vector_db, embedding_manager)
    
    # نمایش آمار
    stats = search_engine.get_search_stats()
    print(f"\n📊 آمار سیستم جستجو:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # تست جستجوهای مختلف
    test_queries = [
        "قانون مقررات انتظامی هیئت علمی",
        "مجازات تخلفات انضباطی",
        "وزارت علوم تحقیقات و فناوری",
        "ماده 7 تخلفات"
    ]
    
    for query in test_queries:
        print(f"\n🔍 تست جستجو: '{query}'")
        
        # جستجوی هیبرید
        results = search_engine.hybrid_search(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. امتیاز: {result['final_score']:.3f}")
                print(f"   متن: {result['text'][:80]}...")
                print(f"   نوع: {result['metadata'].get('document_type', 'نامشخص')}")
        else:
            print("   نتیجه‌ای یافت نشد")


if __name__ == "__main__":
    main()