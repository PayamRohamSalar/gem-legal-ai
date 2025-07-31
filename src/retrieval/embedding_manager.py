# src/retrieval/embedding_manager.py

import os
import json
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
from tqdm import tqdm

class EmbeddingModelManager:
    """
    مدیریت مدل‌های embedding برای متون حقوقی فارسی
    
    این کلاس مسئول انتخاب، تست و مدیریت مدل‌های مختلف embedding است
    """
    
    def __init__(self, cache_dir: str = "models/embeddings"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # مدل‌های پیشنهادی برای فارسی
        self.supported_models = {
            # مدل‌های چندزبانه (پیشنهاد اول)
            'multilingual-mpnet': {
                'model_name': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                'description': 'مدل چندزبانه قدرتمند - بهترین برای فارسی',
                'dimension': 768,
                'max_length': 512,
                'languages': ['fa', 'en', 'ar', 'many others'],
                'recommended': True
            },
            'multilingual-minilm': {
                'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'description': 'مدل چندزبانه سریع - متعادل سرعت و کیفیت',
                'dimension': 384,
                'max_length': 512,
                'languages': ['fa', 'en', 'ar', 'many others'],
                'recommended': True
            },
            
            # مدل‌های فارسی تخصصی
            'bert-fa-base': {
                'model_name': 'HooshvareLab/bert-fa-base-uncased',
                'description': 'BERT فارسی - مناسب برای متون تخصصی',
                'dimension': 768,
                'max_length': 512,
                'languages': ['fa'],
                'recommended': False  # نیاز به fine-tuning
            },
            
            # مدل‌های سبک
            'all-minilm': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'description': 'مدل سبک انگلیسی - برای تست سرعت',
                'dimension': 384,
                'max_length': 512,
                'languages': ['en'],
                'recommended': False  # فقط برای انگلیسی
            }
        }
        
        self.current_model = None
        self.current_model_name = None
        
    def list_available_models(self) -> Dict:
        """لیست مدل‌های قابل استفاده"""
        return self.supported_models
    
    def load_model(self, model_key: str, force_reload: bool = False) -> bool:
        """
        بارگذاری مدل embedding
        
        Args:
            model_key: کلید مدل از لیست supported_models
            force_reload: بارگذاری مجدد حتی اگر قبلاً بارگذاری شده
            
        Returns:
            bool: موفقیت در بارگذاری
        """
        if model_key not in self.supported_models:
            print(f"❌ مدل '{model_key}' پشتیبانی نمی‌شود!")
            print(f"مدل‌های موجود: {list(self.supported_models.keys())}")
            return False
        
        # اگر همین مدل قبلاً بارگذاری شده
        if (not force_reload and 
            self.current_model_name == model_key and 
            self.current_model is not None):
            print(f"✅ مدل '{model_key}' قبلاً بارگذاری شده است")
            return True
        
        model_info = self.supported_models[model_key]
        model_name = model_info['model_name']
        
        print(f"🔄 در حال بارگذاری مدل: {model_key}")
        print(f"   📝 توضیحات: {model_info['description']}")
        print(f"   🔢 بعد بردار: {model_info['dimension']}")
        
        try:
            start_time = time.time()
            
            # بارگذاری مدل
            self.current_model = SentenceTransformer(
                model_name,
                cache_folder=self.cache_dir
            )
            
            # تنظیم دستگاه (GPU اگر موجود باشد)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.current_model.to(device)
            
            loading_time = time.time() - start_time
            self.current_model_name = model_key
            
            print(f"✅ مدل بارگذاری شد در {loading_time:.1f} ثانیه")
            print(f"   🖥️ دستگاه: {device}")
            print(f"   💾 حافظه GPU: {self._get_gpu_memory()}")
            
            return True
            
        except Exception as e:
            print(f"❌ خطا در بارگذاری مدل: {str(e)}")
            self.current_model = None
            self.current_model_name = None
            return False
    
    def _get_gpu_memory(self) -> str:
        """دریافت اطلاعات حافظه GPU"""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_total = torch.cuda.memory_reserved() / 1024**3  # GB
            return f"{memory_used:.1f}/{memory_total:.1f} GB"
        return "CPU mode"
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """
        تبدیل متن به embedding
        
        Args:
            text: متن ورودی
            
        Returns:
            numpy array: بردار embedding یا None در صورت خطا
        """
        if self.current_model is None:
            print("❌ هیچ مدلی بارگذاری نشده! ابتدا load_model() را اجرا کنید.")
            return None
        
        try:
            # پاک‌سازی متن
            text = text.strip()
            if not text:
                return np.zeros(self.get_embedding_dimension())
            
            # تولید embedding
            embedding = self.current_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True  # نرمال‌سازی برای cosine similarity
            )
            
            return embedding
            
        except Exception as e:
            print(f"❌ خطا در encoding: {str(e)}")
            return None
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[np.ndarray]:
        """
        تبدیل مجموعه‌ای از متون به embedding
        
        Args:
            texts: لیست متون
            batch_size: اندازه batch برای پردازش
            show_progress: نمایش progress bar
            
        Returns:
            list: لیست embeddings
        """
        if self.current_model is None:
            print("❌ هیچ مدلی بارگذاری نشده!")
            return []
        
        try:
            # پاک‌سازی متون
            clean_texts = [text.strip() if text else "" for text in texts]
            
            print(f"🔄 تولید embedding برای {len(texts)} متن...")
            start_time = time.time()
            
            # تولید embeddings با batch processing
            embeddings = self.current_model.encode(
                clean_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress
            )
            
            encoding_time = time.time() - start_time
            speed = len(texts) / encoding_time
            
            print(f"✅ {len(embeddings)} embedding تولید شد در {encoding_time:.1f} ثانیه")
            print(f"   ⚡ سرعت: {speed:.1f} متن/ثانیه")
            
            return embeddings.tolist()
            
        except Exception as e:
            print(f"❌ خطا در batch encoding: {str(e)}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """دریافت بعد embedding مدل فعلی"""
        if self.current_model_name:
            return self.supported_models[self.current_model_name]['dimension']
        return 0
    
    def test_model_quality(self, test_texts: List[str] = None) -> Dict:
        """
        تست کیفیت مدل با متون نمونه
        
        Args:
            test_texts: متون تست (اختیاری)
            
        Returns:
            dict: نتایج تست
        """
        if self.current_model is None:
            return {"error": "مدل بارگذاری نشده"}
        
        # متون تست پیش‌فرض (حقوقی فارسی)
        if test_texts is None:
            test_texts = [
                "قانون مقررات انتظامی هیئت علمی دانشگاه‌ها",
                "آیین‌نامه اجرایی قانون انتظامات هیئت علمی",
                "مجازات‌های انتظامی اعضای هیئت علمی",
                "تخلفات انضباطی هیئت علمی دانشگاه",
                "وزارت علوم تحقیقات و فناوری",
                "پژوهش و فناوری در دانشگاه‌های کشور",
                "ماده یک قانون اساسی جمهوری اسلامی",
                "حقوق مالی کارکنان دولت"
            ]
        
        print(f"🧪 تست کیفیت مدل با {len(test_texts)} متن...")
        
        try:
            start_time = time.time()
            
            # تولید embeddings
            embeddings = self.encode_batch(test_texts, show_progress=False)
            
            if not embeddings:
                return {"error": "خطا در تولید embedding"}
            
            # محاسبه similarity matrix
            embeddings_np = np.array(embeddings)
            similarity_matrix = np.dot(embeddings_np, embeddings_np.T)
            
            # آنالیز کیفیت
            avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
            max_similarity = np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
            min_similarity = np.min(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
            
            # تست سرعت
            test_time = time.time() - start_time
            speed = len(test_texts) / test_time
            
            # نتایج
            results = {
                "model_name": self.current_model_name,
                "test_texts_count": len(test_texts),
                "embedding_dimension": len(embeddings[0]),
                "quality_metrics": {
                    "average_similarity": float(avg_similarity),
                    "max_similarity": float(max_similarity),
                    "min_similarity": float(min_similarity),
                    "similarity_range": float(max_similarity - min_similarity)
                },
                "performance_metrics": {
                    "encoding_time": test_time,
                    "speed_texts_per_second": speed,
                    "avg_time_per_text": test_time / len(test_texts)
                },
                "device_info": {
                    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
                    "gpu_memory": self._get_gpu_memory()
                }
            }
            
            # نمایش نتایج
            print(f"✅ تست کامل شد در {test_time:.2f} ثانیه")
            print(f"📊 نتایج کیفیت:")
            print(f"   • میانگین شباهت: {avg_similarity:.3f}")
            print(f"   • دامنه شباهت: {min_similarity:.3f} - {max_similarity:.3f}")
            print(f"   • سرعت: {speed:.1f} متن/ثانیه")
            
            return results
            
        except Exception as e:
            return {"error": f"خطا در تست: {str(e)}"}
    
    def compare_models(self, model_keys: List[str], test_texts: List[str] = None) -> Dict:
        """
        مقایسه چندین مدل embedding
        
        Args:
            model_keys: لیست کلیدهای مدل‌ها
            test_texts: متون تست
            
        Returns:
            dict: نتایج مقایسه
        """
        print(f"🔬 مقایسه {len(model_keys)} مدل embedding...")
        
        comparison_results = {}
        
        for model_key in model_keys:
            print(f"\n🔄 تست مدل: {model_key}")
            
            if self.load_model(model_key):
                results = self.test_model_quality(test_texts)
                comparison_results[model_key] = results
            else:
                comparison_results[model_key] = {"error": "مدل بارگذاری نشد"}
        
        # خلاصه مقایسه
        print(f"\n📊 خلاصه مقایسه:")
        print("-" * 50)
        
        for model_key, results in comparison_results.items():
            if "error" not in results:
                quality = results["quality_metrics"]["average_similarity"]
                speed = results["performance_metrics"]["speed_texts_per_second"]
                print(f"{model_key:20} | کیفیت: {quality:.3f} | سرعت: {speed:.1f}")
            else:
                print(f"{model_key:20} | ❌ {results['error']}")
        
        return comparison_results
    
    def save_model_benchmark(self, results: Dict, output_file: str = None):
        """ذخیره نتایج benchmark"""
        if output_file is None:
            output_file = f"data/metadata/embedding_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 نتایج benchmark ذخیره شد: {output_file}")
    
    def get_recommended_model(self) -> str:
        """دریافت مدل پیشنهادی"""
        for key, info in self.supported_models.items():
            if info.get('recommended', False):
                return key
        return list(self.supported_models.keys())[0]


def main():
    """تابع اصلی برای تست"""
    print("🔧 Legal AI Assistant - Embedding Model Manager")
    print("=" * 50)
    
    # ایجاد مدیر embedding
    embedding_manager = EmbeddingModelManager()
    
    # نمایش مدل‌های موجود
    print("📋 مدل‌های موجود:")
    for key, info in embedding_manager.list_available_models().items():
        status = "⭐ پیشنهادی" if info.get('recommended') else ""
        print(f"  {key}: {info['description']} {status}")
    
    # بارگذاری مدل پیشنهادی
    recommended_model = embedding_manager.get_recommended_model()
    print(f"\n🚀 بارگذاری مدل پیشنهادی: {recommended_model}")
    
    if embedding_manager.load_model(recommended_model):
        # تست کیفیت
        test_results = embedding_manager.test_model_quality()
        
        # ذخیره نتایج
        embedding_manager.save_model_benchmark({
            "single_model_test": test_results,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"\n✅ تست مدل {recommended_model} تکمیل شد!")
    else:
        print(f"\n❌ خطا در بارگذاری مدل {recommended_model}")


if __name__ == "__main__":
    main()