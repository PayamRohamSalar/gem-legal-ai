"""
src/generation/llm_manager.py - مدیریت مدل‌های LLM (نسخه بهبود یافته با Ollama)

این فایل مسئول بارگذاری، مدیریت و استفاده از مدل‌های زبان بزرگ است.
ویژگی جدید: اتصال به Ollama برای مدل‌های واقعی
"""

import torch
import psutil
import time
import gc
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama در دسترس نیست. از مدل Mock استفاده می‌شود.")

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """انواع مدل‌های پشتیبانی‌شده"""
    QWEN_7B = "qwen2.5:7b"
    LLAMA_8B = "llama3.1:8b"
    MISTRAL_7B = "mistral:7b"
    MOCK = "mock_model"

@dataclass
class GenerationMetrics:
    """متریک‌های تولید پاسخ"""
    generation_time: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    memory_usage_mb: float
    model_name: str
    temperature: float
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class LLMConfig:
    """تنظیمات مدل LLM"""
    model_name: str
    model_type: ModelType
    device: str = "auto"
    max_new_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    context_length: int = 8192

# در فایل src/generation/llm_manager.py

class OllamaManager:
    """مدیریت اتصال به Ollama (نسخه قوی‌تر)"""

    def __init__(self, host: str = 'http://localhost:11434'):
        self.host = host
        self.client = None
        self.available_models = []
        self.connection_status = False

        if OLLAMA_AVAILABLE:
            self.client = ollama.Client(host=self.host)
            self._check_connection()
        else:
            logger.warning("Ollama package در دسترس نیست.")

    def _check_connection(self) -> bool:
        if not self.client:
            return False

        try:
            logger.info(f"در حال بررسی اتصال به سرور Ollama در {self.host} ...")
            response = self.client.list()
            self.available_models = [model.get('name') for model in response.get('models', []) if model.get('name')]
            self.connection_status = True
            logger.info(f"اتصال به Ollama برقرار شد. مدل‌های موجود: {self.available_models}")
            if not self.available_models:
                logger.warning("Ollama متصل است اما هیچ مدلی را گزارش نمی‌دهد. لطفاً سرور Ollama را ری‌استارت کنید.")
            return True
        except Exception as e:
            logger.error(f"خطا در اتصال به Ollama: {e}. لطفاً از روشن بودن سرور Ollama مطمئن شوید.")
            self.connection_status = False
            return False

    
    def generate(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """تولید پاسخ با Ollama"""
        if not self.connection_status:
            raise Exception("اتصال به Ollama برقرار نیست")
        
        try:
            start_time = time.time()
            
            # تنظیمات پیش‌فرض
            options = {
                'temperature': kwargs.get('temperature', 0.1),
                'top_p': kwargs.get('top_p', 0.9),
                'top_k': kwargs.get('top_k', 40),
                'repeat_penalty': kwargs.get('repeat_penalty', 1.1),
                'num_predict': kwargs.get('max_new_tokens', 2048)
            }
            
            # ارسال درخواست
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options=options
            )
            
            generation_time = time.time() - start_time
            
            return {
                'success': True,
                'response': response['response'],
                'generation_time': generation_time,
                'model': model_name,
                'total_duration': response.get('total_duration', 0),
                'load_duration': response.get('load_duration', 0),
                'prompt_eval_count': response.get('prompt_eval_count', 0),
                'eval_count': response.get('eval_count', 0)
            }
            
        except Exception as e:
            logger.error(f"خطا در تولید با Ollama: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': '',
                'generation_time': time.time() - start_time
            }

class LLMManager:
    """مدیریت کننده اصلی مدل‌های LLM"""
    
    def __init__(self, configs: Dict[str, LLMConfig]):
        self.configs = configs
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.active_model: Optional[str] = None
        self.generation_history: List[GenerationMetrics] = []
        
        # مدیریت Ollama
        self.ollama_manager = OllamaManager()
        
        logger.info("LLMManager ایجاد شد")
    
    def _get_memory_usage(self) -> float:
        """محاسبه میزان استفاده از حافظه (MB)"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _estimate_tokens(self, text: str) -> int:
        """تخمین تعداد توکن‌ها (تقریبی)"""
        # تخمین ساده: فارسی حدود 4 کاراکتر به ازای هر توکن
        return len(text) // 4
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """لیست مدل‌های موجود"""
        return {
            'ollama_models': self.ollama_manager.available_models,
            'loaded_models': list(self.models.keys()),
            'configured_models': list(self.configs.keys()),
            'ollama_status': self.ollama_manager.connection_status
        }
    
    def load_model_ollama(self, model_key: str) -> bool:
        """بارگذاری مدل از Ollama"""
        if model_key not in self.configs:
            logger.error(f"تنظیمات مدل {model_key} یافت نشد")
            return False
        
        config = self.configs[model_key]
        
        # بررسی اتصال Ollama
        if not self.ollama_manager.connection_status:
            logger.error("اتصال به Ollama برقرار نیست")
            return False
        
        try:
            # بررسی وجود مدل
            if config.model_name not in self.ollama_manager.available_models:
                logger.info(f"مدل {config.model_name} موجود نیست. شروع دانلود...")
                if not self.ollama_manager.pull_model(config.model_name):
                    logger.error(f"خطا در دانلود مدل {config.model_name}")
                    return False
            
            # تست مدل
            test_result = self.ollama_manager.generate(
                config.model_name, 
                "سلام",
                temperature=0.1,
                max_new_tokens=10
            )
            
            if test_result['success']:
                self.models[model_key] = {'type': 'ollama', 'name': config.model_name}
                self.active_model = model_key
                logger.info(f"مدل Ollama {model_key} بارگذاری شد")
                return True
            else:
                logger.error(f"تست مدل {config.model_name} ناموفق: {test_result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل Ollama {model_key}: {e}")
            return False
    
    def load_model_mock(self, model_key: str) -> bool:
        """بارگذاری mock مدل برای تست (fallback)"""
        if model_key not in self.configs:
            logger.error(f"تنظیمات مدل {model_key} یافت نشد")
            return False
        
        try:
            logger.warning(f"بارگذاری mock مدل: {model_key} (Ollama در دسترس نیست)")
            
            # Mock objects برای تست
            mock_model = type('MockModel', (), {
                'generate': lambda *args, **kwargs: type('MockOutput', (), {
                    'sequences': [torch.tensor([[1, 2, 3, 4, 5]])],
                    'scores': None
                })(),
                'device': 'cpu'
            })()
            
            mock_tokenizer = type('MockTokenizer', (), {
                'encode': lambda self, text: list(range(min(len(text)//4, 100))),
                'decode': lambda self, tokens, **kwargs: "این یک پاسخ تست Mock است. در نسخه نهایی، Ollama پاسخ واقعی ارائه خواهد داد.",
                'eos_token': '</s>',
                'eos_token_id': 2,
                'pad_token': '</s>',
                '__call__': lambda self, text, **kwargs: {
                    'input_ids': torch.tensor([[1, 2, 3]]),
                    'attention_mask': torch.tensor([[1, 1, 1]])
                }
            })()
            
            self.models[model_key] = {'type': 'mock', 'model': mock_model}
            self.tokenizers[model_key] = mock_tokenizer
            self.active_model = model_key
            
            logger.info(f"Mock مدل {model_key} بارگذاری شد")
            return True
            
        except Exception as e:
            logger.error(f"خطا در بارگذاری mock مدل {model_key}: {e}")
            return False
    
    def load_model(self, model_key: str, prefer_ollama: bool = True) -> bool:
        """بارگذاری مدل (اولویت با Ollama)"""
        if prefer_ollama and OLLAMA_AVAILABLE:
            success = self.load_model_ollama(model_key)
            if success:
                return True
            logger.warning("بارگذاری Ollama ناموفق. استفاده از Mock...")
        
        return self.load_model_mock(model_key)
    
    def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """تولید پاسخ با مدل فعال"""
        if not self.active_model:
            return {
                'success': False,
                'error': 'هیچ مدلی فعال نیست',
                'response': '',
                'metrics': None
            }
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            config = self.configs[self.active_model]
            model_info = self.models[self.active_model]
            
            # تخمین توکن‌های ورودی
            input_tokens = self._estimate_tokens(prompt)
            
            # تولید بر اساس نوع مدل
            if model_info['type'] == 'ollama':
                # استفاده از Ollama
                result = self.ollama_manager.generate(
                    model_info['name'],
                    prompt,
                    temperature=kwargs.get('temperature', config.temperature),
                    top_p=kwargs.get('top_p', config.top_p),
                    top_k=kwargs.get('top_k', config.top_k),
                    max_new_tokens=kwargs.get('max_new_tokens', config.max_new_tokens),
                    repeat_penalty=kwargs.get('repeat_penalty', config.repeat_penalty)
                )
                
                if result['success']:
                    response = result['response']
                    output_tokens = self._estimate_tokens(response)
                    
                    # استفاده از آمار دقیق Ollama در صورت وجود
                    if 'eval_count' in result:
                        output_tokens = result['eval_count']
                    if 'prompt_eval_count' in result:
                        input_tokens = result['prompt_eval_count']
                else:
                    raise Exception(result['error'])
                    
            else:
                # استفاده از Mock
                tokenizer = self.tokenizers[self.active_model]
                time.sleep(0.3)  # شبیه‌سازی زمان پردازش
                response = tokenizer.decode([], skip_special_tokens=True)
                output_tokens = self._estimate_tokens(response)
            
            # محاسبه متریک‌ها
            generation_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            metrics = GenerationMetrics(
                generation_time=generation_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                memory_usage_mb=memory_usage,
                model_name=self.active_model,
                temperature=kwargs.get('temperature', config.temperature),
                success=True
            )
            
            self.generation_history.append(metrics)
            
            return {
                'success': True,
                'response': response,
                'metrics': metrics,
                'model_used': self.active_model,
                'model_type': model_info['type']
            }
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_metrics = GenerationMetrics(
                generation_time=generation_time,
                input_tokens=input_tokens if 'input_tokens' in locals() else 0,
                output_tokens=0,
                total_tokens=input_tokens if 'input_tokens' in locals() else 0,
                memory_usage_mb=self._get_memory_usage() - start_memory,
                model_name=self.active_model,
                temperature=kwargs.get('temperature', 0.1),
                success=False,
                error_message=str(e)
            )
            
            self.generation_history.append(error_metrics)
            
            logger.error(f"خطا در تولید پاسخ: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': '',
                'metrics': error_metrics
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """اطلاعات کامل مدل‌ها"""
        info = {
            'loaded_models': list(self.models.keys()),
            'active_model': self.active_model,
            'total_generations': len(self.generation_history),
            'memory_usage_mb': self._get_memory_usage(),
            'ollama_status': self.ollama_manager.connection_status,
            'ollama_models': self.ollama_manager.available_models
        }
        
        # آمار عملکرد
        if self.generation_history:
            successful_gens = [m for m in self.generation_history if m.success]
            if successful_gens:
                info['performance'] = {
                    'success_rate': len(successful_gens) / len(self.generation_history),
                    'avg_generation_time': sum(m.generation_time for m in successful_gens) / len(successful_gens),
                    'avg_output_tokens': sum(m.output_tokens for m in successful_gens) / len(successful_gens)
                }
        
        return info
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """آمار تفصیلی تولیدات"""
        if not self.generation_history:
            return {'message': 'هیچ تولیدی انجام نشده'}
        
        successful = [m for m in self.generation_history if m.success]
        failed = [m for m in self.generation_history if not m.success]
        
        stats = {
            'total_generations': len(self.generation_history),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(self.generation_history) * 100,
            'timestamp': datetime.now().isoformat()
        }
        
        if successful:
            stats['performance'] = {
                'avg_time': sum(m.generation_time for m in successful) / len(successful),
                'min_time': min(m.generation_time for m in successful),
                'max_time': max(m.generation_time for m in successful),
                'avg_input_tokens': sum(m.input_tokens for m in successful) / len(successful),
                'avg_output_tokens': sum(m.output_tokens for m in successful) / len(successful),
                'total_tokens_processed': sum(m.total_tokens for m in successful)
            }
        
        return stats
    
    def cleanup(self) -> None:
        """پاک‌سازی منابع"""
        for model_key in list(self.models.keys()):
            del self.models[model_key]
        
        if hasattr(self, 'tokenizers'):
            for tokenizer_key in list(self.tokenizers.keys()):
                del self.tokenizers[tokenizer_key]
            self.tokenizers.clear()
        
        self.models.clear()
        self.active_model = None
        self.generation_history.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("منابع LLM پاک‌سازی شدند")

def create_model_configs() -> Dict[str, LLMConfig]:
    """ایجاد تنظیمات پیش‌فرض مدل‌ها"""
    return {
        'qwen_7b': LLMConfig(
            model_name="qwen2.5:7b",
            model_type=ModelType.QWEN_7B,
            temperature=0.1,
            max_new_tokens=2048,
            top_p=0.9,
            top_k=40
        ),
        'llama_8b': LLMConfig(
            model_name="llama3.1:8b",
            model_type=ModelType.LLAMA_8B,
            temperature=0.1,
            max_new_tokens=2048,
            top_p=0.9,
            top_k=40
        ),
        'mistral_7b': LLMConfig(
            model_name="mistral:7b",
            model_type=ModelType.MISTRAL_7B,
            temperature=0.1,
            max_new_tokens=2048,
            top_p=0.9,
            top_k=40
        )
    }

# تست جامع
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🧪 تست جامع LLMManager با Ollama")
    print("=" * 60)
    
    # ایجاد مدیریت
    configs = create_model_configs()
    manager = LLMManager(configs)
    
    # نمایش مدل‌های موجود
    available = manager.list_available_models()
    print(f"📋 مدل‌های موجود:")
    print(f"   • Ollama: {available['ollama_models']}")
    print(f"   • وضعیت Ollama: {available['ollama_status']}")
    
    # تست بارگذاری
    print(f"\n🔄 تست بارگذاری مدل...")
    model_to_test = 'qwen_7b'
    
    success = manager.load_model(model_to_test, prefer_ollama=True)
    print(f"✅ بارگذاری {'موفق' if success else 'ناموفق'}")
    
    if success:
        # تست تولید
        print(f"\n🤖 تست تولید پاسخ...")
        test_prompts = [
            "سلام، چطوری؟",
            "وظایف اعضای هیئت علمی در پژوهش چیست؟",
            "تعریف شرکت دانش‌بنیان را بیان کنید."
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 تست {i}: {prompt}")
            
            result = manager.generate_response(prompt, temperature=0.1)
            
            if result['success']:
                response = result['response']
                metrics = result['metrics']
                
                print(f"   ✅ موفق - طول پاسخ: {len(response)} کاراکتر")
                print(f"   ⏱️  زمان: {metrics.generation_time:.2f}s")
                print(f"   🔢 توکن‌ها: {metrics.input_tokens}→{metrics.output_tokens}")
                print(f"   💧 نوع مدل: {result['model_type']}")
                print(f"   📄 پاسخ: {response[:100]}...")
            else:
                print(f"   ❌ ناموفق: {result['error']}")
    
    # آمار نهایی
    print(f"\n📊 آمار نهایی:")
    stats = manager.get_generation_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   • {key}:")
            for k, v in value.items():
                print(f"     - {k}: {v}")
        else:
            print(f"   • {key}: {value}")
    
    # پاک‌سازی
    manager.cleanup()
    print("\n✅ تست کامل شد")