# src/generation/llm_manager.py

import ollama
from typing import Dict, Any, Optional
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 💡 **بخش جدید ۱:** تعریف کلاس برای نگهداری معیارهای تولید
@dataclass
class GenerationMetrics:
    """
    نگهداری معیارهای عملکرد تولید پاسخ توسط LLM.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    generation_time: float = 0.0

@dataclass
class LLMConfig:
    """تنظیمات مدل LLM"""
    model_name: str

class LLMManager:
    """مدیریت کننده اصلی مدل‌های LLM متصل به Ollama (نسخه پایدار و کامل)"""

    def __init__(self, configs: Dict[str, LLMConfig]):
        self.configs = configs
        self.active_model_key: Optional[str] = None
        self.client = None

        try:
            self.client = ollama.Client(host='http://localhost:11434')
            self.client.list()
            logger.info("LLMManager با موفقیت به سرور Ollama متصل شد.")
        except Exception as e:
            logger.error(f"خطا در اتصال اولیه به Ollama: {e}")
            logger.error("لطفاً از روشن بودن و در دسترس بودن سرور Ollama اطمینان حاصل کنید.")

    def set_active_model(self, model_key: str) -> bool:
        """تنظیم مدل فعال برای تولید پاسخ"""
        if model_key not in self.configs:
            logger.error(f"تنظیمات مدل {model_key} یافت نشد")
            return False
        self.active_model_key = model_key
        logger.info(f"مدل فعال تنظیم شد: {self.configs[self.active_model_key].model_name}")
        return True

    def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        تولید پاسخ با مدل فعال و بازگرداندن پاسخ به همراه معیارها.
        """
        if not self.active_model_key:
            return {'success': False, 'error': 'هیچ مدلی فعال نیست', 'response': '', 'metrics': None}

        if not self.client:
            return {'success': False, 'error': 'کلاینت Ollama در دسترس نیست.', 'response': '', 'metrics': None}

        config = self.configs[self.active_model_key]
        model_name = config.model_name
        
        start_time = time.time()
        try:
            response = self.client.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            # 💡 **بخش جدید ۲:** محاسبه معیارها و تغییر خروجی
            generation_time = time.time() - start_time
            
            # استخراج تعداد توکن‌ها از پاسخ Ollama
            metrics = GenerationMetrics(
                input_tokens=response.get('prompt_eval_count', 0),
                output_tokens=response.get('eval_count', 0),
                generation_time=round(generation_time, 2)
            )

            return {
                'success': True,
                'response': response['message']['content'],
                'model_used': model_name,
                'metrics': metrics  # <--- بازگرداندن معیارها
            }
        except Exception as e:
            logger.error(f"خطا در ارتباط با Ollama: {e}")
            error_msg = str(e)
            if "model" in str(e) and "not found" in str(e):
                error_msg = f"مدل '{model_name}' روی سرور Ollama یافت نشد. لطفاً با دستور `ollama pull {model_name}` آن را نصب کنید."
            
            return {'success': False, 'error': error_msg, 'response': '', 'metrics': None}

def create_model_configs() -> Dict[str, LLMConfig]:
    """ایجاد تنظیمات پیش‌فرض برای مدل‌های Ollama"""
    return {
        'qwen2.5:7b': LLMConfig(model_name="qwen2.5:7b"), # نام کامل مدل‌ها را اینجا بگذار
        'deepseek': LLMConfig(model_name="deepseek-r1:7b"),
        'mistral': LLMConfig(model_name="mistral:latest")
    }