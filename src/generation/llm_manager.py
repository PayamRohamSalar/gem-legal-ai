# src/generation/llm_manager.py

import ollama
from typing import Dict, List, Optional, Any
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """تنظیمات مدل LLM"""
    model_name: str

class LLMManager:
    """مدیریت کننده اصلی مدل‌های LLM متصل به Ollama (نسخه پایدار)"""
    
    def __init__(self, configs: Dict[str, LLMConfig]):
        self.configs = configs
        self.active_model_key: Optional[str] = None
        
        # اصلاح کلیدی: مشخص کردن صریح هاست و بررسی اولیه اتصال
        try:
            self.client = ollama.Client(host='http://localhost:11434')
            # بررسی اولیه برای اطمینان از اتصال و وجود مدل‌ها
            self.client.list() 
            logger.info("LLMManager با موفقیت به سرور Ollama متصل شد.")
        except Exception as e:
            logger.error(f"خطا در اتصال اولیه به Ollama: {e}")
            logger.error("لطفاً از روشن بودن و در دسترس بودن سرور Ollama اطمینان حاصل کنید.")
            self.client = None

    def set_active_model(self, model_key: str) -> bool:
        """تنظیم مدل فعال برای تولید پاسخ"""
        if model_key not in self.configs:
            logger.error(f"تنظیمات مدل {model_key} یافت نشد")
            return False
        self.active_model_key = model_key
        logger.info(f"مدل فعال تنظیم شد: {self.configs[self.active_model_key].model_name}")
        return True

    def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """تولید پاسخ با مدل فعال از طریق Ollama"""
        if not self.active_model_key:
            return {'success': False, 'error': 'هیچ مدلی فعال نیست', 'response': ''}
        
        if not self.client:
             return {'success': False, 'error': 'کلاینت Ollama در دسترس نیست.', 'response': ''}

        config = self.configs[self.active_model_key]
        model_name = config.model_name
        
        try:
            response = self.client.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return {
                'success': True,
                'response': response['message']['content'],
                'model_used': model_name
            }
        except Exception as e:
            logger.error(f"خطا در ارتباط با Ollama: {e}")
            # تلاش برای ارائه یک پیام خطای واضح‌تر
            if "model" in str(e) and "not found" in str(e):
                error_msg = f"مدل '{model_name}' روی سرور Ollama یافت نشد. لطفاً با دستور `ollama pull {model_name}` آن را نصب کنید."
            else:
                error_msg = str(e)
            return {'success': False, 'error': error_msg, 'response': ''}

def create_model_configs() -> Dict[str, LLMConfig]:
    """ایجاد تنظیمات پیش‌فرض برای مدل‌های Ollama"""
    return {
        'qwen': LLMConfig(model_name="qwen2.5:7b"),
        'deepseek': LLMConfig(model_name="deepseek-r1:7b"),
        'mistral': LLMConfig(model_name="mistral:latest")
    }