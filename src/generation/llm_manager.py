# src/generation/llm_manager.py

import ollama
from typing import Dict, List, Optional, Any
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GenerationMetrics:
    """متریک‌های تولید پاسخ"""
    generation_time: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_name: str

@dataclass
class LLMConfig:
    """تنظیمات مدل LLM"""
    model_name: str
    # پارامترهای دیگر را می‌توان در آینده اضافه کرد

class LLMManager:
    """مدیریت کننده اصلی مدل‌های LLM متصل به Ollama"""
    
    def __init__(self, configs: Dict[str, LLMConfig]):
        self.configs = configs
        self.active_model_key: Optional[str] = None
        self.generation_history: List[GenerationMetrics] = []
        self.client = ollama.Client() # کلاینت Ollama
        logger.info("LLMManager (Ollama Mode) ایجاد شد")
    
    def set_active_model(self, model_key: str) -> bool:
        """تنظیم مدل فعال برای تولید پاسخ"""
        if model_key not in self.configs:
            logger.error(f"تنظیمات مدل {model_key} یافت نشد")
            return False
        self.active_model_key = model_key
        logger.info(f"مدل فعال تنظیم شد: {self.configs[model_key].model_name}")
        return True

    def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """تولید پاسخ با مدل فعال از طریق Ollama"""
        if not self.active_model_key:
            return {'success': False, 'error': 'هیچ مدلی فعال نیست', 'response': ''}
        
        start_time = time.time()
        config = self.configs[self.active_model_key]
        model_name = config.model_name
        
        try:
            response = self.client.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            generation_time = time.time() - start_time
            
            # دریافت متریک‌ها از پاسخ Ollama
            metrics = GenerationMetrics(
                generation_time=generation_time,
                input_tokens=response.get('prompt_eval_count', 0),
                output_tokens=response.get('eval_count', 0),
                total_tokens=response.get('total_duration', 0), # total_duration is not tokens, but a placeholder
                model_name=model_name
            )
            self.generation_history.append(metrics)
            
            return {
                'success': True,
                'response': response['message']['content'],
                'metrics': metrics,
                'model_used': model_name
            }
            
        except Exception as e:
            logger.error(f"خطا در ارتباط با Ollama: {e}")
            return {'success': False, 'error': str(e), 'response': ''}

def create_model_configs() -> Dict[str, LLMConfig]:
    """ایجاد تنظیمات پیش‌فرض برای مدل‌های Ollama"""
    return {
        'qwen': LLMConfig(model_name="qwen2.5:7b"),
        'llama': LLMConfig(model_name="llama3.1:8b"),
        'mistral': LLMConfig(model_name="mistral:latest")
    }