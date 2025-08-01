# src/core/assistant.py

from typing import Dict, List, Optional, Any
from dataclasses import asdict
import logging
import time
from datetime import datetime
import json

# ایمپورت کردن ماژول‌های پروژه
from src.generation.llm_manager import LLMManager, LLMConfig, GenerationMetrics, create_model_configs
from src.generation.prompt_engine import PromptEngine, QueryType, ContextInfo, ContextType
from src.generation.citation_engine import CitationEngine, Citation

# مدل‌های داده‌ای که قبلا در فایل شما بود، اینجا هم لازم است
from pydantic import BaseModel, Field

# (کلاس‌های ResponseRequest و ResponseResult که در فایل شما بود را اینجا کپی می‌کنیم)
# ... (کدهای dataclass های ResponseRequest و ResponseResult را اینجا قرار بده) ...
# برای جلوگیری از تکرار، فرض می‌کنیم همان کدها اینجا هستند.

# کلاس‌های اصلی
class LegalResponseSystem:
    # ... (کد کامل کلاس LegalResponseSystem که فرستادی اینجا قرار می‌گیرد) ...
    # من فقط تغییرات جزئی برای هماهنگی بیشتر اعمال می‌کنم
    def __init__(
        self,
        llm_configs: Optional[Dict[str, LLMConfig]] = None,
        default_model: str = 'qwen2.5:7b', # تغییر به مدلی که نصب داری
        enable_caching: bool = True
    ):
        # ... (بقیه کد __init__ بدون تغییر)
        pass

    async def initialize(self) -> bool:
        """راه‌اندازی اولیه سیستم با استفاده از LLM Manager واقعی"""
        try:
            # استفاده از LLM Manager واقعی
            if not self.llm_manager.client:
                 logger.error("کلاینت Ollama در دسترس نیست. لطفاً از روشن بودن سرور Ollama اطمینان حاصل کنید.")
                 return False

            self.llm_manager.set_active_model(self.default_model)
            logger.info(f"سیستم با مدل فعال '{self.default_model}' با موفقیت راه‌اندازی شد")
            return True

        except Exception as e:
            logger.error(f"خطا در راه‌اندازی سیستم: {e}")
            return False

    # ... (بقیه متدهای کلاس LegalResponseSystem بدون تغییر)
    # _generate_cache_key, _prepare_contexts, _calculate_quality_score, ...
    # generate_response, _update_stats, get_system_stats, ...


class SimpleLegalAssistant:
    """رابط ساده برای استفاده از سیستم در FastAPI."""

    def __init__(self):
        # ساخت یک نمونه از سیستم اصلی با تنظیمات پیش‌فرض
        # مدل پیش‌فرض را به یکی از مدل‌های نصب شده تغییر می‌دهیم
        self.system = LegalResponseSystem(default_model='qwen2.5:7b')
        self.initialized = False

    async def setup(self) -> bool:
        """راه‌اندازی سیستم اصلی."""
        self.initialized = await self.system.initialize()
        return self.initialized

    async def ask(self, request: ResponseRequest) -> ResponseResult:
        """
        ارسال یک درخواست کامل به سیستم و دریافت نتیجه کامل.
        این متد دیگر فقط یک رشته برنمی‌گرداند.
        """
        if not self.initialized:
            raise RuntimeError("سیستم راه‌اندازی نشده است")

        return await self.system.generate_response(request)

    def get_stats(self) -> Dict[str, Any]:
        """دریافت آمار سیستم."""
        return self.system.get_system_stats()

    def clear_cache(self):
        """پاک کردن کش."""
        self.system.clear_cache()

    def shutdown(self) -> None:
        """خاموش کردن سیستم."""
        self.system.shutdown()