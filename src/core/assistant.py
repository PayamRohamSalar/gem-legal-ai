"""
src/generation/integrated_response_system.py - سیستم یکپارچه تولید پاسخ

این فایل تمام اجزای فاز 3 را ادغام می‌کند و API جامع برای تولید پاسخ‌های حقوقی ارائه می‌دهد.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging
import time
from datetime import datetime
import json

# Import کردن ماژول‌های فاز 3
from src.generation.llm_manager import LLMManager, LLMConfig, GenerationMetrics, create_model_configs
from src.generation.prompt_engine import PromptEngine, QueryType, ContextInfo, ContextType
from src.generation.citation_engine import CitationEngine, Citation

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

@dataclass
class ResponseRequest:
    """درخواست تولید پاسخ"""
    question: str
    contexts: List[Dict[str, Any]]
    query_type: Optional[QueryType] = None
    model_preference: Optional[str] = None
    format_style: str = 'standard'
    include_citations: bool = True
    max_response_length: int = 2048
    temperature: float = 0.1
    additional_instructions: Optional[str] = None

@dataclass
class ResponseResult:
    """نتیجه تولید پاسخ"""
    # پاسخ اصلی
    response: str
    enhanced_response: str
    
    # اطلاعات پردازش
    query_type: QueryType
    processing_time: float
    success: bool
    
    # ارجاعات و منابع
    citations: List[Citation]
    references_list: str
    citation_validation: Dict[str, Any]
    
    # متریک‌های تولید
    generation_metrics: Optional[GenerationMetrics]
    
    # کیفیت پاسخ
    quality_score: float
    confidence_score: float
    
    # اطلاعات اضافی
    model_used: str
    prompt_used: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class LegalResponseSystem:
    """سیستم جامع تولید پاسخ حقوقی"""
    
    def __init__(
        self,
        llm_configs: Optional[Dict[str, LLMConfig]] = None,
        default_model: str = 'qwen2.5:7b', # تغییر به مدلی که نصب داری
        enable_caching: bool = True
    ):
        # اجزای اصلی سیستم
        self.llm_manager = LLMManager(llm_configs or create_model_configs())
        self.prompt_engine = PromptEngine()
        self.citation_engine = CitationEngine()
        
        # تنظیمات
        self.default_model = default_model
        self.enable_caching = enable_caching
        self.response_cache: Dict[str, ResponseResult] = {}
        
        # آمار عملکرد
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'cache_hits': 0
        }
        
        logger.info("سیستم تولید پاسخ حقوقی راه‌اندازی شد")

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

    def _generate_cache_key(self, request: ResponseRequest) -> str:
        """تولید کلید یکتا برای کش"""
        key_data = {
            'question': request.question,
            'contexts_hash': hash(str(sorted([str(ctx) for ctx in request.contexts]))),
            'query_type': request.query_type.value if request.query_type else None,
            'model': request.model_preference or self.default_model,
            'temperature': request.temperature
        }
        return str(hash(str(key_data)))
    
    def _prepare_contexts(self, contexts: List[Dict[str, Any]]) -> List[ContextInfo]:
        """تبدیل context های ورودی به فرمت استاندارد"""
        context_infos = []
        
        for ctx in contexts:
            try:
                # تشخیص نوع سند
                doc_type_str = ctx.get('document_type', 'قانون')
                if doc_type_str == 'قانون':
                    doc_type = ContextType.LAW
                elif doc_type_str == 'آیین_نامه':
                    doc_type = ContextType.REGULATION
                elif doc_type_str == 'دستورالعمل':
                    doc_type = ContextType.GUIDELINE
                else:
                    doc_type = ContextType.LAW  # پیش‌فرض
                
                context_info = ContextInfo(
                    content=ctx.get('content', ''),
                    source=ctx.get('source', ''),
                    document_type=doc_type,
                    article_number=ctx.get('article_number'),
                    relevance_score=ctx.get('relevance_score', 0.8)
                )
                context_infos.append(context_info)
                
            except Exception as e:
                logger.warning(f"خطا در پردازش context: {e}")
                continue
        
        return context_infos
    
    def _calculate_quality_score(
        self, 
        response: str, 
        citations: List[Citation],
        generation_metrics: Optional[GenerationMetrics]
    ) -> float:
        """محاسبه امتیاز کیفیت پاسخ"""
        
        score = 0.0
        max_score = 100.0
        
        # طول مناسب پاسخ (20 امتیاز)
        response_length = len(response)
        if 200 <= response_length <= 2000:
            score += 20
        elif 100 <= response_length < 200 or 2000 < response_length <= 3000:
            score += 10
        
        # وجود ارجاعات (25 امتیاز)
        if citations:
            score += 20
            # کیفیت ارجاعات (5 امتیاز اضافی)
            avg_confidence = sum(c.confidence_score for c in citations) / len(citations)
            score += 5 * avg_confidence
        
        # ساختار مناسب پاسخ (20 امتیاز)
        if any(keyword in response for keyword in ['بر اساس', 'مطابق', 'طبق']):
            score += 10
        if any(keyword in response for keyword in ['ماده', 'بند', 'تبصره']):
            score += 10
        
        # وضوح و خوانایی (15 امتیاز)
        sentences = response.split('.')
        if len(sentences) >= 3:  # حداقل 3 جمله
            score += 7
        if len(response.split()) >= 50:  # حداقل 50 کلمه
            score += 8
        
        # عملکرد تولید (20 امتیاز)
        if generation_metrics:
            # زمان مناسب (10 امتیاز)
            if generation_metrics.generation_time < 10:
                score += 10
            elif generation_metrics.generation_time < 20:
                score += 5
            
            # تعداد مناسب توکن (10 امتیاز)
            if generation_metrics.output_tokens >= 50:
                score += 10
            elif generation_metrics.output_tokens >= 20:
                score += 5
        
        return min(score, max_score)
    
    def _calculate_confidence_score(
        self, 
        response: str, 
        citations: List[Citation],
        context_count: int
    ) -> float:
        """محاسبه اطمینان از صحت پاسخ"""
        
        confidence = 0.0
        
        # وجود منابع کافی
        if context_count >= 3:
            confidence += 0.3
        elif context_count >= 1:
            confidence += 0.2
        
        # کیفیت ارجاعات
        if citations:
            avg_citation_confidence = sum(c.confidence_score for c in citations) / len(citations)
            confidence += 0.4 * avg_citation_confidence
        
        # تطابق با الگوهای حقوقی
        legal_patterns = ['بر اساس', 'مطابق', 'طبق', 'وفق', 'ماده', 'بند', 'قانون']
        pattern_matches = sum(1 for pattern in legal_patterns if pattern in response)
        confidence += 0.3 * min(pattern_matches / len(legal_patterns), 1.0)
        
        return min(confidence, 1.0)
    
    async def generate_response(self, request: ResponseRequest) -> ResponseResult:
        """تولید پاسخ کامل با تمام قابلیت‌ها"""
        
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # بررسی کش
            if self.enable_caching:
                cache_key = self._generate_cache_key(request)
                if cache_key in self.response_cache:
                    self.stats['cache_hits'] += 1
                    logger.info("پاسخ از کش بازیابی شد")
                    return self.response_cache[cache_key]
            
            # آماده‌سازی context ها
            context_infos = self._prepare_contexts(request.contexts)
            
            # تشخیص نوع سوال اگر مشخص نشده
            query_type = request.query_type or self.prompt_engine.detect_query_type(request.question)
            
            # ساخت prompt
            additional_fields = {}
            if request.additional_instructions:
                additional_fields['additional_instructions'] = request.additional_instructions
            
            prompt, detected_query_type = self.prompt_engine.build_prompt(
                request.question,
                context_infos,
                query_type,
                **additional_fields
            )
            
            # تولید پاسخ
            generation_result = self.llm_manager.generate_response(
                prompt,
                temperature=request.temperature,
                max_new_tokens=request.max_response_length
            )
            
            if not generation_result['success']:
                raise Exception(generation_result.get('error', 'خطای نامشخص در تولید'))
            
            response_text = generation_result['response']
            generation_metrics = generation_result['metrics']
            
            # پردازش ارجاعات
            citations = []
            enhanced_response = response_text
            references_list = ""
            citation_validation = {}
            
            if request.include_citations:
                citation_result = self.citation_engine.enhance_response_with_citations(
                    response_text, 
                    request.contexts
                )
                
                enhanced_response = citation_result['enhanced_response']
                citations = citation_result['citations']
                references_list = citation_result['references_list']
                citation_validation = citation_result['validation']
            
            # محاسبه امتیازات کیفیت
            quality_score = self._calculate_quality_score(
                response_text, 
                citations, 
                generation_metrics
            )
            
            confidence_score = self._calculate_confidence_score(
                response_text, 
                citations, 
                len(request.contexts)
            )
            
            # ایجاد نتیجه نهایی
            result = ResponseResult(
                response=response_text,
                enhanced_response=enhanced_response,
                query_type=detected_query_type,
                processing_time=time.time() - start_time,
                success=True,
                citations=citations,
                references_list=references_list,
                citation_validation=citation_validation,
                generation_metrics=generation_metrics,
                quality_score=quality_score,
                confidence_score=confidence_score,
                model_used=request.model_preference or self.default_model,
                prompt_used=prompt,
                metadata={
                    'context_count': len(request.contexts),
                    'prompt_length': len(prompt),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # ذخیره در کش
            if self.enable_caching:
                self.response_cache[cache_key] = result
            
            # بروزرسانی آمار
            self.stats['successful_requests'] += 1
            self._update_stats(result.processing_time)
            
            logger.info(f"پاسخ با موفقیت تولید شد - زمان: {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"خطا در تولید پاسخ: {e}")
            
            return ResponseResult(
                response="",
                enhanced_response="",
                query_type=QueryType.GENERAL_INQUIRY,
                processing_time=time.time() - start_time,
                success=False,
                citations=[],
                references_list="",
                citation_validation={},
                generation_metrics=None,
                quality_score=0.0,
                confidence_score=0.0,
                model_used=self.default_model,
                prompt_used="",
                error_message=str(e)
            )
    
    def _update_stats(self, processing_time: float) -> None:
        """بروزرسانی آمار عملکرد"""
        total_successful = self.stats['successful_requests']
        if total_successful > 0:
            current_avg = self.stats['average_response_time']
            self.stats['average_response_time'] = (
                (current_avg * (total_successful - 1) + processing_time) / total_successful
            )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """دریافت آمار عملکرد سیستم"""
        return {
            **self.stats,
            'llm_info': self.llm_manager.get_model_info(),
            'cache_size': len(self.response_cache),
            'prompt_templates': len(self.prompt_engine.templates),
            'uptime': datetime.now().isoformat()
        }
    
    def clear_cache(self) -> None:
        """پاک کردن کش"""
        self.response_cache.clear()
        logger.info("کش سیستم پاک شد")
    
    def shutdown(self) -> None:
        """خاموش کردن سیستم"""
        logger.info("شروع خاموش کردن سیستم...")
        
        # پاک‌سازی منابع
        self.llm_manager.cleanup()
        self.clear_cache()
        
        logger.info("سیستم با موفقیت خاموش شد")

class SimpleLegalAssistant:
    """رابط ساده برای استفاده از سیستم"""
    
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

# تست
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def test_system():
        """تست سیستم یکپارچه"""
        
        print("🚀 تست سیستم یکپارچه تولید پاسخ")
        print("=" * 50)
        
        # ایجاد دستیار ساده
        assistant = SimpleLegalAssistant()
        
        # راه‌اندازی
        success = await assistant.setup()
        if not success:
            print("❌ خطا در راه‌اندازی")
            return
        
        print("✅ سیستم راه‌اندازی شد")
        
        # نمونه context های حقوقی
        contexts = [
            {
                'content': 'اعضای هیئت علمی موظف به انجام پژوهش و تحقیق در زمینه تخصصی خود هستند',
                'source': 'قانون مقررات انتظامی هیئت علمی - مصوب 1364',
                'document_type': 'قانون',
                'article_number': '3',
                'relevance_score': 0.95
            },
            {
                'content': 'پژوهش باید منجر به تولید دانش نوین و کاربردی باشد',
                'source': 'آیین‌نامه ارتقای اعضای هیئت علمی - مصوب 1398',
                'document_type': 'آیین_نامه',
                'article_number': '7',
                'relevance_score': 0.88
            }
        ]
        
        # تست سوالات مختلف
        test_questions = [
            "وظایف اعضای هیئت علمی در زمینه پژوهش چیست؟",
            "معیارهای ارزیابی پژوهش چگونه است؟",
            "لطفاً این آیین‌نامه را تحلیل کنید"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔸 تست {i}: {question}")
            
            start_time = time.time()
            response = await assistant.ask(question, contexts)
            end_time = time.time()
            
            print(f"📝 پاسخ: {response[:200]}...")
            print(f"⏱️  زمان: {end_time - start_time:.2f} ثانیه")
            print("-" * 40)
        
        # نمایش آمار
        stats = assistant.get_stats()
        print(f"\n📊 آمار سیستم:")
        print(f"  - کل درخواست‌ها: {stats['total_requests']}")
        print(f"  - درخواست‌های موفق: {stats['successful_requests']}")
        print(f"  - میانگین زمان پاسخ: {stats['average_response_time']:.2f}s")
        print(f"  - Cache hits: {stats['cache_hits']}")
        print(f"  - مدل فعال: {stats['llm_info']['active_model']}")
        
        # خاموش کردن
        assistant.shutdown()
        print("\n✅ تست کامل شد")
    
    # اجرای تست
    asyncio.run(test_system())