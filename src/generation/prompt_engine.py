"""
src/generation/prompt_engine.py - موتور Prompt Engineering

این فایل مسئول ساخت prompt های بهینه برای انواع مختلف سوالات حقوقی است.
"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """انواع سوالات حقوقی"""
    GENERAL_INQUIRY = "سوال_عمومی"          # سوال ساده درباره قوانین
    DOCUMENT_ANALYSIS = "تحلیل_سند"         # تحلیل یک سند حقوقی
    DOCUMENT_COMPARISON = "مقایسه_اسناد"     # مقایسه چندین سند
    CONTRACT_REVIEW = "بررسی_قرارداد"       # بررسی قرارداد
    LEGAL_ADVICE = "مشاوره_حقوقی"          # ارائه مشاوره

class ContextType(Enum):
    """انواع context موجود"""
    LAW = "قانون"
    REGULATION = "آیین_نامه"
    GUIDELINE = "دستورالعمل"
    CONTRACT = "قرارداد"

@dataclass
class ContextInfo:
    """اطلاعات context بازیابی‌شده"""
    content: str
    source: str
    document_type: ContextType
    article_number: Optional[str] = None
    relevance_score: float = 0.0

class PromptEngine:
    """موتور اصلی Prompt Engineering"""
    
    def __init__(self):
        self.templates: Dict[QueryType, str] = {}
        self.legal_terms: Dict[str, str] = {}
        
        # بارگذاری template ها
        self._load_templates()
        self._load_legal_dictionary()
        
        logger.info("PromptEngine ایجاد شد")
    
    def _load_templates(self) -> None:
        """بارگذاری template های پیش‌تعریف‌شده"""
        
        # Template برای سوالات عمومی
        self.templates[QueryType.GENERAL_INQUIRY] = """شما یک دستیار حقوقی متخصص در حوزه پژوهش و فناوری هستید.

سوال کاربر: {question}

منابع حقوقی مرتبط:
{context}

دستورالعمل پاسخ:
1. پاسخ دقیق و جامع بر اساس منابع قانونی ارائه دهید
2. حتماً به مواد و بندهای مربوطه ارجاع دهید
3. در صورت وجود چندین تفسیر، همه را بیان کنید
4. اگر پاسخ در منابع موجود نیست، صریحاً اعلام کنید
5. از زبان ساده و قابل فهم استفاده کنید

نکات مهم:
- فقط بر اساس منابع ارائه‌شده پاسخ دهید
- شماره دقیق مواد و بندها را ذکر کنید
- در صورت ابهام، نیاز به مراجعه به کارشناس را بیان کنید

پاسخ:"""

        # Template برای تحلیل اسناد
        self.templates[QueryType.DOCUMENT_ANALYSIS] = """شما یک کارشناس حقوقی متخصص در تحلیل اسناد پژوهشی و فناوری هستید.

درخواست تحلیل: {question}

سند مورد تحلیل:
{document_content}

منابع مرجع برای تطبیق:
{context}

راهنمای تحلیل:
1. **ساختار و محتوا**: بررسی کامل ساختار سند و محتوای آن
2. **تطبیق قانونی**: مقایسه با قوانین و مقررات موجود
3. **نقاط قوت**: شناسایی بخش‌های مثبت و استاندارد
4. **نقاط ضعف**: شناسایی کمبودها و مشکلات
5. **تعارضات**: بررسی تعارض با سایر مقررات
6. **پیشنهادات**: ارائه راه‌حل برای بهبود

تحلیل:"""

        # Template برای بررسی قرارداد
        self.templates[QueryType.CONTRACT_REVIEW] = """شما یک مشاور حقوقی متخصص در قراردادهای پژوهشی و فناوری هستید.

نوع بررسی درخواستی: {question}

متن قرارداد:
{contract_content}

مقررات مرجع:
{context}

چک‌لیست بررسی:
1. **اطلاعات طرفین**: صحت و کفایت مشخصات
2. **موضوع قرارداد**: وضوح و دقت تعریف
3. **تعهدات طرفین**: متعادل و قابل اجرا بودن
4. **شرایط مالی**: شفافیت و انطباق با مقررات
5. **مالکیت فکری**: حفاظت و تعیین مالکیت
6. **فسخ و تعلیق**: شرایط و روندهای قانونی

بررسی:"""

        # Template برای مشاوره حقوقی
        self.templates[QueryType.LEGAL_ADVICE] = """شما یک مشاور حقوقی در حوزه پژوهش و فناوری هستید.

سوال مشاوره: {question}

اطلاعات زمینه:
{context}

رویکرد مشاوره:
1. **تحلیل موقعیت**: بررسی وضعیت کنونی
2. **شناسایی مقررات**: قوانین و ضوابط مرتبط
3. **ارزیابی ریسک**: شناسایی خطرات احتمالی
4. **ارائه راه‌حل**: پیشنهادات عملی و قانونی
5. **مراحل اجرا**: نحوه عملی کردن راه‌حل

مشاوره:"""
    
    def _load_legal_dictionary(self) -> None:
        """بارگذاری دیکشنری اصطلاحات حقوقی"""
        self.legal_terms = {
            "هیئت علمی": "اعضای هیئت علمی دانشگاه‌ها و مؤسسات آموزش عالی",
            "پژوهشگر": "فردی که به طور تخصصی به انجام پژوهش می‌پردازد",
            "مالکیت فکری": "حقوق قانونی ناشی از فعالیت ذهنی",
            "قرارداد پژوهشی": "قراردادی که موضوع آن انجام پژوهش است",
            "انتقال فناوری": "فرآیند انتقال دانش فنی از منابع علمی به کاربرد عملی",
            "شرکت دانش‌بنیان": "شرکتی که بر پایه دانش فنی پیشرفته فعالیت می‌کند"
        }
    
    def detect_query_type(self, question: str) -> QueryType:
        """تشخیص خودکار نوع سوال"""
        question_lower = question.lower()
        
        # کلمات کلیدی برای هر نوع سوال
        keywords = {
            QueryType.DOCUMENT_ANALYSIS: [
                "تحلیل", "بررسی", "ارزیابی", "بسنجید", "نظر دهید", "تجزیه"
            ],
            QueryType.DOCUMENT_COMPARISON: [
                "مقایسه", "تفاوت", "شباهت", "در مقابل", "نسبت به", "مقابله"
            ],
            QueryType.CONTRACT_REVIEW: [
                "قرارداد", "پیمان", "توافق‌نامه", "بررسی قرارداد", "عقد"
            ],
            QueryType.LEGAL_ADVICE: [
                "مشاوره", "توصیه", "پیشنهاد", "راهنمایی", "چه کنم", "چگونه"
            ]
        }
        
        # امتیازدهی به انواع مختلف
        scores = {query_type: 0 for query_type in QueryType}
        
        for query_type, words in keywords.items():
            for word in words:
                if word in question_lower:
                    scores[query_type] += 1
        
        # انتخاب نوع با بالاترین امتیاز
        best_type = max(scores.items(), key=lambda x: x[1])
        
        if best_type[1] > 0:
            return best_type[0]
        
        # در صورت عدم تشخیص، سوال عمومی
        return QueryType.GENERAL_INQUIRY
    
    def _clean_and_validate_question(self, question: str) -> str:
        """پاک‌سازی و اعتبارسنجی سوال"""
        # حذف فضاهای اضافی
        question = re.sub(r'\s+', ' ', question.strip())
        
        # بررسی حداقل طول
        if len(question) < 10:
            raise ValueError("سوال بیش از حد کوتاه است")
        
        # بررسی حداکثر طول
        if len(question) > 2000:
            raise ValueError("سوال بیش از حد طولانی است")
        
        return question
    
    def _optimize_context(self, contexts: List[ContextInfo], max_length: int = 3000) -> str:
        """بهینه‌سازی و ترکیب context ها"""
        
        # مرتب‌سازی بر اساس امتیاز مرتبط بودن
        sorted_contexts = sorted(contexts, key=lambda x: x.relevance_score, reverse=True)
        
        result = ""
        current_length = 0
        
        for i, ctx in enumerate(sorted_contexts):
            # ایجاد فرمت استاندارد برای هر context
            formatted_ctx = f"""
منبع {i+1}: {ctx.source}
نوع سند: {ctx.document_type.value}
{f"ماده/بخش: {ctx.article_number}" if ctx.article_number else ""}

محتوا:
{ctx.content}

---
"""
            
            # بررسی محدودیت طول
            if current_length + len(formatted_ctx) > max_length:
                break
            
            result += formatted_ctx
            current_length += len(formatted_ctx)
        
        return result.strip()
    
    def build_prompt(
        self,
        question: str,
        contexts: List[ContextInfo],
        query_type: Optional[QueryType] = None,
        **additional_fields
    ) -> Tuple[str, QueryType]:
        """ساخت prompt نهایی"""
        
        # پاک‌سازی سوال
        clean_question = self._clean_and_validate_question(question)
        
        # تشخیص نوع سوال
        if query_type is None:
            query_type = self.detect_query_type(clean_question)
        
        # دریافت template مناسب
        if query_type not in self.templates:
            logger.warning(f"Template برای {query_type} یافت نشد، از template عمومی استفاده می‌شود")
            query_type = QueryType.GENERAL_INQUIRY
        
        template = self.templates[query_type]
        
        # بهینه‌سازی context
        optimized_context = self._optimize_context(contexts)
        
        # آماده‌سازی فیلدها
        fields = {
            'question': clean_question,
            'context': optimized_context,
            **additional_fields
        }
        
        # ساخت prompt
        try:
            final_prompt = template.format(**fields)
        except KeyError as e:
            # در صورت نبود فیلد، template ساده‌تر استفاده می‌کنیم
            simple_template = self.templates[QueryType.GENERAL_INQUIRY]
            final_prompt = simple_template.format(
                question=clean_question,
                context=optimized_context
            )
        
        logger.info(f"Prompt ساخته شد - نوع: {query_type}, طول: {len(final_prompt)}")
        
        return final_prompt, query_type
    
    def validate_prompt_quality(self, prompt: str) -> Dict[str, Any]:
        """ارزیابی کیفیت prompt ساخته‌شده"""
        
        quality_metrics = {
            'length': len(prompt),
            'has_context': 'منابع' in prompt or 'context' in prompt.lower(),
            'has_instructions': 'دستورالعمل' in prompt,
            'has_question': 'سوال' in prompt,
            'appropriate_length': 1000 <= len(prompt) <= 8000,
            'quality_score': 0
        }
        
        # محاسبه امتیاز کیفیت
        score = 0
        if quality_metrics['has_context']: score += 25
        if quality_metrics['has_instructions']: score += 25
        if quality_metrics['has_question']: score += 25
        if quality_metrics['appropriate_length']: score += 25
        
        quality_metrics['quality_score'] = score
        quality_metrics['status'] = 'عالی' if score >= 90 else 'خوب' if score >= 70 else 'نیازمند بهبود'
        
        return quality_metrics

# تست
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # ایجاد موتور prompt
    engine = PromptEngine()
    
    # نمونه context ها
    contexts = [
        ContextInfo(
            content="اعضای هیئت علمی موظف به انجام پژوهش هستند",
            source="قانون مقررات انتظامی هیئت علمی",
            document_type=ContextType.LAW,
            article_number="3",
            relevance_score=0.9
        )
    ]
    
    # تست سوال‌های مختلف
    test_questions = [
        "وظایف اعضای هیئت علمی چیست؟",
        "لطفاً این آیین‌نامه را تحلیل کنید",
        "تفاوت این دو قانون چیست؟",
        "مشاوره درباره قرارداد پژوهشی می‌خواهم"
    ]
    
    for question in test_questions:
        print(f"\n🔸 سوال: {question}")
        
        # تشخیص نوع
        query_type = engine.detect_query_type(question)
        print(f"🎯 نوع: {query_type.value}")
        
        # ساخت prompt
        prompt, detected_type = engine.build_prompt(question, contexts)
        print(f"📏 طول prompt: {len(prompt)} کاراکتر")
        
        # ارزیابی کیفیت
        quality = engine.validate_prompt_quality(prompt)
        print(f"⭐ کیفیت: {quality['quality_score']}/100")
        
        print("-" * 50)
    
    print("✅ تست PromptEngine کامل شد")