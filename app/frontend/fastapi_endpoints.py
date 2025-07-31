"""
interface/fastapi_endpoints.py - API وب سرویس دستیار حقوقی

این فایل رابط وب سرویس کامل برای سیستم دستیار حقوقی ارائه می‌دهد.
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import asyncio
import logging
import time
import sys
from pathlib import Path

# اضافه کردن مسیر src
current_dir = Path(__file__).parent.parent
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

# Import سیستم اصلی
try:
    from generation.integrated_response_system import (
        LegalResponseSystem, ResponseRequest, SimpleLegalAssistant
    )
    from generation.prompt_engine import QueryType
except ImportError as e:
    print(f"❌ خطا در import: {e}")
    print("⚠️  ابتدا فایل‌های فاز 3 را در مسیر src/generation/ قرار دهید")
    sys.exit(1)

# تنظیم لاگ‌گیری
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# مدل‌های Pydantic برای API

class ContextModel(BaseModel):
    """مدل context برای API"""
    content: str = Field(..., description="محتوای context")
    source: str = Field(..., description="منبع context")
    document_type: str = Field(default="قانون", description="نوع سند")
    article_number: Optional[str] = Field(None, description="شماره ماده")
    relevance_score: float = Field(default=0.8, ge=0.0, le=1.0, description="امتیاز مرتبط بودن")

class QuestionRequest(BaseModel):
    """درخواست پرسش"""
    question: str = Field(..., min_length=10, max_length=2000, description="متن سوال")
    contexts: List[ContextModel] = Field(default=[], description="منابع مرتبط")
    query_type: Optional[str] = Field(None, description="نوع سوال")
    include_citations: bool = Field(default=True, description="شامل ارجاعات")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="دمای تولید")

    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('سوال نمی‌تواند خالی باشد')
        return v.strip()

class ResponseModel(BaseModel):
    """مدل پاسخ API"""
    success: bool
    response: str
    enhanced_response: str
    query_type: str
    processing_time: float
    quality_score: float
    confidence_score: float
    citation_count: int
    references_list: str
    model_used: str
    timestamp: str
    error_message: Optional[str] = None

class SystemStatsModel(BaseModel):
    """مدل آمار سیستم"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    cache_hits: int
    active_model: Optional[str]
    memory_usage_mb: float
    uptime: str

class HealthCheckModel(BaseModel):
    """مدل بررسی سلامت"""
    status: str
    message: str
    timestamp: str
    version: str = "1.0.0"

# ایجاد اپلیکیشن FastAPI
app = FastAPI(
    title="دستیار حقوقی هوشمند",
    description="API جامع برای پاسخگویی به سوالات حقوقی در حوزه پژوهش و فناوری",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# تنظیم CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در production محدود کنید
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# متغیرهای global
legal_assistant: Optional[SimpleLegalAssistant] = None

# رویدادهای شروع و پایان
@app.on_event("startup")
async def startup_event():
    """راه‌اندازی سیستم"""
    global legal_assistant
    
    logger.info("شروع راه‌اندازی دستیار حقوقی...")
    
    try:
        legal_assistant = SimpleLegalAssistant()
        success = await legal_assistant.setup()
        
        if success:
            logger.info("سیستم با موفقیت راه‌اندازی شد")
        else:
            logger.error("خطا در راه‌اندازی سیستم")
            
    except Exception as e:
        logger.error(f"خطا در startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """خاموش کردن سیستم"""
    global legal_assistant
    
    if legal_assistant:
        legal_assistant.shutdown()
        logger.info("سیستم خاموش شد")

# Dependency برای دریافت سیستم فعال
async def get_legal_assistant():
    """دریافت دستیار حقوقی فعال"""
    if legal_assistant is None:
        raise HTTPException(status_code=503, detail="سیستم راه‌اندازی نشده است")
    return legal_assistant

# Root endpoint
@app.get("/", response_model=Dict[str, str])
async def root():
    """صفحه اصلی"""
    return {
        "message": "دستیار حقوقی هوشمند",
        "version": "1.0.0",
        "description": "API برای پاسخگویی به سوالات حقوقی",
        "docs": "/docs",
        "health": "/health",
        "stats": "/stats"
    }

# Health check
@app.get("/health", response_model=HealthCheckModel)
async def health_check():
    """بررسی سلامت سیستم"""
    try:
        status = "healthy" if legal_assistant else "unhealthy"
        message = "سیستم فعال است" if legal_assistant else "سیستم راه‌اندازی نشده"
        
        return HealthCheckModel(
            status=status,
            message=message,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    except Exception as e:
        return HealthCheckModel(
            status="error",
            message=f"خطا در بررسی سلامت: {str(e)}",
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

# اصلی‌ترین endpoint - پرسش و پاسخ
@app.post("/ask", response_model=ResponseModel)
async def ask_question(
    request_data: QuestionRequest,
    request: Request,
    assistant: SimpleLegalAssistant = Depends(get_legal_assistant)
):
    """پرسش از دستیار حقوقی"""
    
    start_time = time.time()
    client_ip = request.client.host
    
    logger.info(f"درخواست جدید از {client_ip}: {request_data.question[:50]}...")
    
    try:
        # تبدیل contexts به فرمت مناسب
        contexts = []
        for ctx in request_data.contexts:
            contexts.append({
                'content': ctx.content,
                'source': ctx.source,
                'document_type': ctx.document_type,
                'article_number': ctx.article_number,
                'relevance_score': ctx.relevance_score
            })
        
        # ارسال درخواست به سیستم
        response = await assistant.ask(request_data.question, contexts)
        
        # دریافت آمار برای اطلاعات اضافی
        stats = assistant.get_stats()
        
        processing_time = time.time() - start_time
        
        # بررسی موفقیت پاسخ
        success = response and "خطا" not in response and len(response) > 10
        
        result = ResponseModel(
            success=success,
            response=response,
            enhanced_response=response,
            query_type=request_data.query_type or "سوال_عمومی",
            processing_time=processing_time,
            quality_score=85.0 if success else 0.0,  # تخمینی
            confidence_score=0.8 if success else 0.0,  # تخمینی
            citation_count=response.count('ماده') if success else 0,  # تخمینی
            references_list="منابع در پاسخ ذکر شده‌اند" if success else "",
            model_used=stats.get('llm_info', {}).get('active_model', 'qwen_7b'),
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            error_message=None if success else "خطا در تولید پاسخ"
        )
        
        logger.info(f"پاسخ تولید شد - زمان: {processing_time:.2f}s")
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"خطا در پردازش سوال: {e}")
        
        return ResponseModel(
            success=False,
            response="",
            enhanced_response="",
            query_type=request_data.query_type or "سوال_عمومی",
            processing_time=processing_time,
            quality_score=0.0,
            confidence_score=0.0,
            citation_count=0,
            references_list="",
            model_used="نامشخص",
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            error_message=str(e)
        )

# آمار سیستم
@app.get("/stats", response_model=SystemStatsModel)
async def get_system_stats(
    assistant: SimpleLegalAssistant = Depends(get_legal_assistant)
):
    """دریافت آمار عملکرد سیستم"""
    try:
        stats = assistant.get_stats()
        
        return SystemStatsModel(
            total_requests=stats.get('total_requests', 0),
            successful_requests=stats.get('successful_requests', 0),
            failed_requests=stats.get('failed_requests', 0),
            average_response_time=stats.get('average_response_time', 0.0),
            cache_hits=stats.get('cache_hits', 0),
            active_model=stats.get('llm_info', {}).get('active_model'),
            memory_usage_mb=stats.get('llm_info', {}).get('memory_usage_mb', 0.0),
            uptime=stats.get('uptime', time.strftime('%Y-%m-%d %H:%M:%S'))
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در دریافت آمار: {e}")

# تست سریع سیستم
@app.get("/test")
async def test_system(
    assistant: SimpleLegalAssistant = Depends(get_legal_assistant)
):
    """تست سریع عملکرد سیستم"""
    try:
        # سوال تست ساده
        test_question = "وظایف اعضای هیئت علمی در پژوهش چیست؟"
        test_contexts = [{
            'content': 'اعضای هیئت علمی موظف به انجام پژوهش هستند',
            'source': 'قانون مقررات انتظامی هیئت علمی',
            'document_type': 'قانون',
            'relevance_score': 0.9
        }]
        
        start_time = time.time()
        response = await assistant.ask(test_question, test_contexts)
        end_time = time.time()
        
        return {
            "status": "success",
            "test_question": test_question,
            "response_preview": response[:200] + "..." if len(response) > 200 else response,
            "response_length": len(response),
            "processing_time": end_time - start_time,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }

# مدیریت کش
@app.delete("/cache")
async def clear_cache(
    assistant: SimpleLegalAssistant = Depends(get_legal_assistant)
):
    """پاک کردن کش سیستم"""
    try:
        assistant.system.clear_cache()
        return {
            "status": "success",
            "message": "کش با موفقیت پاک شد",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در پاک کردن کش: {e}")

# معلومات سیستم
@app.get("/info")
async def system_info():
    """اطلاعات سیستم"""
    return {
        "name": "دستیار حقوقی هوشمند",
        "version": "1.0.0",
        "description": "سیستم پاسخگویی هوشمند به سوالات حقوقی در حوزه پژوهش و فناوری",
        "features": [
            "پاسخگویی به سوالات حقوقی",
            "تحلیل اسناد قانونی",
            "استخراج و فرمت‌بندی ارجاعات",
            "مقایسه اسناد",
            "بررسی قراردادها"
        ],
        "endpoints": {
            "ask": "پرسش اصلی",
            "health": "بررسی سلامت",
            "stats": "آمار عملکرد",
            "test": "تست سریع",
            "cache": "مدیریت کش"
        },
        "supported_document_types": [
            "قانون",
            "آیین_نامه", 
            "دستورالعمل",
            "قرارداد"
        ]
    }

# مدیریت خطاها
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """مدیریت سفارشی خطاهای HTTP"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "path": str(request.url)
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """مدیریت خطاهای سرور"""
    logger.error(f"خطای سرور: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "خطای داخلی سرور",
            "status_code": 500,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
    )

# اجرای سرور برای تست
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 راه‌اندازی سرور دستیار حقوقی...")
    print("📋 در صورت موفقیت:")
    print("   • API Docs: http://localhost:8000/docs")
    print("   • Health Check: http://localhost:8000/health")
    print("   • System Info: http://localhost:8000/info")
    print("   • Test: http://localhost:8000/test")
    
    uvicorn.run(
        "fastapi_endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )