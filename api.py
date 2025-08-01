# api.py - FastAPI Backend for Legal AI Assistant

from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
import uuid
from typing import Dict, List, Any, Optional

# اضافه کردن مسیر src به پایتون
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# ایمپورت سیستم اصلی
try:
    from core.assistant import SimpleLegalAssistant
    from generation.prompt_engine import QueryType
    # ایمپورت کردن مدل‌های داده‌ای از فایل اصلی
    from core.assistant import ResponseRequest, ResponseResult, ContextInfo
except ImportError as e:
    print(f"❌ خطا در import: {e}. مطمئن شوید ساختار پوشه صحیح است.")
    sys.exit(1)


# --- راه‌اندازی ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="⚖️ دستیار حقوقی هوشمند API",
    description="یک API کامل برای پشتیبانی از دستیار حقوقی هوشمند در حوزه پژوهش و فناوری.",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- مدیریت سیستم و Session ---
assistant: Optional[SimpleLegalAssistant] = None
active_sessions: Dict[str, Any] = {}

@app.on_event("startup")
async def startup_event():
    global assistant
    logger.info("🚀 در حال راه‌اندازی سیستم دستیار حقوقی...")
    assistant = SimpleLegalAssistant()
    if not await assistant.setup():
        logger.error("🚨 راه‌اندازی دستیار با شکست مواجه شد!")
    else:
        logger.info("✅ دستیار حقوقی با موفقیت راه‌اندازی شد.")

@app.on_event("shutdown")
def shutdown_event():
    if assistant:
        assistant.shutdown()
        logger.info("👋 سیستم دستیار حقوقی خاموش شد.")

def get_assistant() -> SimpleLegalAssistant:
    if not assistant or not assistant.initialized:
        raise HTTPException(status_code=503, detail="سرویس در حال حاضر در دسترس نیست.")
    return assistant

# --- Endpoints ---

@app.post("/session/create", tags=["Session"])
def create_session():
    """ایجاد یک session جدید برای کاربر."""
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {"created_at": time.time(), "history": []}
    logger.info(f"Session جدید ایجاد شد: {session_id}")
    return {"success": True, "session_id": session_id}

@app.get("/health", tags=["Monitoring"])
def health_check():
    """بررسی وضعیت سلامت API و سرویس‌های زیربنایی."""
    ollama_status = "فعال" if assistant and assistant.initialized else "غیرفعال"
    return {
        "status": "ok",
        "api_version": app.version,
        "ollama_status": ollama_status,
        "active_sessions": len(active_sessions),
    }

@app.post("/ask/enhanced", tags=["Core"])
async def ask_enhanced(req: ResponseRequest, assist: SimpleLegalAssistant = Depends(get_assistant)):
    """ارسال درخواست پیشرفته و دریافت پاسخ کامل با تمام جزئیات."""
    try:
        result = await assist.ask(req)
        return result
    except Exception as e:
        logger.error(f"خطا در حین پردازش /ask/enhanced: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطای داخلی سرور: {str(e)}")

@app.post("/upload/analyze", tags=["File Processing"])
async def analyze_document(file: UploadFile = File(...)):
    """آپلود و تحلیل یک سند (نسخه نمایشی)."""
    # در نسخه واقعی، این بخش باید فایل را پردازش کرده و تحلیل واقعی انجام دهد.
    # در حال حاضر یک پاسخ شبیه‌سازی شده برمی‌گردانیم.
    logger.info(f"فایل دریافت شد: {file.filename}, نوع: {file.content_type}")
    return {
        "success": True,
        "filename": file.filename,
        "analysis": {
            "document_type": "قرارداد پژوهشی",
            "structure_score": 0.85,
            "compliance_score": 0.78,
            "recommendations": [
                "بند مالکیت فکری نیاز به شفاف‌سازی دارد.",
                "شرایط فسخ قرارداد باید دقیق‌تر مشخص شود."
            ]
        },
        "extracted_sections": ["ماده ۱: طرفین قرارداد", "ماده ۲: موضوع قرارداد", "ماده ۵: مالکیت فکری"]
    }

@app.post("/batch/process", tags=["Batch Processing"])
async def batch_process(data: Dict[str, List[str]], assist: SimpleLegalAssistant = Depends(get_assistant)):
    """پردازش دسته‌ای سوالات (نسخه نمایشی)."""
    questions = data.get("questions", [])
    if not questions:
        raise HTTPException(status_code=400, detail="لیست سوالات خالی است.")

    start_time = time.time()
    successful_count = 0
    results = []

    for i, q in enumerate(questions):
        # شبیه‌سازی پردازش
        await asyncio.sleep(0.5)
        results.append({
            "question_index": i,
            "question": q,
            "success": True,
            "quality_score": 0.8 + (i % 3) * 0.05, # امتیاز متغیر
            "processing_time": 1.5 + (i % 4) * 0.2,
            "error_message": None
        })
        successful_count += 1

    total_time = time.time() - start_time
    return {
        "success": True,
        "total_questions": len(questions),
        "successful_questions": successful_count,
        "success_rate": (successful_count / len(questions)) * 100,
        "average_time_per_question": total_time / len(questions),
        "results": results,
    }

@app.get("/stats/system", tags=["Monitoring"])
def get_system_stats(assist: SimpleLegalAssistant = Depends(get_assistant)):
    """دریافت آمار جامع عملکرد سیستم."""
    return {
        "system": assist.get_stats(),
        "sessions": {"active_sessions": len(active_sessions)}
    }

@app.delete("/cache/clear", tags=["Admin"])
def clear_system_cache(assist: SimpleLegalAssistant = Depends(get_assistant)):
    """پاک کردن کش داخلی سیستم."""
    assist.clear_cache()
    return {"success": True, "message": "کش سیستم با موفقیت پاک شد."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)