# run_assistant.py
import asyncio
import sys
import os

# افزودن مسیر پروژه به sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ایمپورت کردن ماژول‌های لازم
from src.retrieval.vector_database import LegalVectorDatabase
from src.retrieval.embedding_manager import EmbeddingModelManager
from src.retrieval.hybrid_search import HybridSearchEngine
from src.generation.integrated_response_system import LegalResponseSystem, ResponseRequest, QueryType

# --- تنظیمات ---
DB_PATH = "data/vector_db"
COLLECTION_NAME = "legal_hybrid_v1"
ACTIVE_LLM_KEY = "qwen"  # <-- اصلاح شد: از کلید کوتاه استفاده می‌کنیم

async def main():
    """تابع اصلی برای اجرای کامل و یکپارچه دستیار حقوقی"""
    print("🚀=============== INITIALIZING LEGAL AI ASSISTANT ===============🚀")
    
    # === راه‌اندازی موتور جستجو ===
    vector_db = LegalVectorDatabase(db_path=DB_PATH, collection_name=COLLECTION_NAME)
    embedding_manager = EmbeddingModelManager()
    embedding_manager.load_model(embedding_manager.get_recommended_model())
    search_engine = HybridSearchEngine(vector_db, embedding_manager)
    print("✅ Search Engine is ready.")
    
    # === راه‌اندازی سیستم پاسخگویی ===
    response_system = LegalResponseSystem(default_model=ACTIVE_LLM_KEY)
    from src.generation.llm_manager import create_model_configs
    response_system.llm_manager.configs = create_model_configs()
    
    # بررسی و تنظیم مدل فعال
    if not response_system.llm_manager.set_active_model(ACTIVE_LLM_KEY):
        print(f"❌ ERROR: Could not set active model to '{ACTIVE_LLM_KEY}'. Please check configs.")
        return
    print("✅ Response System is ready.")

    # === طرح سوال و اجرای کامل خط لوله ===
    user_question = "اهداف سند گسترش کاربرد فناوری نانو در افق ١٤٠٤ چیست؟"
    print(f"\n--- Processing question: '{user_question}' ---")

    # ۱. جستجو برای یافتن context
    contexts = search_engine.hybrid_search(user_question, top_k=7)
    if not contexts:
        print("❌ No relevant documents found.")
        return

    # ۲. نمایش متن کامل chunkهای بازیابی شده
    print("\n--- Inspecting FULL Retrieved Context ---")
    print("==============================")
    for i, ctx in enumerate(contexts, 1):
        print(f"--- CHUNK #{i} | Score: {ctx.get('final_score', 0):.3f} | Source: {ctx['metadata'].get('document_title', 'N/A')} ---")
        print(ctx['text'])
        print("-" * 25)
    print("==============================")
    
    # ۳. ساخت درخواست برای سیستم پاسخگویی
    request = ResponseRequest(
        question=user_question,
        contexts=[{'content': ctx['text'], 'source': ctx['metadata'].get('document_title', '')} for ctx in contexts]
    )

    # ۴. تولید پاسخ نهایی
    result = await response_system.generate_response(request)

    # ۵. نمایش نتیجه
    if result.success:
        print("\n\n✅✅✅ FINAL ENHANCED RESPONSE ✅✅✅")
        print("=" * 50)
        print("✍️ **پاسخ نهایی دستیار حقوقی:**\n")
        print(result.enhanced_response)
        print("\n" + "=" * 50)
        print(result.references_list)
        print("=" * 50)
        print(f"⭐ امتیاز کیفیت پاسخ: {result.quality_score:.1f}/100")
        print(f"🎯 اطمینان از صحت پاسخ: {result.confidence_score*100:.1f}%")
        print(f"⏱️ زمان پردازش: {result.processing_time:.2f} ثانیه")
    else:
        print(f"\n❌ ERROR: {result.error_message}")

    print("\n🚀=============== ASSISTANT RUN COMPLETE ===============🚀")

if __name__ == "__main__":
    asyncio.run(main())