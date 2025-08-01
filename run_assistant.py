# run_assistant.py
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ایمپورت کردن ماژول‌های اصلی
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.vector_database import LegalVectorDatabase
from src.retrieval.embedding_manager import EmbeddingModelManager
from src.generation.llm_manager import LLMManager, create_model_configs
from src.generation.prompt_engine import PromptEngine, ContextInfo, ContextType # <-- اصلاح شد

# --- تنظیمات اصلی ---
DB_PATH = "data/vector_db"
COLLECTION_NAME = "legal_hybrid_v1"
ACTIVE_LLM_KEY = "qwen"

def main():
    print("🚀=============== INITIALIZING LEGAL AI ASSISTANT (Stable Version) ===============🚀")
    
    # === راه‌اندازی موتور جستجو ===
    print("\n--- Initializing Search Engine ---")
    vector_db = LegalVectorDatabase(db_path=DB_PATH, collection_name=COLLECTION_NAME)
    embedding_manager = EmbeddingModelManager()
    embedding_manager.load_model(embedding_manager.get_recommended_model())
    search_engine = HybridSearchEngine(vector_db, embedding_manager)
    print("✅ Search Engine is ready.")

    # === راه‌اندازی مدیر LLM ===
    print("\n--- Initializing LLM Manager ---")
    llm_configs = create_model_configs()
    llm_manager = LLMManager(llm_configs)
    if not llm_manager.set_active_model(ACTIVE_LLM_KEY):
        return
    print("✅ LLM Manager is ready.")

    # === راه‌اندازی موتور پرامپت ===
    prompt_engine = PromptEngine() # <-- اصلاح شد
    print("✅ Prompt Engine is ready.")

    # === طرح سوال و اجرای کامل خط لوله ===
    user_question = "اهداف سند گسترش کاربرد فناوری نانو در افق ١٤٠٤ چیست؟"
    print(f"\n--- Processing question: '{user_question}' ---")

    # ۱. جستجو برای یافتن context
    contexts_raw = search_engine.hybrid_search(user_question, top_k=5)
    if not contexts_raw:
        print("❌ No relevant documents found.")
        return

    # ۲. ساخت پرامپت
    # تبدیل نتایج خام به فرمت ContextInfo
    contexts_info = [ContextInfo(
        content=res['text'],
        source=res['metadata'].get('document_title', 'سند نامشخص'),
        document_type=ContextType.LAW, # برای سادگی، فعلاً همه را قانون در نظر می‌گیریم
        relevance_score=res.get('final_score', 0)
    ) for res in contexts_raw]

    final_prompt, query_type = prompt_engine.build_prompt( # <-- اصلاح شد
        question=user_question,
        contexts=contexts_info
    )
    print(f"✅ Prompt built for query type: {query_type.value}")
    
    # ۳. تولید پاسخ نهایی
    print(f"\n--- Generating response with '{ACTIVE_LLM_KEY}' model ---")
    result = llm_manager.generate_response(final_prompt)
    
    # ۴. نمایش نتیجه
    if result['success']:
        print("\n\n✅✅✅ FINAL RESPONSE ✅✅✅")
        print("=" * 50)
        print("✍️ **پاسخ نهایی دستیار حقوقی:**\n")
        print(result['response'])
        print("\n" + "=" * 50)
    else:
        print(f"\n❌ ERROR: {result['error']}")

    print("\n🚀=============== ASSISTANT RUN COMPLETE ===============🚀")

if __name__ == "__main__":
    main()