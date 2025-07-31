# scripts/run_evaluation.py
import asyncio
import logging
import sys
from pathlib import Path

# افزودن مسیر ریشه پروژه برای ایمپورت صحیح ماژول‌ها
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from src.evaluation.evaluation_system import Phase4EvaluationSystem, EvaluationConfig

async def main():
    """اجرای کامل ارزیابی فاز 4"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🚀=============== START: PHASE 4 - FULL SYSTEM EVALUATION ===============🚀")
    
    # تنظیمات ارزیابی
    config = EvaluationConfig(
        dataset_size=30,      # تعداد کل سوالات برای تولید
        test_subset_size=10,  # ارزیابی فقط روی 10 سوال برای سرعت
        use_ollama=True,
        preferred_model='mistral_7b', # استفاده از کلیدهای کوتاه تعریف شده
        timeout_per_question=45,
        save_detailed_results=True
    )
    
    evaluation_system = Phase4EvaluationSystem()
    
    try:
        # اجرای کامل ارزیابی
        result, report = await evaluation_system.run_full_evaluation(config)
        
        if result:
            print(f"\n📊=============== EVALUATION SUMMARY ===============📊")
            print(f"   • امتیاز کلی سیستم: {result.overall_score:.3f}")
            print(f"   • نرخ موفقیت: {result.successful_evaluations}/{result.total_questions} ({result.successful_evaluations/result.total_questions*100:.1f}%)")
            print(f"   • میانگین زمان پاسخ: {result.detailed_metrics.get('avg_processing_time', 0):.2f} ثانیه")
            
            print(f"\n📈 جزئیات معیارها:")
            print(f"   • دقت بازیابی (Precision@5): {result.detailed_metrics.get('avg_precision_at_5', 0):.3f}")
            print(f"   • امتیاز تولید (ROUGE-L): {result.detailed_metrics.get('avg_rouge_l', 0):.3f}")
            print(f"   • دقت ارجاعات: {result.detailed_metrics.get('avg_citation_accuracy', 0):.3f}")
            
            print(f"\n💡 توصیه‌ها:")
            for rec in report['recommendations']:
                print(f"   • {rec}")
        
    except Exception as e:
        logging.error(f"❌ خطای اصلی در اجرای ارزیابی: {e}", exc_info=True)

    print("\n🚀=============== END: PHASE 4 EVALUATION ===============🚀")

if __name__ == "__main__":
    asyncio.run(main())