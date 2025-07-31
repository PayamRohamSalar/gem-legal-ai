"""
src/evaluation/phase4_evaluation_system.py - سیستم یکپارچه ارزیابی فاز 4

این فایل تمام اجزای فاز 4 را ادغام می‌کند و یک سیستم کامل ارزیابی و بهینه‌سازی ارائه می‌دهد.
"""

import json
import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import statistics
import sys

# اضافه کردن مسیر src
current_dir = Path(__file__).parent.parent.parent
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

# Import اجزای فاز 4
from evaluation.dataset_generator import LegalDatasetGenerator, TestQuestion, QuestionDifficulty, QuestionCategory
from evaluation.metrics_calculator import LegalMetricsCalculator, OverallMetrics
from evaluation.optimization_manager import LegalSystemOptimizer, OptimizationConfig, OptimizationResult

# Import اجزای سیستم اصلی
from generation.integrated_response_system import LegalResponseSystem, ResponseRequest
from generation.llm_manager import LLMManager, create_model_configs

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """تنظیمات ارزیابی"""
    dataset_size: int = 100
    test_subset_size: int = 20  # برای تست سریع
    use_ollama: bool = True
    preferred_model: str = 'qwen_7b'
    timeout_per_question: int = 30  # ثانیه
    parallel_evaluations: int = 1
    save_detailed_results: bool = True
    
@dataclass
class QuestionEvaluation:
    """نتیجه ارزیابی یک سوال"""
    question_id: str
    question: str
    expected_answer: str
    generated_answer: str
    overall_metrics: OverallMetrics
    processing_time: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class SystemEvaluation:
    """نتیجه کامل ارزیابی سیستم"""
    config: EvaluationConfig
    total_questions: int
    successful_evaluations: int
    failed_evaluations: int
    overall_score: float
    detailed_metrics: Dict[str, float]
    question_results: List[QuestionEvaluation]
    evaluation_time: float
    timestamp: str

class Phase4EvaluationSystem:
    """سیستم یکپارچه ارزیابی فاز 4"""
    
    def __init__(self, output_dir: str = "data/phase4_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # اجزای اصلی سیستم
        self.dataset_generator = LegalDatasetGenerator()
        self.metrics_calculator = LegalMetricsCalculator()
        self.legal_system: Optional[LegalResponseSystem] = None
        self.optimizer: Optional[LegalSystemOptimizer] = None
        
        # تنظیمات
        self.current_config: Optional[EvaluationConfig] = None
        self.evaluation_history: List[SystemEvaluation] = []
        
        logger.info("Phase4EvaluationSystem راه‌اندازی شد")
    
    async def initialize_system(self, config: EvaluationConfig) -> bool:
        """راه‌اندازی سیستم حقوقی"""
        
        self.current_config = config
        
        try:
            # ایجاد سیستم حقوقی
            llm_configs = create_model_configs()
            self.legal_system = LegalResponseSystem(
                llm_configs=llm_configs,
                default_model=config.preferred_model
            )
            
            # راه‌اندازی سیستم
            success = await self.legal_system.initialize()
            if not success:
                logger.error("خطا در راه‌اندازی سیستم حقوقی")
                return False
            
            # بارگذاری مدل مناسب
            if config.use_ollama:
                model_loaded = self.legal_system.llm_manager.load_model(
                    config.preferred_model, 
                    prefer_ollama=True
                )
            else:
                model_loaded = self.legal_system.llm_manager.load_model_mock(
                    config.preferred_model
                )
            
            if not model_loaded:
                logger.warning("خطا در بارگذاری مدل. استفاده از Mock")
                self.legal_system.llm_manager.load_model_mock(config.preferred_model)
            
            logger.info("سیستم حقوقی آماده شد")
            return True
            
        except Exception as e:
            logger.error(f"خطا در راه‌اندازی: {e}")
            return False
    
    def generate_test_dataset(self, config: EvaluationConfig) -> List[TestQuestion]:
        """تولید dataset تست"""
        
        logger.info(f"تولید {config.dataset_size} سوال تست...")
        
        # توزیع متعادل برای تست
        difficulty_distribution = {
            QuestionDifficulty.BASIC: 0.5,        # 50% ساده برای شروع
            QuestionDifficulty.INTERMEDIATE: 0.3,  # 30% متوسط
            QuestionDifficulty.ADVANCED: 0.2       # 20% پیشرفته
        }
        
        category_distribution = {
            QuestionCategory.FACULTY_DUTIES: 0.4,
            QuestionCategory.KNOWLEDGE_BASED_COMPANIES: 0.3,
            QuestionCategory.TECHNOLOGY_TRANSFER: 0.2,
            QuestionCategory.RESEARCH_CONTRACTS: 0.1
        }
        
        questions = self.dataset_generator.generate_dataset(
            total_questions=config.dataset_size,
            difficulty_distribution=difficulty_distribution,
            category_distribution=category_distribution
        )
        
        # ذخیره dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_path = self.dataset_generator.save_dataset(
            questions, 
            f"phase4_dataset_{timestamp}.json"
        )
        
        logger.info(f"Dataset ذخیره شد: {dataset_path}")
        
        return questions
    
    async def evaluate_single_question(
        self, 
        question: TestQuestion
    ) -> QuestionEvaluation:
        """ارزیابی یک سوال"""
        
        start_time = time.time()
        
        try:
            # آماده‌سازی context های مربوطه
            contexts = self._prepare_question_contexts(question)
            
            # ایجاد درخواست
            request = ResponseRequest(
                question=question.question,
                contexts=contexts,
                include_citations=True,
                temperature=0.1
            )
            
            # تولید پاسخ با timeout
            response_result = await asyncio.wait_for(
                self.legal_system.generate_response(request),
                timeout=self.current_config.timeout_per_question
            )
            
            processing_time = time.time() - start_time
            
            if not response_result.success:
                return QuestionEvaluation(
                    question_id=question.id,
                    question=question.question,
                    expected_answer=question.expected_answer,
                    generated_answer="",
                    overall_metrics=None,
                    processing_time=processing_time,
                    success=False,
                    error_message=response_result.error_message
                )
            
            # محاسبه معیارهای ارزیابی
            overall_metrics = self.metrics_calculator.calculate_overall_metrics(
                generated_text=response_result.enhanced_response,
                reference_text=question.expected_answer,
                retrieved_docs=contexts,
                relevant_docs=question.context_needed,
                expected_citations=question.relevant_articles,
                source_contexts=[ctx['content'] for ctx in contexts],
                expected_elements=question.keywords
            )
            
            return QuestionEvaluation(
                question_id=question.id,
                question=question.question,
                expected_answer=question.expected_answer,
                generated_answer=response_result.enhanced_response,
                overall_metrics=overall_metrics,
                processing_time=processing_time,
                success=True
            )
            
        except asyncio.TimeoutError:
            return QuestionEvaluation(
                question_id=question.id,
                question=question.question,
                expected_answer=question.expected_answer,
                generated_answer="",
                overall_metrics=None,
                processing_time=time.time() - start_time,
                success=False,
                error_message="Timeout"
            )
            
        except Exception as e:
            return QuestionEvaluation(
                question_id=question.id,
                question=question.question,
                expected_answer=question.expected_answer,
                generated_answer="",
                overall_metrics=None,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _prepare_question_contexts(self, question: TestQuestion) -> List[Dict[str, Any]]:
        """آماده‌سازی context برای سوال"""
        
        # Context های نمونه بر اساس دسته سوال
        context_templates = {
            QuestionCategory.FACULTY_DUTIES: [
                {
                    "content": "اعضای هیئت علمی موظف به انجام پژوهش و تحقیق در زمینه تخصصی خود هستند",
                    "source": "قانون مقررات انتظامی هیئت علمی - ماده 3",
                    "document_type": "قانون",
                    "article_number": "3",
                    "relevance_score": 0.95
                }
            ],
            QuestionCategory.KNOWLEDGE_BASED_COMPANIES: [
                {
                    "content": "شرکت‌های دانش‌بنیان شرکت‌هایی هستند که بر پایه دانش فنی بروز فعالیت می‌کنند",
                    "source": "قانون حمایت از شرکت‌های دانش‌بنیان - ماده 1",
                    "document_type": "قانون",
                    "article_number": "1",
                    "relevance_score": 1.0
                }
            ],
            QuestionCategory.TECHNOLOGY_TRANSFER: [
                {
                    "content": "انتقال فناوری فرآیند انتقال دانش فنی از مراکز تولیدکننده علم به بخش‌های کاربردی است",
                    "source": "آیین‌نامه انتقال فناوری - ماده 2",
                    "document_type": "آیین_نامه",
                    "article_number": "2",
                    "relevance_score": 0.9
                }
            ]
        }
        
        # انتخاب context مناسب
        return context_templates.get(question.category, [
            {
                "content": "متن عمومی مرتبط با سوال",
                "source": "منبع عمومی",
                "document_type": "قانون",
                "relevance_score": 0.7
            }
        ])
    
    async def run_full_evaluation(
        self, 
        config: EvaluationConfig,
        questions: List[TestQuestion] = None
    ) -> SystemEvaluation:
        """اجرای ارزیابی کامل سیستم"""
        
        logger.info("شروع ارزیابی کامل سیستم")
        start_time = time.time()
        
        # راه‌اندازی سیستم
        if not await self.initialize_system(config):
            raise Exception("خطا در راه‌اندازی سیستم")
        
        # تولید سوالات در صورت عدم وجود
        if questions is None:
            questions = self.generate_test_dataset(config)
        
        # انتخاب subset برای تست
        if config.test_subset_size < len(questions):
            import random
            questions = random.sample(questions, config.test_subset_size)
            logger.info(f"انتخاب {len(questions)} سوال برای تست")
        
        # ارزیابی سوالات
        question_results = []
        successful_count = 0
        
        for i, question in enumerate(questions, 1):
            logger.info(f"ارزیابی سوال {i}/{len(questions)}: {question.question[:50]}...")
            
            result = await self.evaluate_single_question(question)
            question_results.append(result)
            
            if result.success:
                successful_count += 1
                logger.info(f"✅ موفق - امتیاز کلی: {result.overall_metrics.overall_score:.3f}")
            else:
                logger.warning(f"❌ ناموفق - خطا: {result.error_message}")
        
        # محاسبه معیارهای کلی
        evaluation_time = time.time() - start_time
        
        if successful_count > 0:
            # میانگین امتیازات
            overall_score = statistics.mean([
                r.overall_metrics.overall_score for r in question_results if r.success
            ])
            
            # جزئیات معیارها
            detailed_metrics = self._calculate_detailed_metrics(question_results)
        else:
            overall_score = 0.0
            detailed_metrics = {}
        
        # ایجاد نتیجه نهایی
        system_evaluation = SystemEvaluation(
            config=config,
            total_questions=len(questions),
            successful_evaluations=successful_count,
            failed_evaluations=len(questions) - successful_count,
            overall_score=overall_score,
            detailed_metrics=detailed_metrics,
            question_results=question_results,
            evaluation_time=evaluation_time,
            timestamp=datetime.now().isoformat()
        )
        
        # ذخیره نتایج
        if config.save_detailed_results:
            self._save_evaluation_results(system_evaluation)
        
        # اضافه به تاریخچه
        self.evaluation_history.append(system_evaluation)
        
        logger.info(f"ارزیابی کامل شد - امتیاز کلی: {overall_score:.3f}")
        
        return system_evaluation
    
    def _calculate_detailed_metrics(
        self, 
        question_results: List[QuestionEvaluation]
    ) -> Dict[str, float]:
        """محاسبه معیارهای تفصیلی"""
        
        successful_results = [r for r in question_results if r.success and r.overall_metrics]
        
        if not successful_results:
            return {}
        
        metrics = {}
        
        # میانگین معیارهای بازیابی
        metrics['avg_precision_at_5'] = statistics.mean([
            r.overall_metrics.retrieval_metrics.precision_at_5 for r in successful_results
        ])
        
        metrics['avg_recall_at_5'] = statistics.mean([
            r.overall_metrics.retrieval_metrics.recall_at_5 for r in successful_results
        ])
        
        metrics['avg_mrr'] = statistics.mean([
            r.overall_metrics.retrieval_metrics.mrr for r in successful_results
        ])
        
        # میانگین معیارهای تولید
        metrics['avg_rouge_l'] = statistics.mean([
            r.overall_metrics.generation_metrics.rouge_l_f for r in successful_results
        ])
        
        metrics['avg_bleu_4'] = statistics.mean([
            r.overall_metrics.generation_metrics.bleu_4 for r in successful_results
        ])
        
        # میانگین معیارهای حقوقی
        metrics['avg_citation_accuracy'] = statistics.mean([
            r.overall_metrics.legal_accuracy.citation_accuracy for r in successful_results
        ])
        
        metrics['avg_legal_compliance'] = statistics.mean([
            r.overall_metrics.legal_accuracy.law_compliance for r in successful_results
        ])
        
        metrics['avg_completeness'] = statistics.mean([
            r.overall_metrics.legal_accuracy.completeness_score for r in successful_results
        ])
        
        # معیارهای عملکرد
        metrics['avg_processing_time'] = statistics.mean([
            r.processing_time for r in successful_results
        ])
        
        metrics['success_rate'] = len(successful_results) / len(question_results)
        
        return metrics
    
    def _save_evaluation_results(self, evaluation: SystemEvaluation) -> str:
        """ذخیره نتایج ارزیابی"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase4_evaluation_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # تبدیل به dictionary قابل ذخیره
        data = {
            "evaluation_summary": {
                "timestamp": evaluation.timestamp,
                "total_questions": evaluation.total_questions,
                "successful_evaluations": evaluation.successful_evaluations,
                "success_rate": evaluation.successful_evaluations / evaluation.total_questions,
                "overall_score": evaluation.overall_score,
                "evaluation_time": evaluation.evaluation_time
            },
            "config": asdict(evaluation.config),
            "detailed_metrics": evaluation.detailed_metrics,
            "question_results": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "success": r.success,
                    "processing_time": r.processing_time,
                    "overall_score": r.overall_metrics.overall_score if r.overall_metrics else 0.0,
                    "error_message": r.error_message
                }
                for r in evaluation.question_results
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"نتایج ارزیابی ذخیره شد: {filepath}")
        
        return str(filepath)
    
    def setup_optimization(self) -> LegalSystemOptimizer:
        """راه‌اندازی سیستم بهینه‌سازی"""
        
        def evaluation_function(parameters: Dict[str, Any]) -> Dict[str, float]:
            """تابع ارزیابی برای بهینه‌سازی"""
            
            # این تابع باید سیستم را با پارامترهای جدید راه‌اندازی کند
            # و نتایج ارزیابی را برگرداند
            
            # برای مثال، mock evaluation
            import random
            base_score = 0.75
            
            # تأثیر برخی پارامترها
            if parameters.get('chunk_size', 500) in [400, 500, 600]:
                base_score += 0.1
            
            if parameters.get('temperature', 0.1) <= 0.2:
                base_score += 0.1
            
            # افزودن نویز
            noise = random.uniform(-0.05, 0.05)
            
            return {
                'retrieval_precision': min(1.0, base_score + noise),
                'retrieval_recall': min(1.0, base_score + noise * 0.8),
                'generation_rouge': min(1.0, base_score + noise * 1.2),
                'legal_accuracy': min(1.0, base_score + noise * 0.9),
                'citation_quality': min(1.0, base_score + noise * 1.1),
                'response_time': random.uniform(3, 12)
            }
        
        self.optimizer = LegalSystemOptimizer(
            evaluation_function=evaluation_function,
            output_dir=str(self.output_dir / "optimization")
        )
        
        return self.optimizer
    
    def generate_phase4_report(self) -> Dict[str, Any]:
        """تولید گزارش کامل فاز 4"""
        
        if not self.evaluation_history:
            return {"message": "هیچ ارزیابی انجام نشده"}
        
        latest_evaluation = self.evaluation_history[-1]
        
        report = {
            "phase4_summary": {
                "completion_date": datetime.now().isoformat(),
                "total_evaluations": len(self.evaluation_history),
                "latest_overall_score": latest_evaluation.overall_score,
                "system_status": "آماده برای فاز 5" if latest_evaluation.overall_score > 0.7 else "نیاز به بهبود"
            },
            
            "dataset_statistics": {
                "total_questions_generated": latest_evaluation.total_questions,
                "success_rate": latest_evaluation.successful_evaluations / latest_evaluation.total_questions,
                "average_processing_time": latest_evaluation.detailed_metrics.get('avg_processing_time', 0)
            },
            
            "performance_metrics": latest_evaluation.detailed_metrics,
            
            "model_performance": {
                "ollama_status": "Active" if self.current_config and self.current_config.use_ollama else "Mock",
                "preferred_model": self.current_config.preferred_model if self.current_config else "unknown"
            },
            
            "optimization_results": self.optimizer.get_optimization_summary() if self.optimizer else {"message": "بهینه‌سازی انجام نشده"},
            
            "recommendations": self._generate_recommendations(latest_evaluation)
        }
        
        return report
    
    def _generate_recommendations(self, evaluation: SystemEvaluation) -> List[str]:
        """تولید توصیه‌ها بر اساس نتایج"""
        
        recommendations = []
        
        if evaluation.overall_score < 0.6:
            recommendations.append("امتیاز کلی پایین است. بررسی prompt templates پیشنهاد می‌شود")
        
        if evaluation.detailed_metrics.get('success_rate', 0) < 0.8:
            recommendations.append("نرخ موفقیت پایین است. بررسی timeout و مدیریت خطا لازم است")
        
        if evaluation.detailed_metrics.get('avg_processing_time', 0) > 10:
            recommendations.append("زمان پردازش بالا است. بهینه‌سازی سیستم بازیابی پیشنهاد می‌شود")
        
        if evaluation.detailed_metrics.get('avg_citation_accuracy', 0) < 0.7:
            recommendations.append("دقت ارجاعات پایین است. بهبود citation engine لازم است")
        
        if evaluation.detailed_metrics.get('avg_legal_compliance', 0) < 0.8:
            recommendations.append("انطباق حقوقی پایین است. بازنگری prompt templates حقوقی لازم است")
        
        if not recommendations:
            recommendations.append("عملکرد سیستم قابل قبول است. آماده برای فاز 5")
        
        return recommendations

# اجرای کامل فاز 4
async def run_phase4_complete_evaluation():
    """اجرای کامل ارزیابی فاز 4"""
    
    print("🚀 شروع ارزیابی کامل فاز 4")
    print("=" * 60)
    
    # تنظیمات ارزیابی
    config = EvaluationConfig(
        dataset_size=30,      # تعداد کم برای تست
        test_subset_size=10,  # فقط 10 سوال برای تست سریع
        use_ollama=True,      # تلاش برای استفاده از Ollama
        preferred_model='qwen_7b',
        timeout_per_question=15,
        save_detailed_results=True
    )
    
    # ایجاد سیستم ارزیابی
    evaluation_system = Phase4EvaluationSystem()
    
    try:
        # اجرای ارزیابی
        result = await evaluation_system.run_full_evaluation(config)
        
        print(f"\n📊 نتایج ارزیابی:")
        print(f"   • تعداد سوالات: {result.total_questions}")
        print(f"   • موفقیت‌ها: {result.successful_evaluations}")
        print(f"   • نرخ موفقیت: {result.successful_evaluations/result.total_questions*100:.1f}%")
        print(f"   • امتیاز کلی: {result.overall_score:.3f}")
        print(f"   • زمان کل: {result.evaluation_time:.1f} ثانیه")
        
        if result.detailed_metrics:
            print(f"\n📈 معیارهای تفصیلی:")
            for metric, value in result.detailed_metrics.items():
                print(f"   • {metric}: {value:.3f}")
        
        # تولید گزارش کامل
        report = evaluation_system.generate_phase4_report()
        
        print(f"\n🎯 وضعیت فاز 4: {report['phase4_summary']['system_status']}")
        
        if report.get('recommendations'):
            print(f"\n💡 توصیه‌ها:")
            for rec in report['recommendations']:
                print(f"   • {rec}")
        
        return result, report
        
    except Exception as e:
        print(f"❌ خطا در ارزیابی: {e}")
        return None, None

# تست
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # اجرای تست
    asyncio.run(run_phase4_complete_evaluation())