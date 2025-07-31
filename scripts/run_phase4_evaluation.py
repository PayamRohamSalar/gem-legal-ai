#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_phase4_evaluation.py - اجرای کامل ارزیابی فاز 4

این فایل مسئول اجرای کامل و هماهنگ تمام اجزای فاز 4 است:
- تست اتصال Ollama
- تولید Dataset
- اجرای ارزیابی کامل
- بهینه‌سازی (اختیاری)
- تولید گزارش نهایی

استفاده:
    python run_phase4_evaluation.py --mode full
    python run_phase4_evaluation.py --mode quick
    python run_phase4_evaluation.py --mode test-only
"""

import sys
import os
import argparse
import asyncio
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# تنظیمات فارسی
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# اضافه کردن مسیر src
current_dir = Path(__file__).parent
src_path = current_dir / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Import اجزای سیستم
try:
    from generation.llm_manager import LLMManager, create_model_configs
    from evaluation.dataset_generator import LegalDatasetGenerator, QuestionDifficulty, QuestionCategory
    from evaluation.phase4_evaluation_system import Phase4EvaluationSystem, EvaluationConfig
    from evaluation.optimization_manager import LegalSystemOptimizer, OptimizationConfig
    from evaluation.metrics_calculator import LegalMetricsCalculator
    SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"❌ خطا در import سیستم: {e}")
    SYSTEM_AVAILABLE = False

# تنظیم لاگ‌گیری
def setup_logging(log_level: str = "INFO") -> None:
    """تنظیم سیستم لاگ‌گیری"""
    
    # ایجاد پوشه logs
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # فایل لاگ با timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"phase4_evaluation_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    print(f"📋 Log file: {log_file}")

class Phase4Runner:
    """کلاس اصلی اجرای فاز 4"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.start_time = datetime.now()
        self.results = {}
        
        # ایجاد پوشه‌های مورد نیاز
        self.setup_directories()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Phase4Runner راه‌اندازی شد")
    
    def setup_directories(self) -> None:
        """ایجاد پوشه‌های مورد نیاز"""
        
        directories = [
            "data/test_dataset",
            "data/phase4_evaluation", 
            "data/optimization_results",
            "logs",
            "reports"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    async def test_ollama_connection(self) -> bool:
        """تست اتصال Ollama"""
        
        print("\n🔍 تست اتصال Ollama...")
        print("=" * 50)
        
        try:
            configs = create_model_configs()
            manager = LLMManager(configs)
            
            # نمایش مدل‌های موجود
            available = manager.list_available_models()
            print(f"📋 وضعیت Ollama: {available['ollama_status']}")
            print(f"📦 مدل‌های موجود: {available['ollama_models']}")
            
            # تست بارگذاری مدل
            preferred_model = self.config.get('preferred_model', 'qwen_7b')
            print(f"\n🔄 تست بارگذاری مدل: {preferred_model}")
            
            success = manager.load_model(preferred_model, prefer_ollama=True)
            
            if success:
                print("✅ مدل بارگذاری شد")
                
                # تست تولید
                test_prompts = [
                    "سلام",
                    "وظایف هیئت علمی چیست؟"
                ]
                
                for prompt in test_prompts:
                    print(f"\n🧪 تست: {prompt}")
                    
                    start = time.time()
                    result = manager.generate_response(prompt)
                    duration = time.time() - start
                    
                    if result['success']:
                        response = result['response'][:100] + "..." if len(result['response']) > 100 else result['response']
                        print(f"✅ پاسخ ({duration:.2f}s): {response}")
                        
                        # متریک‌ها
                        metrics = result.get('metrics')
                        if metrics:
                            print(f"📊 توکن‌ها: {metrics.input_tokens}→{metrics.output_tokens}")
                    else:
                        print(f"❌ خطا: {result['error']}")
                        return False
                
                self.results['ollama_test'] = {
                    'status': 'success',
                    'model': preferred_model,
                    'available_models': available['ollama_models']
                }
                
                return True
            else:
                print("❌ خطا در بارگذاری مدل")
                self.results['ollama_test'] = {'status': 'failed', 'reason': 'model_load_failed'}
                return False
                
        except Exception as e:
            print(f"❌ خطا در تست Ollama: {e}")
            self.results['ollama_test'] = {'status': 'failed', 'reason': str(e)}
            return False
    
    def generate_test_dataset(self) -> Optional[List]:
        """تولید Dataset تست"""
        
        print("\n📊 تولید Dataset تست...")
        print("=" * 50)
        
        try:
            generator = LegalDatasetGenerator()
            
            dataset_size = self.config.get('dataset_size', 50)
            print(f"🎯 تولید {dataset_size} سوال...")
            
            # توزیع متعادل
            difficulty_distribution = {
                QuestionDifficulty.BASIC: 0.4,
                QuestionDifficulty.INTERMEDIATE: 0.4,
                QuestionDifficulty.ADVANCED: 0.2
            }
            
            category_distribution = {
                QuestionCategory.FACULTY_DUTIES: 0.4,
                QuestionCategory.KNOWLEDGE_BASED_COMPANIES: 0.3,
                QuestionCategory.TECHNOLOGY_TRANSFER: 0.2,
                QuestionCategory.RESEARCH_CONTRACTS: 0.1
            }
            
            questions = generator.generate_dataset(
                total_questions=dataset_size,
                difficulty_distribution=difficulty_distribution,
                category_distribution=category_distribution
            )
            
            # ذخیره dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = generator.save_dataset(
                questions, 
                f"phase4_dataset_{timestamp}.json"
            )
            
            # آمار dataset
            stats = generator.get_dataset_stats(questions)
            
            print(f"✅ Dataset تولید شد: {len(questions)} سوال")
            print(f"💾 ذخیره در: {filepath}")
            print(f"📊 آمار:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"   • {key}:")
                    for k, v in value.items():
                        print(f"     - {k}: {v}")
                else:
                    print(f"   • {key}: {value}")
            
            self.results['dataset_generation'] = {
                'status': 'success',
                'questions_count': len(questions),
                'filepath': str(filepath),
                'stats': stats
            }
            
            return questions
            
        except Exception as e:
            print(f"❌ خطا در تولید Dataset: {e}")
            self.results['dataset_generation'] = {'status': 'failed', 'reason': str(e)}
            return None
    
    async def run_system_evaluation(self, questions: List = None) -> Optional[Dict]:
        """اجرای ارزیابی کامل سیستم"""
        
        print("\n🔬 اجرای ارزیابی کامل سیستم...")
        print("=" * 50)
        
        try:
            # تنظیمات ارزیابی
            eval_config = EvaluationConfig(
                dataset_size=self.config.get('dataset_size', 50),
                test_subset_size=self.config.get('test_subset_size', 10),
                use_ollama=self.config.get('use_ollama', True),
                preferred_model=self.config.get('preferred_model', 'qwen_7b'),
                timeout_per_question=self.config.get('timeout_per_question', 30),
                save_detailed_results=True
            )
            
            print(f"⚙️ تنظیمات ارزیابی:")
            print(f"   • تعداد سوالات تست: {eval_config.test_subset_size}")
            print(f"   • استفاده از Ollama: {eval_config.use_ollama}")
            print(f"   • مدل ترجیحی: {eval_config.preferred_model}")
            print(f"   • Timeout: {eval_config.timeout_per_question}s")
            
            # ایجاد سیستم ارزیابی
            evaluation_system = Phase4EvaluationSystem()
            
            # اجرای ارزیابی
            print(f"\n🚀 شروع ارزیابی...")
            evaluation_result = await evaluation_system.run_full_evaluation(
                eval_config, questions
            )
            
            # نمایش نتایج
            print(f"\n📈 نتایج ارزیابی:")
            print(f"   • تعداد سوالات: {evaluation_result.total_questions}")
            print(f"   • موفقیت‌ها: {evaluation_result.successful_evaluations}")
            print(f"   • نرخ موفقیت: {evaluation_result.successful_evaluations/evaluation_result.total_questions*100:.1f}%")
            print(f"   • امتیاز کلی: {evaluation_result.overall_score:.3f}")
            print(f"   • زمان کل: {evaluation_result.evaluation_time:.1f} ثانیه")
            
            if evaluation_result.detailed_metrics:
                print(f"\n📊 معیارهای تفصیلی:")
                for metric, value in evaluation_result.detailed_metrics.items():
                    print(f"   • {metric}: {value:.3f}")
            
            # ارزیابی وضعیت
            status = "عالی" if evaluation_result.overall_score > 0.8 else \
                     "خوب" if evaluation_result.overall_score > 0.6 else "نیازمند بهبود"
            print(f"\n🎯 وضعیت سیستم: {status}")
            
            self.results['system_evaluation'] = {
                'status': 'success',
                'overall_score': evaluation_result.overall_score,
                'success_rate': evaluation_result.successful_evaluations / evaluation_result.total_questions,
                'total_time': evaluation_result.evaluation_time,
                'detailed_metrics': evaluation_result.detailed_metrics,
                'system_status': status
            }
            
            return evaluation_result
            
        except Exception as e:
            print(f"❌ خطا در ارزیابی سیستم: {e}")
            self.results['system_evaluation'] = {'status': 'failed', 'reason': str(e)}
            return None
    
    def run_optimization(self) -> Optional[Dict]:
        """اجرای بهینه‌سازی (اختیاری)"""
        
        if not self.config.get('run_optimization', False):
            print("\n⏭️ بهینه‌سازی رد شد (تنظیمات)")
            return None
        
        print("\n⚙️ اجرای بهینه‌سازی...")
        print("=" * 50)
        
        try:
            # تابع ارزیابی Mock برای مثال
            def evaluation_function(parameters: Dict[str, Any]) -> Dict[str, float]:
                import random
                import time
                
                # شبیه‌سازی ارزیابی
                time.sleep(0.5)
                
                base_score = 0.7
                
                # تأثیر برخی پارامترها
                if parameters.get('chunk_size', 500) in [400, 500, 600]:
                    base_score += 0.1
                
                if parameters.get('temperature', 0.1) <= 0.2:
                    base_score += 0.1
                
                # نویز تصادفی
                noise = random.uniform(-0.1, 0.1)
                
                return {
                    'retrieval_precision': min(1.0, base_score + noise),
                    'retrieval_recall': min(1.0, base_score + noise * 0.8),
                    'generation_rouge': min(1.0, base_score + noise * 1.2),
                    'legal_accuracy': min(1.0, base_score + noise * 0.9),
                    'citation_quality': min(1.0, base_score + noise * 1.1),
                    'response_time': random.uniform(3, 12)
                }
            
            # ایجاد optimizer
            optimizer = LegalSystemOptimizer(evaluation_function)
            
            # تنظیمات بهینه‌سازی
            opt_config = OptimizationConfig(
                max_iterations=self.config.get('optimization_iterations', 20),
                early_stopping_patience=5,
                n_random_starts=3
            )
            
            print(f"🔧 تنظیمات بهینه‌سازی:")
            print(f"   • حداکثر تکرار: {opt_config.max_iterations}")
            print(f"   • روش: Random Search")
            
            # اجرای بهینه‌سازی
            print(f"\n🚀 شروع بهینه‌سازی...")
            best_result = optimizer.random_search(opt_config)
            
            print(f"\n🎯 بهترین نتیجه:")
            print(f"   • امتیاز: {best_result.score:.4f}")
            print(f"   • پارامترهای بهینه:")
            for key, value in best_result.parameters.items():
                print(f"     - {key}: {value}")
            
            # خلاصه بهینه‌سازی
            summary = optimizer.get_optimization_summary()
            print(f"\n📊 خلاصه بهینه‌سازی:")
            print(f"   • تعداد ارزیابی‌ها: {summary['total_evaluations']}")
            print(f"   • بهبود کلی: {summary['improvement']:.4f}")
            
            # ذخیره نتایج
            optimizer.save_optimization_results()
            
            self.results['optimization'] = {
                'status': 'success',
                'best_score': best_result.score,
                'best_parameters': best_result.parameters,
                'total_evaluations': summary['total_evaluations'],
                'improvement': summary['improvement']
            }
            
            return best_result
            
        except Exception as e:
            print(f"❌ خطا در بهینه‌سازی: {e}")
            self.results['optimization'] = {'status': 'failed', 'reason': str(e)}
            return None
    
    def generate_final_report(self) -> str:
        """تولید گزارش نهایی فاز 4"""
        
        print("\n📋 تولید گزارش نهایی...")
        print("=" * 50)
        
        total_duration = datetime.now() - self.start_time
        
        report = {
            "phase4_final_report": {
                "execution_date": self.start_time.isoformat(),
                "total_duration": str(total_duration),
                "config": self.config,
                "results": self.results
            },
            
            "summary": {
                "ollama_status": self.results.get('ollama_test', {}).get('status', 'unknown'),
                "dataset_generated": self.results.get('dataset_generation', {}).get('status', 'unknown') == 'success',
                "evaluation_completed": self.results.get('system_evaluation', {}).get('status', 'unknown') == 'success',
                "optimization_run": self.results.get('optimization', {}).get('status', 'unknown') == 'success',
                "overall_success": all(
                    result.get('status') == 'success' 
                    for key, result in self.results.items() 
                    if key != 'optimization'  # optimization اختیاری است
                )
            },
            
            "recommendations": self._generate_recommendations()
        }
        
        # ذخیره گزارش
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path("reports") / f"phase4_final_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # نمایش خلاصه
        summary = report['summary']
        print(f"✅ Ollama: {'فعال' if summary['ollama_status'] == 'success' else 'غیرفعال'}")
        print(f"✅ Dataset: {'تولید شد' if summary['dataset_generated'] else 'خطا'}")
        print(f"✅ ارزیابی: {'انجام شد' if summary['evaluation_completed'] else 'خطا'}")
        print(f"✅ بهینه‌سازی: {'انجام شد' if summary['optimization_run'] else 'رد شد'}")
        
        overall_status = "موفق" if summary['overall_success'] else "نیازمند بررسی"
        print(f"\n🎯 وضعیت کلی فاز 4: {overall_status}")
        
        if report['recommendations']:
            print(f"\n💡 توصیه‌ها:")
            for rec in report['recommendations']:
                print(f"   • {rec}")
        
        print(f"\n📄 گزارش کامل: {report_file}")
        
        return str(report_file)
    
    def _generate_recommendations(self) -> List[str]:
        """تولید توصیه‌ها بر اساس نتایج"""
        
        recommendations = []
        
        # بررسی Ollama
        if self.results.get('ollama_test', {}).get('status') != 'success':
            recommendations.append("نصب و راه‌اندازی Ollama برای عملکرد بهتر")
        
        # بررسی ارزیابی
        eval_result = self.results.get('system_evaluation', {})
        if eval_result.get('status') == 'success':
            score = eval_result.get('overall_score', 0)
            if score < 0.6:
                recommendations.append("بهبود prompt templates و تنظیمات سیستم")
            elif score < 0.8:
                recommendations.append("بهینه‌سازی hyperparameterها برای بهبود عملکرد")
        
        # بررسی بهینه‌سازی
        if not self.results.get('optimization'):
            recommendations.append("اجرای بهینه‌سازی برای تنظیم دقیق‌تر سیستم")
        
        if not recommendations:
            recommendations.append("سیستم آماده ورود به فاز 5 است")
        
        return recommendations
    
    async def run_full_phase4(self) -> bool:
        """اجرای کامل فاز 4"""
        
        print("🚀 شروع اجرای کامل فاز 4")
        print("=" * 60)
        print(f"⏰ زمان شروع: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⚙️ حالت اجرا: {self.config.get('mode', 'full')}")
        
        success = True
        
        # مرحله 1: تست Ollama
        if not await self.test_ollama_connection():
            print("⚠️ Ollama فعال نیست، ادامه با Mock")
        
        # مرحله 2: تولید Dataset
        questions = self.generate_test_dataset()
        if not questions:
            print("❌ خطا در تولید Dataset")
            success = False
        
        # مرحله 3: ارزیابی سیستم
        evaluation_result = await self.run_system_evaluation(questions)
        if not evaluation_result:
            print("❌ خطا در ارزیابی سیستم")
            success = False
        
        # مرحله 4: بهینه‌سازی (اختیاری)
        self.run_optimization()
        
        # مرحله 5: گزارش نهایی
        report_file = self.generate_final_report()
        
        total_duration = datetime.now() - self.start_time
        
        print(f"\n🏁 پایان اجرای فاز 4")
        print("=" * 60)
        print(f"⏱️ مدت کل: {total_duration}")
        print(f"📊 وضعیت: {'✅ موفق' if success else '❌ با خطا'}")
        print(f"📄 گزارش: {report_file}")
        
        return success

def parse_arguments():
    """تجزیه آرگومان‌های خط فرمان"""
    
    parser = argparse.ArgumentParser(description="اجرای فاز 4 دستیار حقوقی هوشمند")
    
    parser.add_argument(
        '--mode',
        choices=['full', 'quick', 'test-only'],
        default='full',
        help='حالت اجرا: full (کامل), quick (سریع), test-only (فقط تست)'
    )
    
    parser.add_argument('--dataset-size', type=int, default=50, help='تعداد سوالات dataset')
    parser.add_argument('--test-subset', type=int, default=10, help='تعداد سوالات تست')
    parser.add_argument('--model', default='qwen_7b', help='مدل مورد استفاده')
    parser.add_argument('--timeout', type=int, default=30, help='timeout هر سوال (ثانیه)')
    parser.add_argument('--optimize', action='store_true', help='اجرای بهینه‌سازی')
    parser.add_argument('--log-level', default='INFO', help='سطح لاگ‌گیری')
    
    return parser.parse_args()

async def main():
    """تابع اصلی"""
    
    if not SYSTEM_AVAILABLE:
        print("❌ سیستم در دسترس نیست. لطفاً مسیر src را بررسی کنید.")
        return False
    
    # تجزیه آرگومان‌ها
    args = parse_arguments()
    
    # تنظیم لاگ‌گیری
    setup_logging(args.log_level)
    
    # تنظیمات بر اساس حالت
    config = {
        'mode': args.mode,
        'dataset_size': 20 if args.mode == 'quick' else args.dataset_size,
        'test_subset_size': 5 if args.mode == 'quick' else args.test_subset,
        'preferred_model': args.model,
        'use_ollama': True,
        'timeout_per_question': args.timeout,
        'run_optimization': args.optimize,
        'optimization_iterations': 10 if args.mode == 'quick' else 20
    }
    
    if args.mode == 'test-only':
        config.update({
            'dataset_size': 10,
            'test_subset_size': 5,
            'run_optimization': False
        })
    
    # اجرای فاز 4
    runner = Phase4Runner(config)
    success = await runner.run_full_phase4()
    
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ اجرا توسط کاربر متوقف شد")
        sys.exit(1)
    except Exception as e:
        print(f"❌ خطای غیرمنتظره: {e}")
        sys.exit(1)