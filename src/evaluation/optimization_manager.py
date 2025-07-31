"""
src/evaluation/optimization_manager.py - بهینه‌سازی عملکرد سیستم

این فایل مسئول بهینه‌سازی hyperparameterها و تنظیمات سیستم است:
- تنظیم hyperparameterها (chunk size, overlap, top_k)
- بهبود prompt templates
- Fine-tuning embedding model (در صورت نیاز)
"""

import json
import logging
import itertools
import statistics
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import concurrent.futures
import time

logger = logging.getLogger(__name__)

@dataclass
class HyperParameter:
    """تعریف یک hyperparameter"""
    name: str
    current_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    step: Optional[Any] = None
    possible_values: Optional[List[Any]] = None
    description: str = ""

@dataclass
class OptimizationResult:
    """نتیجه بهینه‌سازی"""
    parameters: Dict[str, Any]
    score: float
    metrics: Dict[str, float]
    evaluation_time: float
    iteration: int

@dataclass
class OptimizationConfig:
    """تنظیمات بهینه‌سازی"""
    max_iterations: int = 50
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.001
    n_random_starts: int = 5
    parallel_evaluations: int = 2
    save_intermediate_results: bool = True

class LegalSystemOptimizer:
    """بهینه‌ساز سیستم حقوقی"""
    
    def __init__(
        self,
        evaluation_function: Callable,
        output_dir: str = "data/optimization_results"
    ):
        self.evaluation_function = evaluation_function
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # تعریف hyperparameterها
        self.hyperparameters = self._define_hyperparameters()
        
        # تاریخچه بهینه‌سازی
        self.optimization_history: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None
        
        logger.info("LegalSystemOptimizer راه‌اندازی شد")
    
    def _define_hyperparameters(self) -> Dict[str, HyperParameter]:
        """تعریف hyperparameterهای سیستم"""
        
        return {
            # پارامترهای Chunking
            "chunk_size": HyperParameter(
                name="chunk_size",
                current_value=500,
                possible_values=[300, 400, 500, 600, 700, 800],
                description="اندازه chunk های متن (کلمه)"
            ),
            
            "chunk_overlap": HyperParameter(
                name="chunk_overlap",
                current_value=50,
                possible_values=[25, 50, 75, 100, 125],
                description="همپوشانی بین chunk ها (کلمه)"
            ),
            
            # پارامترهای Retrieval
            "top_k_retrieval": HyperParameter(
                name="top_k_retrieval",
                current_value=5,
                possible_values=[3, 5, 7, 10, 15],
                description="تعداد اسناد بازیابی شده"
            ),
            
            "similarity_threshold": HyperParameter(
                name="similarity_threshold",
                current_value=0.7,
                possible_values=[0.5, 0.6, 0.7, 0.8, 0.9],
                description="آستانه شباهت برای فیلتر اسناد"
            ),
            
            # پارامترهای Generation
            "temperature": HyperParameter(
                name="temperature",
                current_value=0.1,
                possible_values=[0.0, 0.1, 0.2, 0.3, 0.5],
                description="دمای تولید LLM"
            ),
            
            "max_tokens": HyperParameter(
                name="max_tokens",
                current_value=2048,
                possible_values=[1024, 1536, 2048, 3072, 4096],
                description="حداکثر توکن تولیدی"
            ),
            
            "top_p": HyperParameter(
                name="top_p",
                current_value=0.9,
                possible_values=[0.7, 0.8, 0.9, 0.95, 1.0],
                description="Top-p sampling"
            ),
            
            # پارامترهای Context
            "context_window_size": HyperParameter(
                name="context_window_size",
                current_value=3000,
                possible_values=[2000, 2500, 3000, 3500, 4000],
                description="اندازه پنجره context (کاراکتر)"
            ),
            
            "context_diversity_weight": HyperParameter(
                name="context_diversity_weight",
                current_value=0.3,
                possible_values=[0.1, 0.2, 0.3, 0.4, 0.5],
                description="وزن تنوع در انتخاب context"
            )
        }
    
    def _evaluate_parameters(self, parameters: Dict[str, Any]) -> OptimizationResult:
        """ارزیابی یک set از parameters"""
        
        start_time = time.time()
        
        try:
            # اجرای تابع ارزیابی با پارامترهای جدید
            metrics = self.evaluation_function(parameters)
            
            # محاسبه امتیاز کلی
            score = self._calculate_composite_score(metrics)
            
            evaluation_time = time.time() - start_time
            
            result = OptimizationResult(
                parameters=parameters.copy(),
                score=score,
                metrics=metrics,
                evaluation_time=evaluation_time,
                iteration=len(self.optimization_history) + 1
            )
            
            logger.info(f"ارزیابی {result.iteration}: امتیاز={score:.4f}, زمان={evaluation_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"خطا در ارزیابی parameters: {e}")
            
            return OptimizationResult(
                parameters=parameters.copy(),
                score=0.0,
                metrics={},
                evaluation_time=time.time() - start_time,
                iteration=len(self.optimization_history) + 1
            )
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """محاسبه امتیاز ترکیبی از معیارهای مختلف"""
        
        # وزن‌های مختلف برای معیارهای مختلف
        weights = {
            'retrieval_precision': 0.20,
            'retrieval_recall': 0.15,
            'generation_rouge': 0.20,
            'legal_accuracy': 0.25,
            'citation_quality': 0.15,
            'response_time': 0.05  # کمترین وزن برای سرعت
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                # نرمال‌سازی معیارهای زمانی (کمتر بهتر)
                if 'time' in metric_name:
                    # تبدیل زمان به امتیاز (حداکثر 10 ثانیه)
                    metric_score = max(0, 1 - (metrics[metric_name] / 10))
                else:
                    metric_score = metrics[metric_name]
                
                score += weight * metric_score
                total_weight += weight
        
        # نرمال‌سازی نهایی
        return score / total_weight if total_weight > 0 else 0.0
    
    def grid_search(
        self,
        config: OptimizationConfig = None,
        param_subset: List[str] = None
    ) -> OptimizationResult:
        """جستجوی شبکه‌ای (Grid Search)"""
        
        if config is None:
            config = OptimizationConfig()
        
        logger.info("شروع Grid Search")
        
        # انتخاب subset از parameters
        if param_subset:
            params_to_optimize = {
                k: v for k, v in self.hyperparameters.items() 
                if k in param_subset
            }
        else:
            # انتخاب parameters مهم‌تر برای جلوگیری از انفجار ترکیبی
            params_to_optimize = {
                k: v for k, v in self.hyperparameters.items()
                if k in ['chunk_size', 'top_k_retrieval', 'temperature', 'context_window_size']
            }
        
        # تولید تمام ترکیبات ممکن
        param_names = list(params_to_optimize.keys())
        param_values = [params_to_optimize[name].possible_values for name in param_names]
        
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        logger.info(f"تعداد ترکیبات: {total_combinations}")
        
        if total_combinations > config.max_iterations:
            logger.warning(f"تعداد ترکیبات ({total_combinations}) بیش از حد مجاز ({config.max_iterations})")
            # نمونه‌برداری تصادفی
            all_combinations = list(itertools.product(*param_values))
            import random
            random.shuffle(all_combinations)
            selected_combinations = all_combinations[:config.max_iterations]
        else:
            selected_combinations = list(itertools.product(*param_values))
        
        best_result = None
        results = []
        
        for i, combination in enumerate(selected_combinations):
            # ساخت dictionary parameters
            parameters = {
                param_names[j]: combination[j] 
                for j in range(len(param_names))
            }
            
            # اضافه کردن سایر parameters با مقادیر default
            for name, hp in self.hyperparameters.items():
                if name not in parameters:
                    parameters[name] = hp.current_value
            
            # ارزیابی
            result = self._evaluate_parameters(parameters)
            results.append(result)
            self.optimization_history.append(result)
            
            # بروزرسانی بهترین نتیجه
            if best_result is None or result.score > best_result.score:
                best_result = result
                self.best_result = result
                logger.info(f"🎯 بهترین نتیجه جدید: {result.score:.4f}")
            
            # Early stopping
            if self._should_stop_early(results, config):
                logger.info(f"Early stopping در iteration {i+1}")
                break
            
            # ذخیره نتایج میانی
            if config.save_intermediate_results and (i + 1) % 10 == 0:
                self._save_intermediate_results(results)
        
        logger.info(f"Grid Search تکمیل شد. بهترین امتیاز: {best_result.score:.4f}")
        
        return best_result
    
    def random_search(
        self,
        config: OptimizationConfig = None,
        param_subset: List[str] = None
    ) -> OptimizationResult:
        """جستجوی تصادفی (Random Search)"""
        
        if config is None:
            config = OptimizationConfig()
        
        logger.info("شروع Random Search")
        
        import random
        
        best_result = None
        results = []
        no_improvement_count = 0
        
        for iteration in range(config.max_iterations):
            # انتخاب تصادفی parameters
            parameters = {}
            
            for name, hp in self.hyperparameters.items():
                if param_subset is None or name in param_subset:
                    if hp.possible_values:
                        parameters[name] = random.choice(hp.possible_values)
                    else:
                        # برای پارامترهای continuous
                        if hp.min_value is not None and hp.max_value is not None:
                            parameters[name] = random.uniform(hp.min_value, hp.max_value)
                        else:
                            parameters[name] = hp.current_value
                else:
                    parameters[name] = hp.current_value
            
            # ارزیابی
            result = self._evaluate_parameters(parameters)
            results.append(result)
            self.optimization_history.append(result)
            
            # بروزرسانی بهترین نتیجه
            if best_result is None or result.score > best_result.score:
                best_result = result
                self.best_result = result
                no_improvement_count = 0
                logger.info(f"🎯 بهترین نتیجه جدید: {result.score:.4f}")
            else:
                no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count >= config.early_stopping_patience:
                logger.info(f"Early stopping: {no_improvement_count} iteration بدون بهبود")
                break
            
            # ذخیره نتایج میانی
            if config.save_intermediate_results and (iteration + 1) % 10 == 0:
                self._save_intermediate_results(results)
        
        logger.info(f"Random Search تکمیل شد. بهترین امتیاز: {best_result.score:.4f}")
        
        return best_result
    
    def bayesian_optimization(
        self,
        config: OptimizationConfig = None,
        param_subset: List[str] = None
    ) -> OptimizationResult:
        """بهینه‌سازی بیزی (Bayesian Optimization) - ساده‌شده"""
        
        if config is None:
            config = OptimizationConfig()
        
        logger.info("شروع Bayesian Optimization (Simplified)")
        
        # شروع با چند نقطه تصادفی
        results = []
        
        # مرحله اول: Random exploration
        for i in range(config.n_random_starts):
            parameters = self._sample_random_parameters(param_subset)
            result = self._evaluate_parameters(parameters)
            results.append(result)
            self.optimization_history.append(result)
        
        best_result = max(results, key=lambda r: r.score)
        self.best_result = best_result
        
        # مرحله دوم: Exploitation around best points
        for iteration in range(config.n_random_starts, config.max_iterations):
            # انتخاب parameters بر اساس بهترین نتایج قبلی
            if len(results) >= 3:
                # انتخاب 3 بهترین نتیجه
                top_results = sorted(results, key=lambda r: r.score, reverse=True)[:3]
                
                # تولید parameters جدید نزدیک به بهترین‌ها
                parameters = self._sample_around_best(top_results, param_subset)
            else:
                parameters = self._sample_random_parameters(param_subset)
            
            result = self._evaluate_parameters(parameters)
            results.append(result)
            self.optimization_history.append(result)
            
            # بروزرسانی بهترین نتیجه
            if result.score > best_result.score:
                best_result = result
                self.best_result = result
                logger.info(f"🎯 بهترین نتیجه جدید: {result.score:.4f}")
            
            # Early stopping
            if self._should_stop_early(results, config):
                logger.info(f"Early stopping در iteration {iteration+1}")
                break
        
        logger.info(f"Bayesian Optimization تکمیل شد. بهترین امتیاز: {best_result.score:.4f}")
        
        return best_result
    
    def _sample_random_parameters(self, param_subset: List[str] = None) -> Dict[str, Any]:
        """نمونه‌برداری تصادفی از parameters"""
        
        import random
        
        parameters = {}
        
        for name, hp in self.hyperparameters.items():
            if param_subset is None or name in param_subset:
                if hp.possible_values:
                    parameters[name] = random.choice(hp.possible_values)
                else:
                    parameters[name] = hp.current_value
            else:
                parameters[name] = hp.current_value
        
        return parameters
    
    def _sample_around_best(
        self,
        top_results: List[OptimizationResult],
        param_subset: List[str] = None
    ) -> Dict[str, Any]:
        """نمونه‌برداری در اطراف بهترین نتایج"""
        
        import random
        
        # انتخاب یک نتیجه تصادفی از بهترین‌ها
        base_result = random.choice(top_results)
        base_params = base_result.parameters
        
        new_params = {}
        
        for name, hp in self.hyperparameters.items():
            if param_subset is None or name in param_subset:
                if hp.possible_values:
                    current_value = base_params.get(name, hp.current_value)
                    
                    # یافتن index مقدار فعلی
                    try:
                        current_idx = hp.possible_values.index(current_value)
                        
                        # انتخاب مقدار نزدیک (±1 یا ±2 موقعیت)
                        max_shift = min(2, len(hp.possible_values) // 4)
                        shift = random.randint(-max_shift, max_shift)
                        new_idx = max(0, min(len(hp.possible_values) - 1, current_idx + shift))
                        
                        new_params[name] = hp.possible_values[new_idx]
                    except ValueError:
                        # اگر مقدار فعلی در لیست نیست
                        new_params[name] = random.choice(hp.possible_values)
                else:
                    new_params[name] = base_params.get(name, hp.current_value)
            else:
                new_params[name] = hp.current_value
        
        return new_params
    
    def _should_stop_early(
        self,
        results: List[OptimizationResult],
        config: OptimizationConfig
    ) -> bool:
        """بررسی شرط early stopping"""
        
        if len(results) < config.early_stopping_patience:
            return False
        
        # بررسی بهبود در iterations اخیر
        recent_scores = [r.score for r in results[-config.early_stopping_patience:]]
        max_recent = max(recent_scores)
        
        # اگر بهترین امتیاز در iterations اخیر کم بهبود داشته
        if len(results) > config.early_stopping_patience:
            best_before = max(r.score for r in results[:-config.early_stopping_patience])
            improvement = max_recent - best_before
            
            if improvement < config.early_stopping_threshold:
                return True
        
        return False
    
    def _save_intermediate_results(self, results: List[OptimizationResult]) -> None:
        """ذخیره نتایج میانی"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"optimization_intermediate_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "total_results": len(results),
            "best_score": max(r.score for r in results),
            "results": [asdict(r) for r in results[-10:]]  # فقط 10 نتیجه آخر
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def save_optimization_results(self, filename: str = None) -> str:
        """ذخیره تمام نتایج بهینه‌سازی"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        data = {
            "optimization_summary": {
                "total_evaluations": len(self.optimization_history),
                "best_score": self.best_result.score if self.best_result else 0.0,
                "best_parameters": self.best_result.parameters if self.best_result else {},
                "start_time": datetime.now().isoformat()
            },
            "hyperparameters_definition": {
                name: {
                    "current_value": hp.current_value,
                    "possible_values": hp.possible_values,
                    "description": hp.description
                }
                for name, hp in self.hyperparameters.items()
            },
            "optimization_history": [asdict(r) for r in self.optimization_history],
            "best_result": asdict(self.best_result) if self.best_result else None
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"نتایج بهینه‌سازی ذخیره شد: {filepath}")
        
        return str(filepath)
    
    def load_optimization_results(self, filepath: str) -> None:
        """بارگذاری نتایج بهینه‌سازی قبلی"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # بازسازی history
        self.optimization_history = []
        for result_data in data.get('optimization_history', []):
            result = OptimizationResult(**result_data)
            self.optimization_history.append(result)
        
        # بازسازی بهترین نتیجه
        if data.get('best_result'):
            self.best_result = OptimizationResult(**data['best_result'])
        
        logger.info(f"نتایج بهینه‌سازی بارگذاری شد: {len(self.optimization_history)} ارزیابی")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """خلاصه نتایج بهینه‌سازی"""
        
        if not self.optimization_history:
            return {"message": "هیچ بهینه‌سازی انجام نشده"}
        
        scores = [r.score for r in self.optimization_history]
        times = [r.evaluation_time for r in self.optimization_history]
        
        return {
            "total_evaluations": len(self.optimization_history),
            "best_score": max(scores),
            "worst_score": min(scores),
            "mean_score": statistics.mean(scores),
            "score_std": statistics.stdev(scores) if len(scores) > 1 else 0,
            "total_time": sum(times),
            "average_evaluation_time": statistics.mean(times),
            "best_parameters": self.best_result.parameters if self.best_result else {},
            "improvement": max(scores) - scores[0] if len(scores) > 1 else 0
        }
    
    def recommend_parameters(self) -> Dict[str, Any]:
        """توصیه بهترین parameters"""
        
        if not self.best_result:
            return {name: hp.current_value for name, hp in self.hyperparameters.items()}
        
        recommended = self.best_result.parameters.copy()
        
        # اضافه کردن پارامترهای نبود در بهترین نتیجه
        for name, hp in self.hyperparameters.items():
            if name not in recommended:
                recommended[name] = hp.current_value
        
        return recommended

# تست
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("⚙️  تست Optimization Manager")
    print("=" * 50)
    
    # تابع ارزیابی Mock
    def mock_evaluation_function(parameters: Dict[str, Any]) -> Dict[str, float]:
        """تابع ارزیابی Mock برای تست"""
        import random
        import time
        
        # شبیه‌سازی زمان ارزیابی
        time.sleep(0.1)
        
        # محاسبه امتیاز بر اساس parameters (mock)
        score_base = 0.7
        
        # chunk_size: بهتر در اندازه متوسط
        if parameters.get('chunk_size', 500) in [400, 500, 600]:
            score_base += 0.1
        
        # temperature: بهتر در مقادیر پایین
        if parameters.get('temperature', 0.1) <= 0.2:
            score_base += 0.1
        
        # نویز تصادفی
        noise = random.uniform(-0.1, 0.1)
        
        return {
            'retrieval_precision': min(1.0, score_base + noise),
            'retrieval_recall': min(1.0, score_base + noise * 0.8),
            'generation_rouge': min(1.0, score_base + noise * 1.2),
            'legal_accuracy': min(1.0, score_base + noise * 0.9),
            'citation_quality': min(1.0, score_base + noise * 1.1),
            'response_time': random.uniform(2, 8)
        }
    
    # ایجاد optimizer
    optimizer = LegalSystemOptimizer(mock_evaluation_function)
    
    # تست Random Search با تعداد کم iteration
    config = OptimizationConfig(max_iterations=10, early_stopping_patience=5)
    
    print("🔍 شروع Random Search...")
    best_result = optimizer.random_search(
        config=config,
        param_subset=['chunk_size', 'temperature', 'top_k_retrieval']
    )
    
    print(f"\n🎯 بهترین نتیجه:")
    print(f"   • امتیاز: {best_result.score:.4f}")
    print(f"   • Parameters:")
    for key, value in best_result.parameters.items():
        print(f"     - {key}: {value}")
    
    # خلاصه بهینه‌سازی
    summary = optimizer.get_optimization_summary()
    print(f"\n📊 خلاصه بهینه‌سازی:")
    print(f"   • تعداد ارزیابی‌ها: {summary['total_evaluations']}")
    print(f"   • بهترین امتیاز: {summary['best_score']:.4f}")
    print(f"   • میانگین امتیاز: {summary['mean_score']:.4f}")
    print(f"   • بهبود کلی: {summary['improvement']:.4f}")
    
    # ذخیره نتایج
    saved_path = optimizer.save_optimization_results()
    print(f"\n💾 نتایج ذخیره شد: {saved_path}")
    
    print("\n✅ تست کامل شد")