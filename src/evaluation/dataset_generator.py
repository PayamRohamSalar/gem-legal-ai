"""
src/evaluation/dataset_generator.py - تولید Dataset تست برای فاز 4

این فایل مسئول ایجاد dataset جامع برای تست و ارزیابی سیستم دستیار حقوقی است.
"""

import json
import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class QuestionDifficulty(Enum):
    """سطح دشواری سوالات"""
    BASIC = "پایه"           # سوالات ساده و مستقیم
    INTERMEDIATE = "متوسط"   # سوالات نیازمند تحلیل
    ADVANCED = "پیشرفته"     # سوالات پیچیده و چندمرحله‌ای

class QuestionCategory(Enum):
    """دسته‌بندی موضوعی سوالات"""
    FACULTY_DUTIES = "وظایف_هیئت_علمی"
    KNOWLEDGE_BASED_COMPANIES = "شرکت_دانش_بنیان"
    TECHNOLOGY_TRANSFER = "انتقال_فناوری"
    RESEARCH_CONTRACTS = "قراردادهای_پژوهشی"
    INTELLECTUAL_PROPERTY = "مالکیت_فکری"
    UNIVERSITY_INDUSTRY = "ارتباط_صنعت_دانشگاه"
    RESEARCH_EVALUATION = "ارزیابی_پژوهش"

@dataclass
class TestQuestion:
    """ساختار سوال تست"""
    id: str
    question: str
    category: QuestionCategory
    difficulty: QuestionDifficulty
    expected_answer: str
    relevant_articles: List[str]
    keywords: List[str]
    context_needed: List[str]
    evaluation_criteria: List[str]
    created_date: str
    
class LegalDatasetGenerator:
    """تولیدکننده dataset حقوقی"""
    
    def __init__(self, output_dir: str = "data/test_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # بانک سوالات پایه
        self.question_templates = self._load_question_templates()
        self.legal_contexts = self._load_legal_contexts()
        
        logger.info(f"Dataset Generator راه‌اندازی شد - مسیر: {self.output_dir}")
    
    def _load_question_templates(self) -> Dict[QuestionCategory, Dict[QuestionDifficulty, List[Dict]]]:
        """بارگذاری template های سوالات"""
        
        templates = {
            QuestionCategory.FACULTY_DUTIES: {
                QuestionDifficulty.BASIC: [
                    {
                        "template": "وظایف اعضای هیئت علمی در {subject} چیست؟",
                        "subjects": ["پژوهش", "آموزش", "خدمات", "تحقیق"],
                        "expected_pattern": "بر اساس ماده {article} {law_name}",
                        "articles": ["3", "4", "5"],
                        "evaluation_criteria": ["صحت ارجاع", "کامل بودن پاسخ", "وضوح بیان"]
                    },
                    {
                        "template": "معیارهای ارتقای اعضای هیئت علمی در رشته {subject} کدامند؟",
                        "subjects": ["مهندسی", "علوم پایه", "علوم انسانی", "پزشکی"],
                        "expected_pattern": "طبق آیین‌نامه ارتقای اعضای هیئت علمی",
                        "evaluation_criteria": ["دقت معیارها", "ذکر درصد امتیازات"]
                    }
                ],
                QuestionDifficulty.INTERMEDIATE: [
                    {
                        "template": "تفاوت وظایف استاد تمام با {subject} در حوزه پژوهش چیست؟",
                        "subjects": ["استاد دانشیار", "استادیار", "مربی"],
                        "expected_pattern": "مطابق جدول رتبه‌بندی",
                        "evaluation_criteria": ["مقایسه دقیق", "ذکر تفاوت‌ها"]
                    }
                ],
                QuestionDifficulty.ADVANCED: [
                    {
                        "template": "در صورت عدم انجام تعهدات پژوهشی توسط عضو هیئت علمی، چه اقدامات انتظامی قابل اتخاذ است؟",
                        "subjects": [""],
                        "expected_pattern": "بر اساس مواد انتظامی قانون",
                        "evaluation_criteria": ["شناسایی مراحل", "ذکر مجازات‌ها", "روند قانونی"]
                    }
                ]
            },
            
            QuestionCategory.KNOWLEDGE_BASED_COMPANIES: {
                QuestionDifficulty.BASIC: [
                    {
                        "template": "تعریف شرکت دانش‌بنیان بر اساس قانون چیست؟",
                        "subjects": [""],
                        "expected_pattern": "ماده 1 قانون حمایت از شرکت‌های دانش‌بنیان",
                        "evaluation_criteria": ["دقت تعریف", "ذکر منبع قانونی"]
                    },
                    {
                        "template": "مزایای {subject} شرکت‌های دانش‌بنیان کدامند؟",
                        "subjects": ["مالیاتی", "اعتباری", "صادراتی", "گمرکی"],
                        "expected_pattern": "مواد 5-8 قانون حمایت",
                        "evaluation_criteria": ["کامل بودن لیست", "ذکر درصد معافیت‌ها"]
                    }
                ],
                QuestionDifficulty.INTERMEDIATE: [
                    {
                        "template": "شرایط احراز و نگهداری عنوان دانش‌بنیان برای شرکت {subject} چیست؟",
                        "subjects": ["نرم‌افزاری", "بیوتکنولوژی", "نانو", "انرژی"],
                        "expected_pattern": "آیین‌نامه احراز و ارزیابی",
                        "evaluation_criteria": ["ذکر معیارهای کمی", "شرایط نگهداری"]
                    }
                ],
                QuestionDifficulty.ADVANCED: [
                    {
                        "template": "فرآیند تبدیل یافته‌های پژوهشی دانشگاه به شرکت دانش‌بنیان چگونه است؟",
                        "subjects": [""],
                        "expected_pattern": "چندین قانون و آیین‌نامه",
                        "evaluation_criteria": ["ذکر مراحل", "قوانین مرتبط", "نهادهای درگیر"]
                    }
                ]
            },
            
            QuestionCategory.TECHNOLOGY_TRANSFER: {
                QuestionDifficulty.BASIC: [
                    {
                        "template": "تعریف انتقال فناوری در آیین‌نامه مربوطه چیست؟",
                        "subjects": [""],
                        "expected_pattern": "ماده 2 آیین‌نامه انتقال فناوری",
                        "evaluation_criteria": ["دقت تعریف", "ذکر اجزای تعریف"]
                    }
                ],
                QuestionDifficulty.INTERMEDIATE: [
                    {
                        "template": "انواع مختلف قراردادهای انتقال فناوری و ویژگی‌های هر یک چیست؟",
                        "subjects": [""],
                        "expected_pattern": "مواد 8-12 آیین‌نامه",
                        "evaluation_criteria": ["شناسایی انواع", "ویژگی‌های هر نوع"]
                    }
                ],
                QuestionDifficulty.ADVANCED: [
                    {
                        "template": "مراحل قانونی انتقال فناوری از دانشگاه به صنعت و نقش نهادهای مختلف چیست؟",
                        "subjects": [""],
                        "expected_pattern": "آیین‌نامه انتقال فناوری + قوانین مرتبط",
                        "evaluation_criteria": ["ذکر مراحل", "نقش نهادها", "الزامات قانونی"]
                    }
                ]
            }
        }
        
        return templates
    
    def _load_legal_contexts(self) -> Dict[str, List[Dict]]:
        """بارگذاری context های حقوقی مرتبط"""
        
        return {
            "faculty_duties": [
                {
                    "content": "اعضای هیئت علمی موظف به انجام پژوهش و تحقیق در زمینه تخصصی خود هستند",
                    "source": "قانون مقررات انتظامی هیئت علمی - ماده 3",
                    "article": "3",
                    "relevance_score": 0.95
                },
                {
                    "content": "نیروی آموزشی و تحقیقاتی دانشگاه‌ها متشکل از اعضای هیئت علمی است",
                    "source": "قانون تشکیلات وزارت علوم - ماده 15",
                    "article": "15",
                    "relevance_score": 0.85
                }
            ],
            
            "knowledge_based_companies": [
                {
                    "content": "شرکت‌های دانش‌بنیان شرکت‌هایی هستند که بر پایه دانش فنی بروز و با بهره‌گیری از یافته‌های نوین علمی و تحقیقاتی، اقدام به تولید محصولات، فرآیندها و ارائه خدمات فناورانه می‌کنند",
                    "source": "قانون حمایت از شرکت‌های دانش‌بنیان - ماده 1",
                    "article": "1",
                    "relevance_score": 1.0
                },
                {
                    "content": "شرکت‌های دانش‌بنیان از معافیت مالیاتی تا پنج سال برخوردار می‌شوند",
                    "source": "قانون حمایت از شرکت‌های دانش‌بنیان - ماده 5",
                    "article": "5",
                    "relevance_score": 0.9
                }
            ]
        }
    
    def _generate_single_question(
        self, 
        category: QuestionCategory, 
        difficulty: QuestionDifficulty, 
        template: Dict,
        question_id: str
    ) -> TestQuestion:
        """تولید یک سوال تست"""
        
        # انتخاب subject تصادفی
        if template["subjects"] and template["subjects"][0]:
            subject = random.choice(template["subjects"])
            question_text = template["template"].format(subject=subject)
        else:
            question_text = template["template"]
        
        # تولید پاسخ مورد انتظار
        expected_answer = self._generate_expected_answer(category, difficulty, template, question_text)
        
        # استخراج مواد مرتبط
        relevant_articles = template.get("articles", ["نامشخص"])
        
        # تولید کلمات کلیدی
        keywords = self._extract_keywords(question_text, category)
        
        # تعیین context مورد نیاز
        context_needed = self._determine_context_needed(category)
        
        return TestQuestion(
            id=question_id,
            question=question_text,
            category=category,
            difficulty=difficulty,
            expected_answer=expected_answer,
            relevant_articles=relevant_articles,
            keywords=keywords,
            context_needed=context_needed,
            evaluation_criteria=template["evaluation_criteria"],
            created_date=datetime.now().isoformat()
        )
    
    def _generate_expected_answer(
        self, 
        category: QuestionCategory, 
        difficulty: QuestionDifficulty, 
        template: Dict,
        question: str
    ) -> str:
        """تولید پاسخ مورد انتظار"""
        
        base_patterns = {
            QuestionCategory.FACULTY_DUTIES: {
                QuestionDifficulty.BASIC: """بر اساس ماده 3 قانون مقررات انتظامی هیئت علمی مصوب 1364:

🔹 وظایف اصلی اعضای هیئت علمی:
1. انجام پژوهش و تحقیق در زمینه تخصصی
2. تدریس و آموزش دانشجویان  
3. انتشار نتایج تحقیقات در مجلات معتبر
4. راهنمایی پایان‌نامه‌ها و رساله‌ها
5. مشارکت در فعالیت‌های علمی دانشگاه

📋 مرجع: ماده 3 قانون مقررات انتظامی هیئت علمی""",

                QuestionDifficulty.INTERMEDIATE: """مقایسه وظایف پژوهشی بر اساس رتبه علمی:

🎓 استاد تمام: رهبری پروژه‌های تحقیقاتی ملی، راهنمایی رساله دکتری
🎓 دانشیار: انجام پژوهش‌های مستقل، راهنمایی پایان‌نامه کارشناسی ارشد  
🎓 استادیار: مشارکت در پروژه‌های گروهی، راهنمایی پایان‌نامه کارشناسی

📋 مرجع: آیین‌نامه ارتقای اعضای هیئت علمی""",

                QuestionDifficulty.ADVANCED: """فرآیند انتظامی عدم انجام تعهدات پژوهشی:

⚖️ مراحل قانونی:
1. اخطار کتبی (ماده 18)
2. کسر از حقوق (ماده 19) 
3. تعلیق موقت (ماده 20)
4. انفصال از خدمت (ماده 21)

🏛️ مراجع رسیدگی:
- کمیسیون انتظامی دانشگاه
- هیئت عالی انتظامی

📋 مرجع: مواد 18-21 قانون مقررات انتظامی"""
            },
            
            QuestionCategory.KNOWLEDGE_BASED_COMPANIES: {
                QuestionDifficulty.BASIC: """بر اساس ماده 1 قانون حمایت از شرکت‌های دانش‌بنیان:

🏢 تعریف: شرکت‌هایی که بر پایه دانش فنی بروز و با بهره‌گیری از یافته‌های نوین علمی و تحقیقاتی، اقدام به تولید محصولات، فرآیندها و ارائه خدمات فناورانه می‌کنند.

🎯 ویژگی‌ها:
✅ استفاده از فناوری پیشرفته
✅ نیروی کار متخصص (حداقل 30%)
✅ سرمایه‌گذاری در تحقیق و توسعه

📋 مرجع: ماده 1 قانون حمایت از شرکت‌های دانش‌بنیان"""
            }
        }
        
        # انتخاب pattern مناسب
        if category in base_patterns and difficulty in base_patterns[category]:
            return base_patterns[category][difficulty]
        
        return f"پاسخ نمونه برای سوال از دسته {category.value} در سطح {difficulty.value}"
    
    def _extract_keywords(self, question: str, category: QuestionCategory) -> List[str]:
        """استخراج کلمات کلیدی از سوال"""
        
        category_keywords = {
            QuestionCategory.FACULTY_DUTIES: ["هیئت علمی", "وظایف", "پژوهش", "تدریس", "ارتقا"],
            QuestionCategory.KNOWLEDGE_BASED_COMPANIES: ["دانش‌بنیان", "شرکت", "فناوری", "مزایا"],
            QuestionCategory.TECHNOLOGY_TRANSFER: ["انتقال فناوری", "قرارداد", "تجاری‌سازی"],
        }
        
        base_keywords = category_keywords.get(category, [])
        
        # استخراج کلمات از متن سوال
        question_words = question.lower().split()
        important_words = [word for word in question_words if len(word) > 3]
        
        return base_keywords + important_words[:3]
    
    def _determine_context_needed(self, category: QuestionCategory) -> List[str]:
        """تعیین context های مورد نیاز"""
        
        context_map = {
            QuestionCategory.FACULTY_DUTIES: ["faculty_duties", "promotion_criteria"],
            QuestionCategory.KNOWLEDGE_BASED_COMPANIES: ["knowledge_based_companies", "tax_benefits"],
            QuestionCategory.TECHNOLOGY_TRANSFER: ["technology_transfer", "contracts"],
        }
        
        return context_map.get(category, ["general"])
    
    def generate_dataset(
        self, 
        total_questions: int = 150,
        difficulty_distribution: Dict[QuestionDifficulty, float] = None,
        category_distribution: Dict[QuestionCategory, float] = None
    ) -> List[TestQuestion]:
        """تولید dataset کامل"""
        
        # توزیع پیش‌فرض دشواری
        if difficulty_distribution is None:
            difficulty_distribution = {
                QuestionDifficulty.BASIC: 0.4,        # 40% ساده
                QuestionDifficulty.INTERMEDIATE: 0.4,  # 40% متوسط  
                QuestionDifficulty.ADVANCED: 0.2       # 20% پیشرفته
            }
        
        # توزیع پیش‌فرض دسته‌بندی
        if category_distribution is None:
            category_distribution = {
                QuestionCategory.FACULTY_DUTIES: 0.3,
                QuestionCategory.KNOWLEDGE_BASED_COMPANIES: 0.25,
                QuestionCategory.TECHNOLOGY_TRANSFER: 0.2,
                QuestionCategory.RESEARCH_CONTRACTS: 0.15,
                QuestionCategory.INTELLECTUAL_PROPERTY: 0.1
            }
        
        questions = []
        
        logger.info(f"شروع تولید {total_questions} سوال...")
        
        # محاسبه تعداد سوالات برای هر دسته و سطح
        for category in QuestionCategory:
            if category not in category_distribution:
                continue
                
            category_count = int(total_questions * category_distribution[category])
            
            for difficulty in QuestionDifficulty:
                difficulty_count = int(category_count * difficulty_distribution[difficulty])
                
                # تولید سوالات برای این دسته و سطح
                for i in range(difficulty_count):
                    if category in self.question_templates and difficulty in self.question_templates[category]:
                        templates = self.question_templates[category][difficulty]
                        if templates:
                            template = random.choice(templates)
                            
                            question_id = f"{category.value}_{difficulty.value}_{len(questions)+1:03d}"
                            
                            question = self._generate_single_question(
                                category, difficulty, template, question_id
                            )
                            questions.append(question)
        
        # تکمیل تعداد کل در صورت نیاز
        while len(questions) < total_questions:
            # انتخاب تصادفی دسته و سطح
            category = random.choice(list(QuestionCategory))
            difficulty = random.choice(list(QuestionDifficulty))
            
            if (category in self.question_templates and 
                difficulty in self.question_templates[category] and
                self.question_templates[category][difficulty]):
                
                template = random.choice(self.question_templates[category][difficulty])
                question_id = f"extra_{len(questions)+1:03d}"
                
                question = self._generate_single_question(
                    category, difficulty, template, question_id
                )
                questions.append(question)
        
        # ترتیب تصادفی
        random.shuffle(questions)
        
        logger.info(f"✅ تولید {len(questions)} سوال کامل شد")
        
        return questions
    
    def save_dataset(self, questions: List[TestQuestion], filename: str = None) -> str:
        """ذخیره dataset در فایل JSON (نسخه اصلاح شده)"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"legal_test_dataset_{timestamp}.json"

        filepath = self.output_dir / filename

        # تبدیل Enum ها به رشته قبل از ذخیره‌سازی
        questions_as_dicts = []
        for q in questions:
            q_dict = asdict(q)
            q_dict['category'] = q_dict['category'].value
            q_dict['difficulty'] = q_dict['difficulty'].value
            questions_as_dicts.append(q_dict)

        dataset_dict = {
            "metadata": {
                "total_questions": len(questions),
                "created_date": datetime.now().isoformat(),
                "difficulty_distribution": self._analyze_difficulty_distribution(questions),
                "category_distribution": self._analyze_category_distribution(questions)
            },
            "questions": questions_as_dicts
        }

        # ذخیره فایل
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"Dataset ذخیره شد: {filepath}")

        return str(filepath)
    
    def _analyze_difficulty_distribution(self, questions: List[TestQuestion]) -> Dict[str, int]:
        """تحلیل توزیع دشواری"""
        distribution = {}
        for difficulty in QuestionDifficulty:
            count = len([q for q in questions if q.difficulty == difficulty])
            distribution[difficulty.value] = count
        return distribution
    
    def _analyze_category_distribution(self, questions: List[TestQuestion]) -> Dict[str, int]:
        """تحلیل توزیع دسته‌بندی"""
        distribution = {}
        for category in QuestionCategory:
            count = len([q for q in questions if q.category == category])
            distribution[category.value] = count
        return distribution
    
    def load_dataset(self, filepath: str) -> List[TestQuestion]:
        """بارگذاری dataset از فایل"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        for q_data in data['questions']:
            # تبدیل enum ها
            q_data['category'] = QuestionCategory(q_data['category'])
            q_data['difficulty'] = QuestionDifficulty(q_data['difficulty'])
            
            questions.append(TestQuestion(**q_data))
        
        logger.info(f"Dataset بارگذاری شد: {len(questions)} سوال")
        
        return questions
    
    def get_dataset_stats(self, questions: List[TestQuestion]) -> Dict[str, Any]:
        """آمار dataset"""
        
        return {
            "total": len(questions),
            "by_difficulty": self._analyze_difficulty_distribution(questions),
            "by_category": self._analyze_category_distribution(questions),
            "avg_question_length": sum(len(q.question) for q in questions) / len(questions),
            "avg_answer_length": sum(len(q.expected_answer) for q in questions) / len(questions),
            "total_articles_referenced": len(set(
                article for q in questions for article in q.relevant_articles
            ))
        }

# تست و نمایش
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🏗️  تست Dataset Generator")
    print("=" * 50)
    
    # ایجاد generator
    generator = LegalDatasetGenerator()
    
    # تولید dataset نمونه
    questions = generator.generate_dataset(total_questions=50)
    
    print(f"✅ تولید {len(questions)} سوال")
    
    # نمایش نمونه سوالات
    print("\n📋 نمونه سوالات:")
    for i, q in enumerate(questions[:3], 1):
        print(f"\n{i}. [{q.category.value}] [{q.difficulty.value}]")
        print(f"   سوال: {q.question}")
        print(f"   پاسخ: {q.expected_answer[:100]}...")
        print(f"   مواد: {q.relevant_articles}")
    
    # ذخیره dataset
    filepath = generator.save_dataset(questions, "test_dataset_sample.json")
    print(f"\n💾 ذخیره شد: {filepath}")
    
    # آمار
    stats = generator.get_dataset_stats(questions)
    print(f"\n📊 آمار Dataset:")
    for key, value in stats.items():
        print(f"   • {key}: {value}")
    
    print("\n✅ تست کامل شد")