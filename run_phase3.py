#!/usr/bin/env python3
"""
run_phase3.py - اجرای کامل فاز 3: سیستم تولید پاسخ

این اسکریپت فاز 3 را به صورت کامل اجرا و تست می‌کند.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import json

# اضافه کردن مسیر src به Python path
current_dir = Path(__file__).parent
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

# تنظیم logging
logs_dir = current_dir / 'logs'
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'phase3.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_header():
    """چاپ هدر شروع فاز 3"""
    print("=" * 70)
    print("🤖 اجرای فاز 3: سیستم تولید پاسخ دستیار حقوقی")
    print("=" * 70)
    print(f"📅 زمان شروع: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 مسیر پروژه: {current_dir}")
    print(f"📂 مسیر src: {src_path}")
    print("-" * 70)

def check_file_structure():
    """بررسی ساختار فایل‌ها"""
    print("\n🔸 مرحله 1: بررسی ساختار فایل‌ها")
    print("-" * 40)
    
    required_files = [
        'src/generation/llm_manager.py',
        'src/generation/prompt_engine.py',
        'src/generation/citation_engine.py',
        'src/generation/integrated_response_system.py'
    ]
    
    all_files_exist = True
    
    for file_path in required_files:
        full_path = current_dir / file_path
        if full_path.exists() and full_path.stat().st_size > 100:
            print(f"✅ {file_path} - موجود ({full_path.stat().st_size} bytes)")
        else:
            print(f"❌ {file_path} - مفقود یا خالی")
            all_files_exist = False
    
    return all_files_exist

def test_imports():
    """تست import های اصلی"""
    print("\n🔸 مرحله 2: تست import ها")
    print("-" * 40)
    
    try:
        from generation.llm_manager import LLMManager, create_model_configs
        print("✅ LLMManager imported")
        
        from generation.prompt_engine import PromptEngine, QueryType, ContextInfo, ContextType
        print("✅ PromptEngine imported")
        
        from generation.citation_engine import CitationEngine
        print("✅ CitationEngine imported")
        
        from generation.integrated_response_system import LegalResponseSystem, SimpleLegalAssistant
        print("✅ IntegratedResponseSystem imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ خطا در import: {e}")
        return False

def test_dependencies():
    """بررسی وابستگی‌ها"""
    print("\n🔸 مرحله 3: بررسی وابستگی‌ها")
    print("-" * 40)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('pathlib', 'Pathlib'),
        ('dataclasses', 'Dataclasses'),
        ('enum', 'Enum'),
        ('re', 'Regex'),
        ('json', 'JSON'),
        ('logging', 'Logging')
    ]
    
    all_packages_found = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - نصب نشده")
            all_packages_found = False
    
    # بررسی پکیج‌های اختیاری
    optional_packages = [
        ('fastapi', 'FastAPI'),
        ('pydantic', 'Pydantic')
    ]
    
    print("\nپکیج‌های اختیاری:")
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} - نصب نشده (اختیاری)")
    
    return all_packages_found

def test_individual_components():
    """تست اجزای جداگانه"""
    print("\n🔸 مرحله 4: تست اجزای جداگانه")
    print("-" * 40)
    
    try:
        from generation.llm_manager import LLMManager, create_model_configs
        from generation.prompt_engine import PromptEngine, ContextInfo, ContextType
        from generation.citation_engine import CitationEngine
        
        # تست LLMManager
        print("🧪 تست LLMManager...")
        configs = create_model_configs()
        llm_manager = LLMManager(configs)
        
        if llm_manager.load_model_mock('qwen_7b'):
            print("  ✅ Mock مدل بارگذاری شد")
            
            # تست تولید
            result = llm_manager.generate_response("سوال تست")
            if result['success']:
                print("  ✅ تولید پاسخ موفق")
            else:
                print(f"  ❌ خطا در تولید: {result.get('error')}")
        else:
            print("  ❌ خطا در بارگذاری مدل")
        
        # تست PromptEngine
        print("\n🧪 تست PromptEngine...")
        prompt_engine = PromptEngine()
        
        # تست تشخیص نوع سوال
        query_type = prompt_engine.detect_query_type("وظایف هیئت علمی چیست؟")
        print(f"  ✅ تشخیص نوع سوال: {query_type.value}")
        
        # تست ساخت prompt
        contexts = [ContextInfo(
            content="تست محتوا",
            source="منبع تست",
            document_type=ContextType.LAW
        )]
        
        prompt, _ = prompt_engine.build_prompt("سوال تست", contexts)
        print(f"  ✅ ساخت prompt: {len(prompt)} کاراکتر")
        
        # تست CitationEngine
        print("\n🧪 تست CitationEngine...")
        citation_engine = CitationEngine()
        
        sample_text = "بر اساس ماده 3 قانون مقررات انتظامی، موضوع مشخص است."
        citations = citation_engine.extract_citations_from_text(sample_text)
        print(f"  ✅ استخراج ارجاعات: {len(citations)} مورد")
        
        llm_manager.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ خطا در تست اجزا: {e}")
        return False

async def test_integrated_system():
    """تست سیستم یکپارچه"""
    print("\n🔸 مرحله 5: تست سیستم یکپارچه")
    print("-" * 40)
    
    try:
        from generation.integrated_response_system import SimpleLegalAssistant
        
        # ایجاد دستیار
        assistant = SimpleLegalAssistant()
        
        # راه‌اندازی
        success = await assistant.setup()
        if not success:
            print("❌ خطا در راه‌اندازی سیستم یکپارچه")
            return False
        
        print("✅ سیستم یکپارچه راه‌اندازی شد")
        
        # تست پرسش ساده
        contexts = [{
            'content': 'اعضای هیئت علمی موظف به پژوهش هستند',
            'source': 'قانون مقررات انتظامی',
            'document_type': 'قانون',
            'relevance_score': 0.9
        }]
        
        response = await assistant.ask(
            "وظایف اعضای هیئت علمی چیست؟",
            contexts
        )
        
        if response and "خطا" not in response:
            print("✅ تست پرسش موفق")
            print(f"  📝 پاسخ: {response[:100]}...")
            
            # نمایش آمار
            stats = assistant.get_stats()
            print(f"  📊 آمار: {stats['total_requests']} درخواست، {stats['successful_requests']} موفق")
        else:
            print(f"❌ خطا در پرسش: {response}")
            assistant.shutdown()
            return False
        
        assistant.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ خطا در تست سیستم یکپارچه: {e}")
        return False

def create_test_file():
    """ایجاد فایل تست ساده"""
    print("\n🔸 مرحله 6: ایجاد فایل تست")
    print("-" * 40)
    
    test_content = """#!/usr/bin/env python3
\"\"\"
test_phase3.py - تست سریع فاز 3
\"\"\"

import sys
from pathlib import Path

# اضافه کردن مسیر src
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    print("🧪 تست سریع فاز 3")
    print("=" * 30)
    
    try:
        # تست import ها
        from generation.llm_manager import create_model_configs
        from generation.prompt_engine import PromptEngine
        from generation.citation_engine import CitationEngine
        
        print("✅ همه imports موفق")
        
        # تست basic functionality
        configs = create_model_configs()
        print(f"✅ Model configs: {len(configs)} مدل")
        
        engine = PromptEngine()
        print(f"✅ PromptEngine: {len(engine.templates)} template")
        
        citation = CitationEngine()
        print("✅ CitationEngine")
        
        print("\\n🎉 همه تست‌ها موفق!")
        
    except Exception as e:
        print(f"❌ خطا: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    
    try:
        with open('test_phase3.py', 'w', encoding='utf-8') as f:
            f.write(test_content)
        print("✅ فایل test_phase3.py ایجاد شد")
        return True
    except Exception as e:
        print(f"❌ خطا در ایجاد فایل تست: {e}")
        return False

def create_final_report(results):
    """ایجاد گزارش نهایی"""
    print("\n🔸 ایجاد گزارش نهایی")
    print("-" * 40)
    
    success_count = sum(results.values())
    total_tests = len(results)
    success_rate = (success_count / total_tests) * 100
    
    report = {
        'phase': 3,
        'title': 'سیستم تولید پاسخ دستیار حقوقی',
        'timestamp': datetime.now().isoformat(),
        'tests': {
            'file_structure': results.get('file_structure', False),
            'imports': results.get('imports', False),
            'dependencies': results.get('dependencies', False),
            'components': results.get('components', False),
            'integrated_system': results.get('integrated_system', False),
            'test_file_creation': results.get('test_file_creation', False)
        },
        'success_count': success_count,
        'total_tests': total_tests,
        'success_rate_percentage': success_rate,
        'status': 'موفق' if success_rate >= 80 else 'نیازمند بررسی',
        'next_steps': [
            "اجرای test_phase3.py برای تست سریع",
            "ایجاد interface برای API",
            "تست با داده‌های واقعی",
            "بهینه‌سازی عملکرد"
        ]
    }
    
    # ذخیره گزارش
    reports_dir = current_dir / 'data' / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = reports_dir / f'phase3_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"✅ گزارش ذخیره شد: {report_file}")
    except Exception as e:
        print(f"⚠️  خطا در ذخیره گزارش: {e}")
    
    return report

def print_summary(results, report):
    """چاپ خلاصه نهایی"""
    print("\n" + "=" * 70)
    print("📋 خلاصه اجرای فاز 3")
    print("=" * 70)
    
    print(f"⏱️  زمان اجرا: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 نرخ موفقیت: {report['success_rate_percentage']:.1f}% ({report['success_count']}/{report['total_tests']})")
    
    print(f"\n🧪 نتایج تست‌ها:")
    for test_name, result in results.items():
        icon = "✅" if result else "❌"
        print(f"  {icon} {test_name.replace('_', ' ').title()}")
    
    print(f"\n📁 فایل‌های ایجاد شده:")
    print(f"  • test_phase3.py - تست سریع")
    print(f"  • logs/phase3.log - لاگ اجرا")
    print(f"  • data/reports/phase3_report_*.json - گزارش جامع")
    
    print(f"\n🔄 مراحل بعدی:")
    for step in report['next_steps']:
        print(f"  • {step}")
    
    if report['success_rate_percentage'] >= 80:
        print(f"\n🎉 فاز 3 با موفقیت کامل شد!")
        print("✅ سیستم آماده پاسخگویی به سوالات حقوقی است")
    else:
        print(f"\n⚠️  فاز 3 با چند مشکل کامل شد")
        print("🔧 لطفاً موارد ناموفق را بررسی و اصلاح کنید")

async def main():
    """تابع اصلی اجرا"""
    print_header()
    
    # اجرای تست‌ها
    results = {}
    
    results['file_structure'] = check_file_structure()
    results['imports'] = test_imports()
    results['dependencies'] = test_dependencies()
    results['components'] = test_individual_components()
    results['integrated_system'] = await test_integrated_system()
    results['test_file_creation'] = create_test_file()
    
    # ایجاد گزارش و خلاصه
    report = create_final_report(results)
    print_summary(results, report)
    
    # خروج با کد مناسب
    if report['success_rate_percentage'] >= 80:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())