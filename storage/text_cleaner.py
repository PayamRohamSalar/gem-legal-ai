# src/data_processing/text_cleaner.py
import re

def normalize_persian_text(text: str) -> str:
    """
    کاراکترهای عربی و فارسی را یکسان‌سازی کرده و به فرم استاندارد فارسی تبدیل می‌کند.
    """
    # تبدیل 'ي' و 'ك' عربی به 'ی' و 'ک' فارسی
    text = text.replace('ي', 'ی').replace('ك', 'ک')
    # می‌توانید موارد بیشتری را در صورت نیاز اضافه کنید
    return text

def clean_text(text: str) -> str:
    """
    عملیات پاکسازی اولیه روی متن را انجام می‌دهد.

    - یکسان‌سازی کاراکترهای فارسی
    - حذف فاصله‌ها، تب‌ها و خطوط جدید اضافه
    """
    # 1. نرمال‌سازی کاراکترها
    text = normalize_persian_text(text)
    
    # 2. حذف خطوط خالی متعدد و جایگزینی با یک خط جدید
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # 3. حذف فاصله‌ها و تب‌های متعدد و جایگزینی با یک فاصله
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()