# 00_create_project_structure.py
import os

# لیست پوشه‌های اصلی و زیرپوشه‌ها
project_structure = {
    "app": ["backend", "frontend"],
    "config": [],
    "data": ["raw_documents", "vector_store"],
    "evaluation": ["reports"],
    "models": [],
    "scripts": [],
    "src": ["data_processing", "retrieval", "generation"]
}

# لیست فایل‌های placeholder برای توضیحات
readme_files = {
    "app/README.md": "This folder contains the application code (backend and frontend).",
    "config/README.md": "Configuration files (e.g., API keys, settings).",
    "data/README.md": "All project data, from raw to processed.",
    "data/raw_documents/README.md": "Place initial raw documents (e.g., .docx, .pdf) here.",
    "evaluation/README.md": "Evaluation scripts, datasets, and reports.",
    "models/README.md": "Locally stored AI models (e.g., embedding models).",
    "scripts/README.md": "Helper and one-off scripts (e.g., data import, testing).",
    "src/README.md": "Core source code for the AI pipeline.",
    "src/__init__.py": "",
    "src/data_processing/__init__.py": "",
    "src/retrieval/__init__.py": "",
    "src/generation/__init__.py": ""
}

def create_project():
    """ایجاد ساختار پوشه و فایل‌های پروژه"""
    print("Creating project structure...")
    
    # ایجاد پوشه‌ها
    for main_dir, sub_dirs in project_structure.items():
        os.makedirs(main_dir, exist_ok=True)
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(main_dir, sub_dir), exist_ok=True)
            
    # ایجاد فایل‌های README.md و __init__.py
    for file_path, content in readme_files.items():
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
    print("Project structure created successfully! ✅")

if __name__ == "__main__":
    create_project()