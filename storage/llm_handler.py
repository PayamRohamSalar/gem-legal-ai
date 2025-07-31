# src/generation/llm_handler.py
import ollama
from typing import Dict, Any

def generate_response(model_name: str, prompt: str) -> Dict[str, Any]:
    """
    پرامپت را به مدل مشخص شده در Ollama ارسال کرده و پاسخ را دریافت می‌کند.
    """
    print(f"\nSending prompt to LLM ('{model_name}')...")
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        # در صورت بروز خطا، یک پاسخ استاندارد برگردان
        return {
            'message': {
                'content': 'خطا در برقراری ارتباط با سرور Ollama. لطفاً از روشن بودن و در دسترس بودن آن اطمینان حاصل کنید.'
            }
        }