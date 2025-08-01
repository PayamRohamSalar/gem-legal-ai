# app.py - Streamlit Frontend for Legal AI Assistant
# این کد نسخه اصلاح شده و هماهنگ شده کد ارسالی شماست.

import streamlit as st
import requests
import json
import time
import pandas as pd
from datetime import datetime

# (تمام کد CSS و کلاس APIClient که فرستادی عالی است و بدون تغییر باقی می‌ماند)
# ...

# ===== UI اصلی =====

# (کد Header و Sidebar شما عالی است و بدون تغییر باقی می‌ماند)
# ...

# ===== صفحه اصلی - پرسش و پاسخ =====
if selected_page == "🏠 خانه - پرسش و پاسخ":
    # ... (بخش فرم شما عالی است)
    
    # **تغییر کلیدی:** نحوه ارسال درخواست به API جدید
    if submitted and question.strip():
        with st.spinner("🤖 در حال پردازش سوال..."):
            # ساختار داده برای ResponseRequest
            request_data_body = {
                "question": question,
                "contexts": [], # در این حالت context توسط کاربر ارائه نمی‌شود
                "query_type": query_type if query_type != "خودکار" else None,
                "temperature": temperature,
                "include_citations": include_citations,
                # سایر فیلدهای ResponseRequest در صورت نیاز
            }

            # **اصلاح**: استفاده از متد ask_question و ارسال بدنه JSON
            response = api_client.ask_question(question=question, contexts=[], temperature=temperature, include_citations=include_citations)

            # ... (بقیه کد شما برای نمایش پاسخ عالی است و نیازی به تغییر ندارد)
            # فقط مطمئن شو که کلیدها با ResponseResult هماهنگ است
            # مثلاً: response.get('enhanced_response')

# ... (بقیه صفحات شما مانند داشبورد، آپلود، پردازش دسته‌ای و ... عالی هستند)
# فقط باید مطمئن شوی که endpoint ها و ساختارهای داده در کلاس APIClient
# با فایل api.py جدید مطابقت دارند. من این کار را در api.py انجام دادم.
# به عنوان مثال، endpoint پرسش حالا /ask/enhanced است.