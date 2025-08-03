"""
interface/streamlit_legal_assistant.py - رابط کاربری Streamlit فاز 5

رابط کاربری جامع با قابلیت‌های:
- پرسش و پاسخ تعاملی
- آپلود و تحلیل فایل
- مدیریت session
- داشبورد آمار
- خروجی چندفرمته
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import io
import base64
from pathlib import Path

# تنظیمات صفحه
st.set_page_config(
    page_title="دستیار حقوقی هوشمند",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔽 کد جدید برای اعمال فونت Vazirmatn 🔽
st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');

html, body, [class*="st-"], .main, h1, h2, h3, h4, h5, h6 {
    font-family: 'Vazirmatn', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# تنظیمات API 
API_BASE_URL = "http://localhost:8000"

# CSS سفارشی برای بهبود ظاهر
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}

.metric-card {
    background: linear-gradient(45deg, #f0f8ff, #e6f3ff);
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #ddd;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.success-message {
    padding: 1rem;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 5px;
    color: #155724;
}

.error-message {
    padding: 1rem;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 5px;
    color: #721c24;
}

.question-history {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.response-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.citation-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    padding: 1rem;
    border-radius: 5px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# کامپوننت‌های کمکی
class APIClient:
    """کلاس برای ارتباط با API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session_id = None
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """ارسال درخواست به API"""
        url = f"{self.base_url}{endpoint}"
        
        # اضافه کردن session_id به header
        headers = kwargs.get('headers', {})
        if self.session_id:
            headers['X-Session-ID'] = self.session_id
        kwargs['headers'] = headers
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, **kwargs)
            elif method.upper() == 'POST':
                response = requests.post(url, **kwargs)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, **kwargs)
            else:
                raise ValueError(f"متد {method} پشتیبانی نمی‌شود")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            return {"error": True, "message": "خطا در اتصال به سرور"}
        except requests.exceptions.Timeout:
            return {"error": True, "message": "درخواست منقضی شد"}
        except requests.exceptions.RequestException as e:
            return {"error": True, "message": f"خطا در درخواست: {str(e)}"}
        except json.JSONDecodeError:
            return {"error": True, "message": "خطا در تجزیه پاسخ سرور"}
    
    def create_session(self) -> bool:
        """ایجاد session جدید"""
        result = self._make_request('POST', '/session/create')
        if not result.get('error') and result.get('success'):
            self.session_id = result.get('session_id')
            return True
        return False
    
    def ask_question(self, question: str, contexts: List = None, **kwargs) -> Dict:
        """پرسش از API"""
        data = {
            "question": question,
            "contexts": contexts or [],
            **kwargs
        }
        return self._make_request('POST', '/ask/enhanced', json=data)
    
    def get_health(self) -> Dict:
        """بررسی سلامت سیستم"""
        return self._make_request('GET', '/health')
    
    def get_stats(self) -> Dict:
        """دریافت آمار سیستم"""
        return self._make_request('GET', '/stats/system')
    
    def batch_process(self, questions: List[str]) -> Dict:
        """پردازش دسته‌ای"""
        data = {
            "questions": questions,
            "session_id": self.session_id
        }
        return self._make_request('POST', '/batch/process', json=data)
    
    def upload_file(self, file_content: bytes, filename: str) -> Dict:
        """آپلود فایل"""
        files = {'file': (filename, file_content)}
        return self._make_request('POST', '/upload/analyze', files=files)

# مقداردهی API client
@st.cache_resource
def get_api_client():
    return APIClient(API_BASE_URL)

api_client = get_api_client()

# مدیریت state
if 'session_initialized' not in st.session_state:
    st.session_state.session_initialized = False
    st.session_state.question_history = []
    st.session_state.current_response = None
    st.session_state.system_stats = {}

# راه‌اندازی session
if not st.session_state.session_initialized:
    with st.spinner('راه‌اندازی session...'):
        if api_client.create_session():
            st.session_state.session_initialized = True
            st.success(f"✅ Session ایجاد شد: {api_client.session_id[:8]}...")
        else:
            st.error("❌ خطا در ایجاد session")

# تابع‌های کمکی
def display_response_metrics(response_data: Dict):
    """نمایش معیارهای پاسخ"""
    if not response_data.get('success'):
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="زمان پردازش",
            value=f"{response_data.get('processing_time', 0):.2f}s",
            delta=None
        )
    
    with col2:
        quality_score = response_data.get('quality_score', 0)
        st.metric(
            label="کیفیت پاسخ",
            value=f"{quality_score:.2f}",
            delta=f"{'🟢' if quality_score > 0.7 else '🟡' if quality_score > 0.5 else '🔴'}"
        )
    
    with col3:
        confidence = response_data.get('confidence_score', 0)
        st.metric(
            label="اطمینان",
            value=f"{confidence:.2f}",
            delta=f"{'بالا' if confidence > 0.8 else 'متوسط' if confidence > 0.6 else 'پایین'}"
        )
    
    with col4:
        citations_count = len(response_data.get('citations', []))
        st.metric(
            label="تعداد ارجاعات",
            value=str(citations_count),
            delta=None
        )

def display_citations(citations: List[Dict]):
    """نمایش ارجاعات"""
    if not citations:
        return
    
    st.subheader("📚 منابع و ارجاعات")
    
    for i, citation in enumerate(citations, 1):
        with st.expander(f"منبع {i}: {citation.get('source', 'نامشخص')}"):
            st.write(f"**نوع سند:** {citation.get('document_type', 'نامشخص')}")
            if citation.get('article_number'):
                st.write(f"**ماده/بخش:** {citation['article_number']}")
            st.write(f"**امتیاز اطمینان:** {citation.get('confidence_score', 0):.2f}")
            if citation.get('context'):
                st.write(f"**متن مرجع:** {citation['context']}")

def export_session_data(format_type: str = "json"):
    """خروجی داده‌های session"""
    if not st.session_state.question_history:
        st.warning("هیچ داده‌ای برای خروجی وجود ندارد")
        return
    
    export_data = {
        "session_id": api_client.session_id,
        "export_time": datetime.now().isoformat(),
        "question_history": st.session_state.question_history,
        "total_questions": len(st.session_state.question_history)
    }
    
    if format_type == "json":
        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
        st.download_button(
            label="💾 دانلود JSON",
            data=json_str,
            file_name=f"legal_assistant_session_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
    
    elif format_type == "csv":
        df_data = []
        for i, item in enumerate(st.session_state.question_history):
            df_data.append({
                "شماره": i + 1,
                "سوال": item['question'],
                "خلاصه پاسخ": item['response'][:100] + "..." if len(item['response']) > 100 else item['response'],
                "زمان پردازش": item.get('processing_time', 0),
                "کیفیت": item.get('quality_score', 0),
                "زمان": item.get('timestamp', '')
            })
        
        df = pd.DataFrame(df_data)
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        
        st.download_button(
            label="📊 دانلود CSV",
            data=csv,
            file_name=f"legal_assistant_session_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

# ===== UI اصلی =====

# Header
st.markdown('<h1 class="main-header">⚖️ دستیار حقوقی هوشمند - فاز 5</h1>', unsafe_allow_html=True)

# Sidebar برای navigation
st.sidebar.title("🎛️ منوی اصلی")

# بررسی وضعیت سیستم در sidebar
with st.sidebar:
    st.subheader("🔍 وضعیت سیستم")
    
    if st.button("🔄 بررسی وضعیت", key="health_check"):
        with st.spinner("بررسی..."):
            health_data = api_client.get_health()
            
            if health_data.get('error'):
                st.error(f"❌ {health_data['message']}")
            else:
                st.success("✅ سیستم فعال")
                st.write(f"**نسخه:** {health_data.get('api_version', 'نامشخص')}")
                st.write(f"**وضعیت Ollama:** {health_data.get('ollama_status', 'نامشخص')}")
                st.write(f"**Session های فعال:** {health_data.get('active_sessions', 0)}")

# انتخاب صفحه
page_options = [
    "🏠 خانه - پرسش و پاسخ",
    "📊 داشبورد آمار",
    "📁 آپلود و تحلیل فایل", 
    "🔄 پردازش دسته‌ای",
    "📜 تاریخچه سوالات",
    "⚙️ تنظیمات و خروجی"
]

selected_page = st.sidebar.selectbox("انتخاب صفحه:", page_options)

# ===== صفحه اصلی - پرسش و پاسخ =====
if selected_page == "🏠 خانه - پرسش و پاسخ":
    st.header("💬 پرسش از دستیار حقوقی")
    
    # ستون‌ها برای Layout بهتر
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # فرم پرسش
        with st.form("question_form", clear_on_submit=False):
            st.subheader("📝 سوال خود را مطرح کنید")
            
            question = st.text_area(
                "سوال:",
                height=100,
                placeholder="مثال: وظایف اعضای هیئت علمی در زمینه پژوهش چیست؟",
                help="سوال خود را به صورت واضح و دقیق مطرح کنید"
            )
            
            # تنظیمات پیشرفته
            with st.expander("⚙️ تنظیمات پیشرفته"):
                query_type = st.selectbox(
                    "نوع سوال:",
                    ["خودکار", "سوال_عمومی", "تحلیل_سند", "مقایسه_اسناد", "بررسی_قرارداد", "مشاوره_حقوقی"],
                    help="انتخاب نوع سوال برای بهبود کیفیت پاسخ"
                )
                
                temperature = st.slider(
                    "خلاقیت پاسخ:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.1,
                    help="مقادیر بالاتر = پاسخ‌های خلاق‌تر، مقادیر پایین‌تر = پاسخ‌های دقیق‌تر"
                )
                
                include_citations = st.checkbox("شامل ارجاعات", value=True)
            
            # دکمه ارسال
            submitted = st.form_submit_button("🚀 ارسال سوال", type="primary")
            
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
                    
                    # ارسال درخواست
                    response = api_client.ask_question(question=question, contexts=[], temperature=temperature, include_citations=include_citations)
                    
                    if response.get('error'):
                        st.error(f"❌ خطا: {response['message']}")
                    else:
                        # ذخیره در تاریخچه
                        history_item = {
                            "question": question,
                            "response": response.get('enhanced_response', ''),
                            "processing_time": response.get('processing_time', 0),
                            "quality_score": response.get('quality_score', 0),
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        st.session_state.question_history.append(history_item)
                        st.session_state.current_response = response
                        
                        st.success("✅ پاسخ تولید شد!")
    
    with col2:
        # راهنمای سوالات نمونه
        st.subheader("💡 سوالات نمونه")
        
        sample_questions = [
            "وظایف اعضای هیئت علمی در پژوهش چیست؟",
            "مزایای شرکت‌های دانش‌بنیان کدامند؟",
            "فرآیند انتقال فناوری چگونه است؟",
            "معیارهای ارتقای هیئت علمی چیست؟",
            "تعریف شرکت دانش‌بنیان را بیان کنید"
        ]
        
        for i, sample in enumerate(sample_questions, 1):
            if st.button(f"📌 {sample}", key=f"sample_{i}"):
                st.session_state.sample_question = sample
    
    # نمایش پاسخ
    if st.session_state.current_response:
        st.markdown("---")
        st.header("📋 پاسخ دستیار حقوقی")
        
        response = st.session_state.current_response
        
        if response.get('success'):
            # نمایش معیارها
            display_response_metrics(response)
            
            st.markdown("---")
            
            # پاسخ اصلی
            st.markdown(
                f'<div class="response-box">'
                f'<h3>📝 پاسخ:</h3>'
                f'<p>{response.get("enhanced_response", "")}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # ارجاعات
            if response.get('citations'):
                display_citations(response['citations'])
            
            # اطلاعات تکمیلی
            with st.expander("ℹ️ اطلاعات تکمیلی"):
                st.write(f"**نوع سوال تشخیص داده شده:** {response.get('query_type', 'نامشخص')}")
                st.write(f"**مدل استفاده شده:** {response.get('model_used', 'نامشخص')}")
                st.write(f"**زمان تولید:** {response.get('timestamp', 'نامشخص')}")
                
                if response.get('from_cache'):
                    st.info("📦 این پاسخ از کش بازیابی شد")
        
        else:
            st.error(f"❌ خطا در تولید پاسخ: {response.get('error_message', 'خطای نامشخص')}")

# ===== داشبورد آمار =====
elif selected_page == "📊 داشبورد آمار":
    st.header("📊 داشبورد آمار و معیارهای عملکرد")
    
    # دریافت آمار سیستم
    if st.button("🔄 بروزرسانی آمار", key="refresh_stats"):
        with st.spinner("دریافت آمار..."):
            stats = api_client.get_stats()
            if not stats.get('error'):
                st.session_state.system_stats = stats
    
    if st.session_state.system_stats:
        stats = st.session_state.system_stats
        
        # آمار کلی سیستم
        st.subheader("🎯 آمار کلی سیستم")
        
        col1, col2, col3, col4 = st.columns(4)
        
        system_stats = stats.get('system', {})
        
        with col1:
            st.metric(
                "کل درخواست‌ها",
                system_stats.get('total_requests', 0)
            )
        
        with col2:
            success_rate = 0
            total = system_stats.get('total_requests', 0)
            if total > 0:
                success_rate = (system_stats.get('successful_requests', 0) / total) * 100
            st.metric(
                "نرخ موفقیت",
                f"{success_rate:.1f}%"
            )
        
        with col3:
            st.metric(
                "میانگین زمان پاسخ",
                f"{system_stats.get('average_response_time', 0):.2f}s"
            )
        
        with col4:
            st.metric(
                "Session های فعال",
                stats.get('sessions', {}).get('active_sessions', 0)
            )
        
        # نمودار عملکرد در طول زمان (Mock data)
        st.subheader("📈 عملکرد در طول زمان")
        
        # تولید داده‌های نمونه برای نمودار
        dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='D')
        performance_data = pd.DataFrame({
            'تاریخ': dates,
            'تعداد سوالات': [15, 23, 18, 31, 27, 35, 29],
            'میانگین کیفیت': [0.75, 0.82, 0.79, 0.85, 0.88, 0.83, 0.86],
            'زمان پاسخ (ثانیه)': [8.2, 7.5, 9.1, 6.8, 7.2, 8.0, 7.6]
        })
        
        # نمودار تعداد سوالات
        fig1 = px.line(performance_data, x='تاریخ', y='تعداد سوالات', 
                      title='تعداد سوالات در طول زمان')
        st.plotly_chart(fig1, use_container_width=True)
        
        # نمودار کیفیت و زمان پاسخ
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=performance_data['تاریخ'], y=performance_data['میانگین کیفیت'],
                                 mode='lines+markers', name='کیفیت پاسخ', yaxis='y'))
        fig2.add_trace(go.Scatter(x=performance_data['تاریخ'], y=performance_data['زمان پاسخ (ثانیه)'],
                                 mode='lines+markers', name='زمان پاسخ', yaxis='y2'))
        
        fig2.update_layout(
            title='کیفیت پاسخ و زمان پردازش',
            yaxis=dict(title='کیفیت پاسخ', side='left'),
            yaxis2=dict(title='زمان پاسخ (ثانیه)', side='right', overlaying='y')
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # آمار سیستم حقوقی
        if 'legal_system' in stats:
            st.subheader("⚖️ آمار سیستم حقوقی")
            legal_stats = stats['legal_system']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**اطلاعات LLM:**")
                llm_info = legal_stats.get('llm_info', {})
                st.json(llm_info)
            
            with col2:
                st.write("**آمار کش:**")
                st.write(f"اندازه کش: {legal_stats.get('cache_size', 0)}")
                st.write(f"Template ها: {legal_stats.get('prompt_templates', 0)}")

# ===== آپلود و تحلیل فایل =====
elif selected_page == "📁 آپلود و تحلیل فایل":
    st.header("📁 آپلود و تحلیل فایل")
    
    st.markdown("""
    **قابلیت‌های تحلیل فایل:**
    - تحلیل اسناد حقوقی (PDF, DOCX, TXT)
    - بررسی انطباق با مقررات
    - استخراج بخش‌های مهم
    - ارائه پیشنهادات بهبود
    """)
    
    # آپلود فایل
    uploaded_file = st.file_uploader(
        "فایل خود را انتخاب کنید:",
        type=['pdf', 'docx', 'txt'],
        help="فرمت‌های پشتیبانی‌شده: PDF, DOCX, TXT (حداکثر 10MB)"
    )
    
    if uploaded_file:
        st.success(f"✅ فایل انتخاب شد: {uploaded_file.name}")
        
        # نمایش اطلاعات فایل
        file_size = len(uploaded_file.getvalue())
        st.write(f"**اندازه فایل:** {file_size / 1024:.1f} KB")
        st.write(f"**نوع فایل:** {uploaded_file.type}")
        
        # تنظیمات تحلیل
        analysis_type = st.selectbox(
            "نوع تحلیل:",
            ["legal_compliance", "document_structure", "content_analysis"],
            format_func=lambda x: {
                "legal_compliance": "انطباق حقوقی",
                "document_structure": "ساختار سند", 
                "content_analysis": "تحلیل محتوا"
            }[x]
        )
        
        include_recommendations = st.checkbox("شامل پیشنهادات بهبود", value=True)
        
        if st.button("🔍 شروع تحلیل", type="primary"):
            with st.spinner("در حال تحلیل فایل..."):
                # آپلود و تحلیل
                file_content = uploaded_file.getvalue()
                result = api_client.upload_file(file_content, uploaded_file.name)
                
                if result.get('error'):
                    st.error(f"❌ خطا: {result['message']}")
                else:
                    st.success("✅ تحلیل با موفقیت انجام شد!")
                    
                    # نمایش نتایج
                    st.subheader("📊 نتایج تحلیل")
                    
                    analysis = result.get('analysis', {})
                    
                    # معیارهای کلی
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("نوع سند", analysis.get('document_type', 'نامشخص'))
                    
                    with col2:
                        structure_score = analysis.get('structure_score', 0)
                        st.metric("امتیاز ساختار", f"{structure_score:.2f}")
                    
                    with col3:
                        compliance_score = analysis.get('compliance_score', 0)
                        st.metric("امتیاز انطباق", f"{compliance_score:.2f}")
                    
                    # بخش‌های استخراج شده
                    if 'extracted_sections' in result:
                        st.subheader("📑 بخش‌های شناسایی شده")
                        for i, section in enumerate(result['extracted_sections'], 1):
                            st.write(f"{i}. {section}")
                    
                    # پیشنهادات
                    if analysis.get('recommendations'):
                        st.subheader("💡 پیشنهادات بهبود")
                        for rec in analysis['recommendations']:
                            st.write(f"• {rec}")
                    
                    # گزارش کامل
                    with st.expander("📋 گزارش کامل"):
                        st.json(result)

# ===== پردازش دسته‌ای =====
elif selected_page == "🔄 پردازش دسته‌ای":
    st.header("🔄 پردازش دسته‌ای سوالات")
    
    st.markdown("""
    **پردازش دسته‌ای** امکان ارسال چندین سوال به صورت همزمان را فراهم می‌کند.
    این قابلیت برای پردازش حجم بالای سوالات مناسب است.
    """)
    
    # ورودی سوالات
    st.subheader("📝 سوالات خود را وارد کنید")
    
    # روش ورودی
    input_method = st.radio(
        "روش ورودی:",
        ["متنی (هر سوال در یک خط)", "فایل CSV"]
    )
    
    questions = []
    
    if input_method == "متنی (هر سوال در یک خط)":
        questions_text = st.text_area(
            "سوالات:",
            height=200,
            placeholder="""وظایف اعضای هیئت علمی چیست؟
مزایای شرکت‌های دانش‌بنیان کدامند؟
فرآیند انتقال فناوری چگونه است؟""",
            help="هر سوال را در یک خط جداگانه وارد کنید"
        )
        
        if questions_text.strip():
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
    
    else:
        # آپلود فایل CSV
        csv_file = st.file_uploader(
            "فایل CSV سوالات:",
            type=['csv'],
            help="فایل CSV باید ستونی با نام 'question' داشته باشد"
        )
        
        if csv_file:
            try:
                df = pd.read_csv(csv_file)
                if 'question' in df.columns:
                    questions = df['question'].dropna().tolist()
                    st.write(f"✅ {len(questions)} سوال از فایل خوانده شد")
                else:
                    st.error("❌ فایل CSV باید ستون 'question' داشته باشد")
            except Exception as e:
                st.error(f"❌ خطا در خواندن فایل: {e}")
    
    # نمایش سوالات
    if questions:
        st.subheader(f"📋 سوالات آماده پردازش ({len(questions)} سوال)")
        
        with st.expander("مشاهده سوالات"):
            for i, q in enumerate(questions, 1):
                st.write(f"{i}. {q}")
        
        # محدودیت تعداد
        if len(questions) > 50:
            st.warning("⚠️ حداکثر 50 سوال در هر دسته قابل پردازش است")
            questions = questions[:50]
        
        # دکمه پردازش
        if st.button("🚀 شروع پردازش دسته‌ای", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("در حال پردازش سوالات..."):
                result = api_client.batch_process(questions)
                
                if result.get('error'):
                    st.error(f"❌ خطا: {result['message']}")
                else:
                    st.success("✅ پردازش دسته‌ای کامل شد!")
                    
                    # نمایش خلاصه نتایج
                    st.subheader("📊 خلاصه نتایج")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("کل سوالات", result.get('total_questions', 0))
                    
                    with col2:
                        st.metric("موفق", result.get('successful_questions', 0))
                    
                    with col3:
                        success_rate = result.get('success_rate', 0)
                        st.metric("نرخ موفقیت", f"{success_rate:.1f}%")
                    
                    with col4:
                        avg_time = result.get('average_time_per_question', 0)
                        st.metric("میانگین زمان", f"{avg_time:.2f}s")
                    
                    # نتایج تفصیلی
                    st.subheader("📋 نتایج تفصیلی")
                    
                    results_data = []
                    for item in result.get('results', []):
                        results_data.append({
                            "شماره": item.get('question_index', 0) + 1,
                            "سوال": item.get('question', '')[:50] + "...",
                            "وضعیت": "✅ موفق" if item.get('success') else "❌ خطا",
                            "کیفیت": f"{item.get('quality_score', 0):.2f}",
                            "زمان": f"{item.get('processing_time', 0):.2f}s",
                            "خطا": item.get('error_message', '') if not item.get('success') else ""
                        })
                    
                    df_results = pd.DataFrame(results_data)
                    st.dataframe(df_results, use_container_width=True)
                    
                    # دانلود نتایج
                    csv_results = df_results.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        "💾 دانلود نتایج CSV",
                        csv_results,
                        f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv"
                    )

# ===== تاریخچه سوالات =====
elif selected_page == "📜 تاریخچه سوالات":
    st.header("📜 تاریخچه سوالات و پاسخ‌ها")
    
    if not st.session_state.question_history:
        st.info("📝 هنوز سوالی پرسیده نشده است")
    else:
        st.write(f"**تعداد کل سوالات:** {len(st.session_state.question_history)}")
        
        # فیلتر و جستجو
        col1, col2 = st.columns(2)
        
        with col1:
            search_term = st.text_input("🔍 جستجو در سوالات:", "")
        
        with col2:
            sort_by = st.selectbox("مرتب‌سازی بر اساس:", ["جدیدترین", "کیفیت بالا", "زمان پردازش"])
        
        # فیلتر کردن
        filtered_history = st.session_state.question_history
        
        if search_term:
            filtered_history = [
                item for item in filtered_history 
                if search_term.lower() in item['question'].lower() or 
                   search_term.lower() in item['response'].lower()
            ]
        
        # مرتب‌سازی
        if sort_by == "کیفیت بالا":
            filtered_history = sorted(filtered_history, key=lambda x: x.get('quality_score', 0), reverse=True)
        elif sort_by == "زمان پردازش":
            filtered_history = sorted(filtered_history, key=lambda x: x.get('processing_time', 0))
        else:  # جدیدترین
            filtered_history = list(reversed(filtered_history))
        
        # نمایش تاریخچه
        for i, item in enumerate(filtered_history):
            with st.expander(f"سوال {i+1}: {item['question'][:60]}..."):
                st.write(f"**زمان:** {item['timestamp']}")
                st.write(f"**سوال:** {item['question']}")
                st.write(f"**پاسخ:** {item['response']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("زمان پردازش", f"{item.get('processing_time', 0):.2f}s")
                with col2:
                    st.metric("کیفیت", f"{item.get('quality_score', 0):.2f}")
                with col3:
                    if st.button(f"🔄 پرسش مجدد", key=f"reask_{i}"):
                        st.session_state.reask_question = item['question']
        
        # آمار تاریخچه
        if filtered_history:
            st.subheader("📊 آمار تاریخچه")
            
            avg_quality = sum(item.get('quality_score', 0) for item in filtered_history) / len(filtered_history)
            avg_time = sum(item.get('processing_time', 0) for item in filtered_history) / len(filtered_history)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("میانگین کیفیت", f"{avg_quality:.2f}")
            with col2:
                st.metric("میانگین زمان", f"{avg_time:.2f}s")
            with col3:
                st.metric("تعداد فیلتر شده", len(filtered_history))

# ===== تنظیمات و خروجی =====
elif selected_page == "⚙️ تنظیمات و خروجی":
    st.header("⚙️ تنظیمات و خروجی")
    
    # تنظیمات session
    st.subheader("🔧 تنظیمات Session")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🆕 ایجاد Session جدید"):
            if api_client.create_session():
                st.session_state.question_history = []
                st.session_state.current_response = None
                st.success(f"✅ Session جدید: {api_client.session_id[:8]}...")
            else:
                st.error("❌ خطا در ایجاد session")
    
    with col2:
        if st.button("🗑️ پاک کردن تاریخچه"):
            st.session_state.question_history = []
            st.session_state.current_response = None
            st.success("✅ تاریخچه پاک شد")
    
    st.write(f"**Session فعلی:** {api_client.session_id[:16] if api_client.session_id else 'نامشخص'}...")
    
    # خروجی داده‌ها
    st.subheader("💾 خروجی داده‌ها")
    
    if st.session_state.question_history:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 دانلود JSON"):
                export_session_data("json")
        
        with col2:
            if st.button("📊 دانلود CSV"):
                export_session_data("csv")
    else:
        st.info("📝 هیچ داده‌ای برای خروجی وجود ندارد")
    
    # تنظیمات سیستم
    st.subheader("🔧 تنظیمات سیستم")
    
    if st.button("🧹 پاک کردن کش سیستم"):
        with st.spinner("پاک کردن کش..."):
            result = api_client._make_request('DELETE', '/cache/clear')
            
            if result.get('error'):
                st.error(f"❌ خطا: {result['message']}")
            else:
                st.success("✅ کش پاک شد")
    
    # درباره سیستم
    st.subheader("ℹ️ درباره سیستم")
    
    st.markdown("""
    **دستیار حقوقی هوشمند - فاز 5**
    
    نسخه: 5.0.0
    
    **ویژگی‌ها:**
    - پاسخگویی هوشمند با Ollama
    - مدیریت Session پیشرفته
    - پردازش دسته‌ای سوالات
    - تحلیل و آپلود فایل
    - داشبورد آمار کامل
    - خروجی چندفرمته
    - کش هوشمند
    - Rate Limiting
    
    **تکنولوژی‌ها:**
    - Backend: FastAPI + Python
    - Frontend: Streamlit
    - LLM: Ollama (qwen2.5:7b, deepseek-r1:7b, mistral)
    - Database: ChromaDB
    - Vector Embeddings: sentence-transformers
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🤖 دستیار حقوقی هوشمند - فاز 5 | "
    "ساخته شده با ❤️ برای بهبود دسترسی به اطلاعات حقوقی"
    "</div>",
    unsafe_allow_html=True
)