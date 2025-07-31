"""
src/generation/citation_engine.py - سیستم Citation و ارجاع

این فایل مسئول شناسایی، ردیابی و فرمت‌بندی دقیق ارجاعات حقوقی است.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CitationType(Enum):
    """انواع ارجاعات حقوقی"""
    LAW_ARTICLE = "ماده_قانون"           # ماده قانون
    REGULATION_ARTICLE = "ماده_آیین_نامه"  # ماده آیین‌نامه
    CLAUSE = "بند"                       # بند
    NOTE = "تبصره"                      # تبصره
    CHAPTER = "فصل"                     # فصل
    CIRCULAR = "بخشنامه"                 # بخشنامه

@dataclass
class Citation:
    """ساختار یک ارجاع حقوقی"""
    source_document: str                 # نام سند اصلی
    citation_type: CitationType          # نوع ارجاع
    article_number: Optional[str] = None # شماره ماده
    clause_number: Optional[str] = None  # شماره بند
    note_number: Optional[str] = None    # شماره تبصره
    approval_date: Optional[str] = None  # تاریخ تصویب
    full_text: str = ""                  # متن کامل بخش ارجاع‌شده
    confidence_score: float = 0.0        # اطمینان از صحت ارجاع
    document_url: Optional[str] = None   # لینک به سند اصلی

class CitationEngine:
    """موتور اصلی سیستم Citation"""
    
    def __init__(self):
        self.citation_patterns = {}
        self.citation_formats = {}
        
        # بارگذاری الگوهای ارجاع
        self._load_citation_patterns()
        self._load_citation_formats()
        
        logger.info("CitationEngine ایجاد شد")
    
    def _load_citation_patterns(self) -> None:
        """بارگذاری الگوهای regex برای تشخیص ارجاعات"""
        
        self.citation_patterns = {
            # الگوی ماده قانون: "ماده 5 قانون ..."
            CitationType.LAW_ARTICLE: [
                r'ماده\s+(\d+)\s+([^،\.]+قانون[^،\.]*)',
                r'مواد\s+(\d+)\s+(?:تا|و)\s+(\d+)\s+([^،\.]+قانون[^،\.]*)',
            ],
            
            # الگوی ماده آیین‌نامه
            CitationType.REGULATION_ARTICLE: [
                r'ماده\s+(\d+)\s+([^،\.]*آیین\s*نامه[^،\.]*)',
                r'مواد\s+(\d+)\s+(?:تا|و)\s+(\d+)\s+([^،\.]*آیین\s*نامه[^،\.]*)'
            ],
            
            # الگوی بند
            CitationType.CLAUSE: [
                r'بند\s+([الف-ی]|\d+)\s+ماده\s+(\d+)',
                r'بندهای\s+([الف-ی]|\d+)\s+(?:تا|و)\s+([الف-ی]|\d+)\s+ماده\s+(\d+)'
            ],
            
            # الگوی تبصره
            CitationType.NOTE: [
                r'تبصره\s+(\d+)\s+ماده\s+(\d+)',
                r'تبصره\s+([الف-ی]|\d+)\s+بند\s+([الف-ی]|\d+)\s+ماده\s+(\d+)',
                r'تبصره\s+(\d+)\s+([^،\.]+(?:قانون|آیین\s*نامه)[^،\.]*)'
            ],
            
            # الگوی فصل
            CitationType.CHAPTER: [
                r'فصل\s+(\d+|[الف-ی]+)\s+([^،\.]+)',
                r'فصول\s+(\d+|[الف-ی]+)\s+(?:تا|و)\s+(\d+|[الف-ی]+)'
            ],
            
            # الگوی بخشنامه
            CitationType.CIRCULAR: [
                r'بخشنامه\s+(?:شماره\s+)?([^\s،\.]+)\s+(?:مورخ\s+)?([^\s،\.]+)',
                r'بخشنامه\s+([^،\.]+)'
            ]
        }
    
    def _load_citation_formats(self) -> None:
        """بارگذاری فرمت‌های مختلف ارجاع"""
        
        self.citation_formats = {
            'standard': {
                CitationType.LAW_ARTICLE: "ماده {article} {document}",
                CitationType.REGULATION_ARTICLE: "ماده {article} {document}",
                CitationType.CLAUSE: "بند {clause} ماده {article} {document}",
                CitationType.NOTE: "تبصره {note} ماده {article} {document}",
                CitationType.CHAPTER: "فصل {chapter} {document}",
                CitationType.CIRCULAR: "بخشنامه {document}"
            },
            
            'detailed': {
                CitationType.LAW_ARTICLE: "ماده {article} {document} (مصوب {date})",
                CitationType.REGULATION_ARTICLE: "ماده {article} {document} (مصوب {date})",
                CitationType.CLAUSE: "بند {clause} ماده {article} {document}",
                CitationType.NOTE: "تبصره {note} ماده {article} {document}",
            }
        }
    
    def extract_citations_from_text(self, text: str) -> List[Citation]:
        """استخراج تمام ارجاعات از متن"""
        citations = []
        
        for citation_type, patterns in self.citation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    citation = self._parse_citation_match(match, citation_type)
                    if citation:
                        citations.append(citation)
        
        # حذف تکراری‌ها
        unique_citations = self._remove_duplicate_citations(citations)
        
        logger.info(f"تعداد {len(unique_citations)} ارجاع در متن یافت شد")
        return unique_citations
    
    def _parse_citation_match(self, match: re.Match, citation_type: CitationType) -> Optional[Citation]:
        """تجزیه تطابق regex و ایجاد Citation"""
        
        groups = match.groups()
        if not groups:
            return None
        
        try:
            citation = Citation(
                source_document="",
                citation_type=citation_type,
                confidence_score=0.8  # امتیاز پایه
            )
            
            if citation_type == CitationType.LAW_ARTICLE:
                if len(groups) >= 2:
                    citation.article_number = groups[0]
                    citation.source_document = groups[1].strip()
                    
            elif citation_type == CitationType.REGULATION_ARTICLE:
                if len(groups) >= 2:
                    citation.article_number = groups[0]
                    citation.source_document = groups[1].strip()
                    
            elif citation_type == CitationType.CLAUSE:
                if len(groups) >= 2:
                    citation.clause_number = groups[0]
                    citation.article_number = groups[1]
                    if len(groups) >= 3:
                        citation.source_document = groups[2].strip()
                        
            elif citation_type == CitationType.NOTE:
                if len(groups) >= 2:
                    citation.note_number = groups[0]
                    citation.article_number = groups[1]
                    if len(groups) >= 3:
                        citation.source_document = groups[2].strip()
                        
            elif citation_type == CitationType.CIRCULAR:
                citation.source_document = groups[0].strip()
                if len(groups) >= 2:
                    citation.approval_date = groups[1].strip()
            
            # تنظیم متن کامل
            citation.full_text = match.group(0)
            
            return citation
            
        except Exception as e:
            logger.warning(f"خطا در تجزیه ارجاع: {e}")
            return None
    
    def _remove_duplicate_citations(self, citations: List[Citation]) -> List[Citation]:
        """حذف ارجاعات تکراری"""
        
        seen = set()
        unique_citations = []
        
        for citation in citations:
            # ایجاد کلید یکتا برای هر ارجاع
            key = (
                citation.source_document,
                citation.citation_type,
                citation.article_number,
                citation.clause_number,
                citation.note_number
            )
            
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)
        
        return unique_citations
    
    def format_citation(
        self, 
        citation: Citation, 
        format_style: str = 'standard',
        include_link: bool = True
    ) -> str:
        """فرمت‌بندی ارجاع با استایل مشخص"""
        
        if format_style not in self.citation_formats:
            format_style = 'standard'
        
        format_dict = self.citation_formats[format_style]
        
        if citation.citation_type not in format_dict:
            # فرمت پیش‌فرض
            formatted = citation.full_text
        else:
            template = format_dict[citation.citation_type]
            
            # آماده‌سازی متغیرها
            variables = {
                'document': citation.source_document,
                'article': citation.article_number or '',
                'clause': citation.clause_number or '',
                'note': citation.note_number or '',
                'chapter': citation.article_number or '',  # فرض فصل = شماره
                'date': citation.approval_date or ''
            }
            
            try:
                formatted = template.format(**variables)
            except KeyError as e:
                logger.warning(f"متغیر ناموجود در template: {e}")
                formatted = citation.full_text
        
        # اضافه کردن لینک
        if include_link and citation.document_url:
            formatted = f'<a href="{citation.document_url}" target="_blank">{formatted}</a>'
        
        return formatted
    
    def generate_citation_list(self, citations: List[Citation]) -> str:
        """تولید فهرست منابع"""
        
        if not citations:
            return "منابع مشخصی یافت نشد."
        
        # گروه‌بندی بر اساس نوع سند
        grouped = {}
        for citation in citations:
            doc_name = citation.source_document
            if doc_name not in grouped:
                grouped[doc_name] = []
            grouped[doc_name].append(citation)
        
        # ایجاد فهرست
        result = "## منابع مورد استفاده:\n\n"
        
        for i, (doc_name, doc_citations) in enumerate(grouped.items(), 1):
            result += f"{i}. **{doc_name}**\n"
            
            # اضافه کردن جزئیات
            articles = set()
            clauses = set()
            notes = set()
            
            for cit in doc_citations:
                if cit.article_number:
                    articles.add(cit.article_number)
                if cit.clause_number:
                    clauses.add(f"بند {cit.clause_number}")
                if cit.note_number:
                    notes.add(f"تبصره {cit.note_number}")
            
            details = []
            if articles:
                details.append(f"مواد: {', '.join(sorted(articles))}")
            if clauses:
                details.append(f"بندها: {', '.join(sorted(clauses))}")
            if notes:
                details.append(f"تبصره‌ها: {', '.join(sorted(notes))}")
            
            if details:
                result += f"   - {' | '.join(details)}\n"
            
            result += "\n"
        
        return result
    
    def validate_citations(self, citations: List[Citation]) -> Dict[str, Any]:
        """اعتبارسنجی ارجاعات"""
        
        validation_result = {
            'total_citations': len(citations),
            'valid_citations': 0,
            'invalid_citations': 0,
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'issues': []
        }
        
        for citation in citations:
            is_valid = True
            
            # بررسی وجود نام سند
            if not citation.source_document:
                validation_result['issues'].append(f"نام سند مشخص نیست: {citation.full_text}")
                is_valid = False
            
            # بررسی فرمت شماره ماده
            if citation.article_number and not re.match(r'^\d+$', citation.article_number):
                validation_result['issues'].append(f"شماره ماده نامعتبر: {citation.article_number}")
                is_valid = False
            
            # تحلیل اطمینان
            if citation.confidence_score >= 0.8:
                validation_result['confidence_distribution']['high'] += 1
            elif citation.confidence_score >= 0.6:
                validation_result['confidence_distribution']['medium'] += 1
            else:
                validation_result['confidence_distribution']['low'] += 1
            
            if is_valid:
                validation_result['valid_citations'] += 1
            else:
                validation_result['invalid_citations'] += 1
        
        # محاسبه درصد اعتبار کلی
        if validation_result['total_citations'] > 0:
            validation_result['accuracy_percentage'] = (
                validation_result['valid_citations'] / validation_result['total_citations'] * 100
            )
        else:
            validation_result['accuracy_percentage'] = 0
        
        return validation_result
    
    def enhance_response_with_citations(
        self, 
        response_text: str, 
        source_contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """تقویت پاسخ با ارجاعات دقیق"""
        
        # استخراج ارجاعات از پاسخ
        citations = self.extract_citations_from_text(response_text)
        
        # تولید فهرست منابع
        references_list = self.generate_citation_list(citations)
        
        # اعتبارسنجی
        validation = self.validate_citations(citations)
        
        return {
            'enhanced_response': response_text,
            'original_response': response_text,
            'citations': citations,
            'references_list': references_list,
            'validation': validation,
            'citation_count': len(citations)
        }

# تست
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # ایجاد موتور citation
    engine = CitationEngine()
    
    # نمونه متن پاسخ
    sample_response = """
    بر اساس ماده 3 قانون مقررات انتظامی هیئت علمی، اعضای هیئت علمی موظف به انجام پژوهش هستند.
    همچنین تبصره 1 ماده 5 آیین‌نامه ارتقای هیئت علمی نیز به این موضوع اشاره دارد.
    طبق بخشنامه 1234 وزارت علوم، این مقررات الزام‌آور است.
    """
    
    print("🔍 تست استخراج ارجاعات:")
    print(f"متن نمونه: {sample_response}")
    
    # استخراج ارجاعات
    citations = engine.extract_citations_from_text(sample_response)
    print(f"\n📋 تعداد ارجاعات یافته: {len(citations)}")
    
    for i, citation in enumerate(citations, 1):
        print(f"\n{i}. نوع: {citation.citation_type.value}")
        print(f"   سند: {citation.source_document}")
        if citation.article_number:
            print(f"   ماده: {citation.article_number}")
        if citation.note_number:
            print(f"   تبصره: {citation.note_number}")
        print(f"   اطمینان: {citation.confidence_score}")
        
        # تست فرمت‌بندی
        formatted = engine.format_citation(citation)
        print(f"   فرمت: {formatted}")
    
    # تست فهرست منابع
    print(f"\n📚 فهرست منابع:")
    references = engine.generate_citation_list(citations)
    print(references)
    
    # تست اعتبارسنجی
    print(f"\n✅ اعتبارسنجی:")
    validation = engine.validate_citations(citations)
    print(f"کل ارجاعات: {validation['total_citations']}")
    print(f"معتبر: {validation['valid_citations']}")
    print(f"درصد اعتبار: {validation['accuracy_percentage']:.1f}%")
    
    print("\n✅ تست CitationEngine کامل شد")