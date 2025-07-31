"""
src/evaluation/metrics_calculator.py - محاسبه معیارهای ارزیابی فاز 4

این فایل مسئول محاسبه تمام معیارهای ارزیابی سیستم است:
- Retrieval Metrics: Precision@K, Recall@K, MRR
- Generation Metrics: BLEU, ROUGE, BERTScore  
- Legal Accuracy: تطابق با منابع قانونی
- Citation Quality: دقت ارجاعات
"""

import re
import math
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from collections import Counter
import statistics

# Import برای معیارهای NLP
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  rouge-score نصب نیست. pip install rouge-score")

try:
    import sacrebleu
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("⚠️  sacrebleu نصب نیست. pip install sacrebleu")

logger = logging.getLogger(__name__)

@dataclass
class RetrievalMetrics:
    """معیارهای بازیابی"""
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float  # Mean Reciprocal Rank
    map_score: float  # Mean Average Precision
    ndcg_at_5: float  # Normalized Discounted Cumulative Gain

@dataclass 
class GenerationMetrics:
    """معیارهای تولید متن"""
    bleu_1: float
    bleu_2: float
    bleu_4: float
    rouge_1_f: float
    rouge_2_f: float
    rouge_l_f: float
    meteor_score: float
    bert_score_f1: float

@dataclass
class LegalAccuracyMetrics:
    """معیارهای دقت حقوقی"""
    citation_accuracy: float      # دقت ارجاعات
    legal_term_coverage: float    # پوشش اصطلاحات حقوقی
    factual_consistency: float    # سازگاری واقعی
    law_compliance: float         # انطباق با قوانین
    completeness_score: float     # کامل بودن پاسخ

@dataclass
class CitationMetrics:
    """معیارهای کیفیت ارجاع"""
    citation_count: int
    valid_citations: int
    citation_precision: float
    citation_recall: float
    citation_format_score: float
    source_diversity: float

@dataclass
class OverallMetrics:
    """معیارهای کلی"""
    overall_score: float
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics  
    legal_accuracy: LegalAccuracyMetrics
    citation_quality: CitationMetrics

class LegalMetricsCalculator:
    """محاسبه‌گر معیارهای حقوقی"""
    
    def __init__(self):
        self.legal_terms = self._load_legal_terms()
        self.citation_patterns = self._load_citation_patterns()
        
        # تنظیم ROUGE scorer
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=False
            )
        
        logger.info("LegalMetricsCalculator راه‌اندازی شد")
    
    def _load_legal_terms(self) -> Set[str]:
        """بارگذاری اصطلاحات حقوقی"""
        return {
            # اصطلاحات کلیدی
            "ماده", "بند", "تبصره", "قانون", "آیین‌نامه", "دستورالعمل",
            "مصوب", "مجلس", "شورای عالی", "وزارت", "هیئت علمی",
            "پژوهش", "تحقیق", "فناوری", "دانش‌بنیان", "انتقال فناوری",
            
            # عبارات حقوقی
            "بر اساس", "مطابق", "طبق", "وفق", "در راستای",
            "با رعایت", "مشروط بر", "در صورت", "الزام", "ممنوع",
            "مجاز", "قابل قبول", "غیرقابل قبول", "ضروری", "اختیاری",
            
            # نهادها
            "دانشگاه", "مؤسسه", "مرکز تحقیقات", "شرکت", "سازمان",
            "معاونت", "دفتر", "ستاد", "کمیسیون", "هیئت"
        }
    
    def _load_citation_patterns(self) -> List[str]:
        """الگوهای ارجاع معتبر"""
        return [
            r"ماده\s+\d+",                    # ماده ۳
            r"بند\s+[آ-ی]+",                  # بند الف
            r"تبصره\s+\d+",                   # تبصره ۱
            r"قانون\s+.+مصوب\s+\d{4}",       # قانون ... مصوب ۱۳۹۸
            r"آیین‌نامه\s+.+مصوب\s+\d{4}",    # آیین‌نامه ... مصوب ۱۳۹۸
            r"دستورالعمل\s+.+",              # دستورالعمل ...
            r"بر اساس\s+ماده\s+\d+",         # بر اساس ماده ۳
            r"مطابق\s+.+",                   # مطابق قانون
            r"طبق\s+.+"                      # طبق آیین‌نامه
        ]
    
    # === Retrieval Metrics ===
    
    def calculate_precision_at_k(
        self, 
        retrieved_docs: List[Dict], 
        relevant_docs: List[str], 
        k: int
    ) -> float:
        """محاسبه Precision@K"""
        
        if not retrieved_docs or k <= 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_count = 0
        
        for doc in top_k:
            doc_id = doc.get('document_id') or doc.get('source', '')
            if any(rel_id in doc_id for rel_id in relevant_docs):
                relevant_count += 1
        
        return relevant_count / min(len(top_k), k)
    
    def calculate_recall_at_k(
        self, 
        retrieved_docs: List[Dict], 
        relevant_docs: List[str], 
        k: int
    ) -> float:
        """محاسبه Recall@K"""
        
        if not relevant_docs:
            return 0.0
        
        top_k = retrieved_docs[:k]
        found_relevant = 0
        
        for doc in top_k:
            doc_id = doc.get('document_id') or doc.get('source', '')
            if any(rel_id in doc_id for rel_id in relevant_docs):
                found_relevant += 1
        
        return found_relevant / len(relevant_docs)
    
    def calculate_mrr(
        self, 
        queries_results: List[Tuple[List[Dict], List[str]]]
    ) -> float:
        """محاسبه Mean Reciprocal Rank"""
        
        reciprocal_ranks = []
        
        for retrieved_docs, relevant_docs in queries_results:
            rr = 0.0
            
            for rank, doc in enumerate(retrieved_docs, 1):
                doc_id = doc.get('document_id') or doc.get('source', '')
                if any(rel_id in doc_id for rel_id in relevant_docs):
                    rr = 1.0 / rank
                    break
            
            reciprocal_ranks.append(rr)
        
        return statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_ndcg_at_k(
        self, 
        retrieved_docs: List[Dict], 
        relevant_docs: Dict[str, float], 
        k: int
    ) -> float:
        """محاسبه NDCG@K"""
        
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        # محاسبه DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            doc_id = doc.get('document_id') or doc.get('source', '')
            relevance = 0.0
            
            for rel_id, score in relevant_docs.items():
                if rel_id in doc_id:
                    relevance = score
                    break
            
            if i == 0:
                dcg += relevance
            else:
                dcg += relevance / math.log2(i + 1)
        
        # محاسبه IDCG (DCG بهینه)
        sorted_relevances = sorted(relevant_docs.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevances):
            if i == 0:
                idcg += relevance
            else:
                idcg += relevance / math.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_retrieval_metrics(
        self, 
        retrieved_docs: List[Dict], 
        relevant_docs: List[str],
        relevance_scores: Dict[str, float] = None
    ) -> RetrievalMetrics:
        """محاسبه تمام معیارهای بازیابی"""
        
        if relevance_scores is None:
            relevance_scores = {doc: 1.0 for doc in relevant_docs}
        
        return RetrievalMetrics(
            precision_at_1=self.calculate_precision_at_k(retrieved_docs, relevant_docs, 1),
            precision_at_3=self.calculate_precision_at_k(retrieved_docs, relevant_docs, 3),
            precision_at_5=self.calculate_precision_at_k(retrieved_docs, relevant_docs, 5),
            recall_at_1=self.calculate_recall_at_k(retrieved_docs, relevant_docs, 1),
            recall_at_3=self.calculate_recall_at_k(retrieved_docs, relevant_docs, 3),
            recall_at_5=self.calculate_recall_at_k(retrieved_docs, relevant_docs, 5),
            mrr=self.calculate_mrr([(retrieved_docs, relevant_docs)]),
            map_score=self._calculate_average_precision(retrieved_docs, relevant_docs),
            ndcg_at_5=self.calculate_ndcg_at_k(retrieved_docs, relevance_scores, 5)
        )
    
    def _calculate_average_precision(
        self, 
        retrieved_docs: List[Dict], 
        relevant_docs: List[str]
    ) -> float:
        """محاسبه Average Precision"""
        
        if not relevant_docs:
            return 0.0
        
        precision_sum = 0.0
        relevant_found = 0
        
        for i, doc in enumerate(retrieved_docs):
            doc_id = doc.get('document_id') or doc.get('source', '')
            
            if any(rel_id in doc_id for rel_id in relevant_docs):
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs)
    
    # === Generation Metrics ===
    
    def calculate_bleu_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """محاسبه امتیازات BLEU"""
        
        if not BLEU_AVAILABLE:
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0}
        
        try:
            # تمیز کردن متن‌ها
            gen_clean = self._clean_text_for_bleu(generated)
            ref_clean = self._clean_text_for_bleu(reference)
            
            # محاسبه BLEU
            bleu_1 = sacrebleu.sentence_bleu(gen_clean, [ref_clean], smooth_method='exp').score
            bleu_2 = sacrebleu.sentence_bleu(gen_clean, [ref_clean], smooth_method='exp').score  
            bleu_4 = sacrebleu.sentence_bleu(gen_clean, [ref_clean]).score
            
            return {
                "bleu_1": bleu_1 / 100.0,  # تبدیل به مقیاس 0-1
                "bleu_2": bleu_2 / 100.0,
                "bleu_4": bleu_4 / 100.0
            }
        except Exception as e:
            logger.warning(f"خطا در محاسبه BLEU: {e}")
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0}
    
    def calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """محاسبه امتیازات ROUGE"""
        
        if not ROUGE_AVAILABLE:
            return {"rouge_1_f": 0.0, "rouge_2_f": 0.0, "rouge_l_f": 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, generated)
            
            return {
                "rouge_1_f": scores['rouge1'].fmeasure,
                "rouge_2_f": scores['rouge2'].fmeasure,
                "rouge_l_f": scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"خطا در محاسبه ROUGE: {e}")
            return {"rouge_1_f": 0.0, "rouge_2_f": 0.0, "rouge_l_f": 0.0}
    
    def calculate_meteor_score(self, generated: str, reference: str) -> float:
        """محاسبه METEOR (تقریبی برای فارسی)"""
        
        # پیاده‌سازی ساده METEOR برای فارسی
        gen_words = set(generated.split())
        ref_words = set(reference.split())
        
        if not ref_words:
            return 0.0
        
        # محاسبه precision و recall
        matches = len(gen_words.intersection(ref_words))
        precision = matches / len(gen_words) if gen_words else 0.0
        recall = matches / len(ref_words)
        
        # F-measure with recall bias
        if precision + recall == 0:
            return 0.0
        
        f_mean = (10 * precision * recall) / (9 * precision + recall)
        
        return f_mean
    
    def _clean_text_for_bleu(self, text: str) -> str:
        """تمیز کردن متن برای BLEU"""
        # حذف کاراکترهای اضافی
        text = re.sub(r'[^\w\s]', '', text)
        # تبدیل به کلمات
        words = text.split()
        return ' '.join(words)
    
    def calculate_generation_metrics(
        self, 
        generated: str, 
        reference: str
    ) -> GenerationMetrics:
        """محاسبه تمام معیارهای تولید"""
        
        bleu_scores = self.calculate_bleu_scores(generated, reference)
        rouge_scores = self.calculate_rouge_scores(generated, reference)
        meteor = self.calculate_meteor_score(generated, reference)
        
        return GenerationMetrics(
            bleu_1=bleu_scores["bleu_1"],
            bleu_2=bleu_scores["bleu_2"], 
            bleu_4=bleu_scores["bleu_4"],
            rouge_1_f=rouge_scores["rouge_1_f"],
            rouge_2_f=rouge_scores["rouge_2_f"],
            rouge_l_f=rouge_scores["rouge_l_f"],
            meteor_score=meteor,
            bert_score_f1=0.0  # نیاز به پیاده‌سازی جداگانه
        )
    
    # === Legal Accuracy Metrics ===
    
    def calculate_citation_accuracy(
        self, 
        generated_text: str, 
        expected_citations: List[str]
    ) -> float:
        """دقت ارجاعات حقوقی"""
        
        # استخراج ارجاعات از متن تولید شده
        found_citations = self._extract_citations(generated_text)
        
        if not expected_citations:
            return 1.0 if not found_citations else 0.5
        
        # بررسی تطابق
        correct_citations = 0
        for expected in expected_citations:
            if any(expected in found for found in found_citations):
                correct_citations += 1
        
        return correct_citations / len(expected_citations)
    
    def calculate_legal_term_coverage(self, generated_text: str) -> float:
        """پوشش اصطلاحات حقوقی"""
        
        text_words = set(generated_text.lower().split())
        found_terms = len(text_words.intersection(self.legal_terms))
        
        # نرمال‌سازی بر اساس طول متن
        text_length_factor = min(len(text_words) / 50, 1.0)  # حداکثر 50 کلمه
        
        return (found_terms / 10) * text_length_factor  # حداکثر 10 اصطلاح
    
    def calculate_factual_consistency(
        self, 
        generated_text: str, 
        source_contexts: List[str]
    ) -> float:
        """سازگاری واقعی با منابع"""
        
        if not source_contexts:
            return 0.5
        
        # استخراج ادعاهای اصلی از متن تولید شده
        generated_claims = self._extract_claims(generated_text)
        
        # بررسی تطابق با منابع
        consistent_claims = 0
        for claim in generated_claims:
            if any(self._is_claim_supported(claim, context) for context in source_contexts):
                consistent_claims += 1
        
        return consistent_claims / len(generated_claims) if generated_claims else 0.5
    
    def calculate_law_compliance(self, generated_text: str) -> float:
        """انطباق با ساختار حقوقی"""
        
        compliance_score = 0.0
        
        # بررسی وجود ارجاع به منابع قانونی
        if re.search(r"(ماده|بند|تبصره|قانون|آیین‌نامه)", generated_text):
            compliance_score += 0.3
        
        # بررسی استفاده از عبارات حقوقی
        legal_phrases = ["بر اساس", "مطابق", "طبق", "وفق"]
        if any(phrase in generated_text for phrase in legal_phrases):
            compliance_score += 0.3
        
        # بررسی ساختار منطقی پاسخ
        if "📋" in generated_text or "مرجع:" in generated_text:
            compliance_score += 0.2
        
        # بررسی وضوح و نظم
        sentences = generated_text.split('.')
        if len(sentences) >= 3:  # حداقل 3 جمله
            compliance_score += 0.2
        
        return min(compliance_score, 1.0)
    
    def calculate_completeness_score(
        self, 
        generated_text: str, 
        expected_elements: List[str]
    ) -> float:
        """کامل بودن پاسخ"""
        
        if not expected_elements:
            return 0.8  # امتیاز پایه
        
        found_elements = 0
        for element in expected_elements:
            if element.lower() in generated_text.lower():
                found_elements += 1
        
        base_score = found_elements / len(expected_elements)
        
        # اضافه کردن امتیاز برای طول مناسب
        length_bonus = min(len(generated_text) / 500, 0.2)  # حداکثر 0.2 امتیاز
        
        return min(base_score + length_bonus, 1.0)
    
    def calculate_legal_accuracy_metrics(
        self,
        generated_text: str,
        expected_citations: List[str],
        source_contexts: List[str],
        expected_elements: List[str]
    ) -> LegalAccuracyMetrics:
        """محاسبه تمام معیارهای دقت حقوقی"""
        
        return LegalAccuracyMetrics(
            citation_accuracy=self.calculate_citation_accuracy(generated_text, expected_citations),
            legal_term_coverage=self.calculate_legal_term_coverage(generated_text),
            factual_consistency=self.calculate_factual_consistency(generated_text, source_contexts),
            law_compliance=self.calculate_law_compliance(generated_text),
            completeness_score=self.calculate_completeness_score(generated_text, expected_elements)
        )
    
    # === Citation Quality Metrics ===
    
    def _extract_citations(self, text: str) -> List[str]:
        """استخراج ارجاعات از متن"""
        
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return citations
    
    def _extract_claims(self, text: str) -> List[str]:
        """استخراج ادعاهای اصلی از متن"""
        
        # تقسیم به جملات
        sentences = re.split(r'[.!?]', text)
        
        # فیلتر جملات مهم
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                any(keyword in sentence for keyword in ["است", "می‌باشد", "شامل", "عبارت"])):
                claims.append(sentence)
        
        return claims
    
    def _is_claim_supported(self, claim: str, context: str) -> bool:
        """بررسی پشتیبانی ادعا توسط منبع"""
        
        claim_words = set(claim.lower().split())
        context_words = set(context.lower().split())
        
        # حداقل 60% تطابق کلمات
        overlap = len(claim_words.intersection(context_words))
        return overlap / len(claim_words) >= 0.6
    
    def calculate_citation_metrics(
        self,
        generated_text: str,
        expected_citations: List[str],
        available_sources: List[str]
    ) -> CitationMetrics:
        """محاسبه معیارهای کیفیت ارجاع"""
        
        found_citations = self._extract_citations(generated_text)
        
        # تعداد ارجاعات
        citation_count = len(found_citations)
        
        # ارجاعات معتبر
        valid_citations = len([c for c in found_citations if any(source in c for source in available_sources)])
        
        # دقت ارجاعات
        citation_precision = valid_citations / citation_count if citation_count > 0 else 0.0
        
        # بازیابی ارجاعات
        expected_found = len([c for c in expected_citations if any(c in f for f in found_citations)])
        citation_recall = expected_found / len(expected_citations) if expected_citations else 0.0
        
        # کیفیت فرمت
        format_score = self._calculate_citation_format_score(found_citations)
        
        # تنوع منابع
        unique_sources = len(set([c.split()[0] for c in found_citations if c.split()]))
        source_diversity = min(unique_sources / 3, 1.0)  # حداکثر 3 منبع متنوع
        
        return CitationMetrics(
            citation_count=citation_count,
            valid_citations=valid_citations,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            citation_format_score=format_score,
            source_diversity=source_diversity
        )
    
    def _calculate_citation_format_score(self, citations: List[str]) -> float:
        """امتیاز کیفیت فرمت ارجاعات"""
        
        if not citations:
            return 0.0
        
        format_scores = []
        for citation in citations:
            score = 0.0
            
            # بررسی وجود شماره ماده
            if re.search(r"ماده\s+\d+", citation):
                score += 0.3
            
            # بررسی وجود نام قانون
            if any(word in citation for word in ["قانون", "آیین‌نامه", "دستورالعمل"]):
                score += 0.3
            
            # بررسی وجود سال مصوبه
            if re.search(r"\d{4}", citation):
                score += 0.2
            
            # بررسی استفاده از کلمات ارجاع
            if any(word in citation for word in ["بر اساس", "مطابق", "طبق"]):
                score += 0.2
            
            format_scores.append(score)
        
        return statistics.mean(format_scores)
    
    # === Overall Metrics ===
    
    def calculate_overall_metrics(
        self,
        generated_text: str,
        reference_text: str,
        retrieved_docs: List[Dict],
        relevant_docs: List[str],
        expected_citations: List[str],
        source_contexts: List[str],
        expected_elements: List[str]
    ) -> OverallMetrics:
        """محاسبه تمام معیارها"""
        
        # محاسبه هر دسته از معیارها
        retrieval_metrics = self.calculate_retrieval_metrics(retrieved_docs, relevant_docs)
        generation_metrics = self.calculate_generation_metrics(generated_text, reference_text)
        legal_accuracy = self.calculate_legal_accuracy_metrics(
            generated_text, expected_citations, source_contexts, expected_elements
        )
        citation_quality = self.calculate_citation_metrics(
            generated_text, expected_citations, [doc.get('source', '') for doc in retrieved_docs]
        )
        
        # محاسبه امتیاز کلی (وزن‌دار)
        overall_score = (
            0.25 * retrieval_metrics.precision_at_5 +  # بازیابی
            0.25 * generation_metrics.rouge_l_f +      # تولید
            0.30 * statistics.mean([                   # دقت حقوقی
                legal_accuracy.citation_accuracy,
                legal_accuracy.legal_term_coverage,
                legal_accuracy.factual_consistency,
                legal_accuracy.law_compliance,
                legal_accuracy.completeness_score
            ]) +
            0.20 * citation_quality.citation_precision  # کیفیت ارجاع
        )
        
        return OverallMetrics(
            overall_score=overall_score,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            legal_accuracy=legal_accuracy,
            citation_quality=citation_quality
        )

# تست
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧮 تست Metrics Calculator")
    print("=" * 50)
    
    calculator = LegalMetricsCalculator()
    
    # داده‌های تست
    generated = """بر اساس ماده 3 قانون مقررات انتظامی هیئت علمی مصوب 1364:
    وظایف اعضای هیئت علمی شامل پژوهش، تدریس و خدمات است.
    📋 مرجع: ماده 3 قانون مقررات انتظامی هیئت علمی"""
    
    reference = """طبق ماده 3 قانون مقررات انتظامی هیئت علمی:
    اعضای هیئت علمی مکلف به انجام پژوهش، تدریس و ارائه خدمات هستند."""
    
    retrieved_docs = [
        {"source": "قانون مقررات انتظامی هیئت علمی", "content": "متن قانون"},
        {"source": "آیین‌نامه ارتقا", "content": "متن آیین‌نامه"}
    ]
    
    relevant_docs = ["قانون مقررات انتظامی"]
    expected_citations = ["ماده 3", "قانون مقررات انتظامی"]
    
    # تست تمام معیارها
    overall = calculator.calculate_overall_metrics(
        generated_text=generated,
        reference_text=reference,
        retrieved_docs=retrieved_docs,
        relevant_docs=relevant_docs,
        expected_citations=expected_citations,
        source_contexts=[reference],
        expected_elements=["پژوهش", "تدریس", "خدمات"]
    )
    
    print(f"📊 نتایج ارزیابی:")
    print(f"   • امتیاز کلی: {overall.overall_score:.3f}")
    print(f"   • Precision@5: {overall.retrieval_metrics.precision_at_5:.3f}")
    print(f"   • ROUGE-L: {overall.generation_metrics.rouge_l_f:.3f}")
    print(f"   • دقت ارجاعات: {overall.legal_accuracy.citation_accuracy:.3f}")
    print(f"   • انطباق حقوقی: {overall.legal_accuracy.law_compliance:.3f}")
    
    print("\n✅ تست کامل شد")