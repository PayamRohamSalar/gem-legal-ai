# src/data_processing/smart_chunker_improved.py

import os
import json
import re
from typing import List, Dict, Tuple
from datetime import datetime
import hashlib

class LegalTextChunker:
    """
    کلاس تقسیم‌بندی هوشمند متون حقوقی - نسخه بهینه شده
    """
    
    def __init__(self, 
                 chunk_size: int = 400, 
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100):
        """
        Args: 
            chunk_size: حداکثر تعداد کلمات در هر chunk
            chunk_overlap: تعداد کلمات overlap بین chunk ها
            min_chunk_size: حداقل تعداد کلمات در هر chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # پترن‌های بهبود یافته برای تشخیص ساختار حقوقی
        self.structure_patterns = {
            'article': [
                re.compile(r'^[\s]*[‌]?ماده\s*[\u06F0-\u06F9\u0660-\u0669\d]+[\.\-\s]', re.UNICODE),
                re.compile(r'^[\s]*ماده\s*[\u06F0-\u06F9\u0660-\u0669\d]+', re.UNICODE),
                re.compile(r'[‌]ماده\s*[\u06F0-\u06F9\u0660-\u0669\d]+', re.UNICODE)
            ],
            'clause': [
                re.compile(r'^[\s]*([الف-ی]|[۱-۹]|\d+)\s*[\.\-\:\)]', re.UNICODE),
                re.compile(r'^[\s]*[الف-ی]\s*[\.\-]', re.UNICODE),
                re.compile(r'^[\s]*\d+\s*[\.\-]', re.UNICODE)
            ],
            'note': [
                re.compile(r'^[\s]*[‌]?تبصره\s*[\u06F0-\u06F9\u0660-\u0669\d]*', re.UNICODE),
                re.compile(r'تبصره\s*[\u06F0-\u06F9\u0660-\u0669\d]*', re.UNICODE)
            ],
            'chapter': [
                re.compile(r'^[\s]*فصل\s*[\u06F0-\u06F9\u0660-\u0669\d]+(.*)', re.UNICODE),
                re.compile(r'فصل\s*(اول|دوم|سوم|چهارم|پنجم|ششم)', re.UNICODE)
            ],
            'section': [
                re.compile(r'^[\s]*بخش\s*[\u06F0-\u06F9\u0660-\u0669\d]+(.*)', re.UNICODE),
                re.compile(r'بخش\s*(اول|دوم|سوم)', re.UNICODE)
            ],
            'law_title': [
                re.compile(r'قانون\s+(.{10,})', re.UNICODE)
            ]
        }
        
        # وزن‌های اولویت برای انواع مختلف boundaries
        self.boundary_weights = {
            'law_title': 12,
            'chapter': 10,
            'section': 9,
            'article': 8,
            'note': 7,
            'clause': 6,
            'paragraph': 5,
            'sentence': 3
        }
    
    def identify_structure_boundaries(self, text: str) -> List[Dict]:
        """
        شناسایی boundaries ساختاری در متن - نسخه بهینه شده
        """
        boundaries = []
        lines = text.split('\n')
        
        print(f"🔍 تحلیل {len(lines)} خط متن...")
        
        current_position = 0
        
        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()
            
            # رد کردن خطوط خالی یا کوتاه
            if not line or len(line) < 3:
                current_position += len(original_line) + 1  # +1 برای \n
                continue
            
            # بررسی انواع مختلف boundaries
            boundary_found = False
            
            for boundary_type, patterns in self.structure_patterns.items():
                if boundary_found:
                    break
                    
                for pattern in patterns:
                    match = pattern.search(line)
                    if match:
                        # استخراج شماره از match
                        number = ""
                        if match.groups():
                            number = match.group(1)
                        else:
                            # سعی در استخراج شماره از خود خط
                            number_match = re.search(r'[\u06F0-\u06F9\u0660-\u0669\d]+', line)
                            if number_match:
                                number = number_match.group()
                        
                        boundaries.append({
                            'type': boundary_type,
                            'position': current_position,
                            'line_number': i,
                            'text': line[:80] + "..." if len(line) > 80 else line,
                            'number': number,
                            'weight': self.boundary_weights.get(boundary_type, 1),
                            'full_line': line,
                            'line_length': len(line)
                        })
                        boundary_found = True
                        #print(f"  📍 {boundary_type}: {line[:60]}...")
                        break
            
            # اگر boundary نبود، پاراگراف‌های طولانی را بررسی کن
            if not boundary_found and len(line) > 150:
                boundaries.append({
                    'type': 'paragraph',
                    'position': current_position,
                    'line_number': i,
                    'text': line[:80] + "..." if len(line) > 80 else line,
                    'number': '',
                    'weight': self.boundary_weights['paragraph'],
                    'full_line': line,
                    'line_length': len(line)
                })
            
            current_position += len(original_line) + 1  # +1 برای \n
        
        print(f"📊 مجموع boundaries یافت شده: {len(boundaries)}")
        
        # اگر boundaries کافی نداریم، تقسیم‌بندی بر اساس پاراگراف انجام دهیم
        if len(boundaries) < 10:
            print("⚠️ boundaries کافی یافت نشد. اضافه کردن paragraph boundaries...")
            boundaries.extend(self._add_paragraph_boundaries(text, lines, len(boundaries)))
        
        return sorted(boundaries, key=lambda x: x['position'])
    
    def _add_paragraph_boundaries(self, text: str, lines: List[str], existing_count: int) -> List[Dict]:
        """
        اضافه کردن boundaries بر اساس پاراگراف‌های متوسط
        """
        paragraph_boundaries = []
        current_position = 0
        
        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()
            
            # شرایط تبدیل به paragraph boundary
            if (len(line) > 50 and  # حداقل طول
                not any(char in line[:20] for char in ['ماده', 'بند', 'تبصره', 'فصل']) and  # نباشد boundary اصلی
                line.count('.') >= 1):  # حداقل یک جمله
                
                paragraph_boundaries.append({
                    'type': 'paragraph',
                    'position': current_position,
                    'line_number': i,
                    'text': line[:80] + "..." if len(line) > 80 else line,
                    'number': str(len(paragraph_boundaries) + 1),
                    'weight': self.boundary_weights['paragraph'],
                    'full_line': line,
                    'line_length': len(line)
                })
            
            current_position += len(original_line) + 1
        
        print(f"📝 اضافه شد {len(paragraph_boundaries)} paragraph boundary")
        return paragraph_boundaries
    
    def create_semantic_chunks(self, text: str, boundaries: List[Dict]) -> List[Dict]:
        """
        ایجاد chunk های معنایی بر اساس boundaries - نسخه بهینه شده
        """
        if not boundaries:
            return self._create_simple_chunks(text)
        
        chunks = []
        current_chunk_lines = []
        current_word_count = 0
        current_metadata = {
            'structures': [],
            'start_position': 0,
            'boundaries_included': []
        }
        
        print(f"🔄 ایجاد chunks از {len(boundaries)} boundary...")
        
        # تقسیم متن به خطوط
        lines = text.split('\n')
        processed_lines = 0
        
        for boundary in boundaries:
            try:
                # خطوط مربوط به این boundary
                boundary_line_num = boundary['line_number']
                
                # اضافه کردن خطوط قبل از این boundary (اگر باقی مانده باشد)
                while processed_lines < boundary_line_num:
                    if processed_lines < len(lines):
                        line = lines[processed_lines].strip()
                        if line:
                            current_chunk_lines.append(line)
                            current_word_count += len(line.split())
                    processed_lines += 1
                
                # اضافه کردن خط boundary
                boundary_line = boundary['full_line']
                current_chunk_lines.append(boundary_line)
                current_word_count += len(boundary_line.split())
                processed_lines += 1
                
                # بررسی اینکه آیا باید chunk جدید شروع کنیم
                should_break = (
                    current_word_count >= self.chunk_size or
                    (current_word_count >= self.min_chunk_size and 
                     boundary['weight'] >= self.boundary_weights['article'])
                )
                
                if should_break and current_chunk_lines:
                    # ایجاد chunk فعلی
                    chunk_text = '\n'.join(current_chunk_lines)
                    chunk_data = self._create_chunk_data(
                        chunk_text, 
                        current_metadata, 
                        len(chunks)
                    )
                    chunks.append(chunk_data)
                    
                    # شروع chunk جدید با overlap
                    overlap_lines = self._create_overlap_lines(current_chunk_lines)
                    current_chunk_lines = overlap_lines
                    current_word_count = sum(len(line.split()) for line in overlap_lines)
                    current_metadata = {
                        'structures': [boundary['type']],
                        'start_position': boundary['position'],
                        'boundaries_included': [boundary]
                    }
                else:
                    # ادامه chunk فعلی
                    current_metadata['structures'].append(boundary['type'])
                    current_metadata['boundaries_included'].append(boundary)
                    
            except Exception as e:
                print(f"⚠️ خطا در پردازش boundary {boundary.get('line_number', '?')}: {str(e)}")
                continue
        
        # اضافه کردن باقی خطوط
        while processed_lines < len(lines):
            line = lines[processed_lines].strip()
            if line:
                current_chunk_lines.append(line)
                current_word_count += len(line.split())
            processed_lines += 1
        
        # اضافه کردن آخرین chunk
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            if len(chunk_text.split()) >= self.min_chunk_size:
                chunk_data = self._create_chunk_data(
                    chunk_text, 
                    current_metadata, 
                    len(chunks)
                )
                chunks.append(chunk_data)
        
        print(f"✅ تولید {len(chunks)} chunk")
        return chunks
    
    def _create_overlap_lines(self, lines: List[str]) -> List[str]:
        """
        ایجاد overlap بین chunks بر اساس خطوط
        """
        if not lines:
            return []
        
        # محاسبه تعداد خطوط overlap بر اساس کلمات
        overlap_words = 0
        overlap_lines = []
        
        for line in reversed(lines):
            line_words = len(line.split())
            if overlap_words + line_words <= self.chunk_overlap:
                overlap_lines.insert(0, line)
                overlap_words += line_words
            else:
                break
        
        return overlap_lines
    
    def _create_simple_chunks(self, text: str) -> List[Dict]:
        """
        ایجاد chunks ساده در صورت عدم شناسایی boundaries کافی
        """
        chunks = []
        lines = text.split('\n')
        
        current_lines = []
        current_word_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_words = len(line.split())
            
            # بررسی اینکه آیا باید chunk جدید شروع کنیم
            if current_word_count + line_words > self.chunk_size and current_lines:
                # ایجاد chunk فعلی
                chunk_text = '\n'.join(current_lines)
                if len(chunk_text.split()) >= self.min_chunk_size:
                    chunk_data = {
                        'chunk_id': self._generate_chunk_id(chunk_text, len(chunks)),
                        'chunk_index': len(chunks),
                        'text': chunk_text,
                        'word_count': len(chunk_text.split()),
                        'char_count': len(chunk_text),
                        'sentence_count': len([s for s in chunk_text.split('.') if s.strip()]),
                        'structures': ['simple_paragraph'],
                        'start_position': 0,
                        'boundaries_info': [],
                        'quality_score': self._calculate_simple_quality_score(chunk_text),
                        'keywords': self._extract_keywords(chunk_text),
                        'legal_entities': self._extract_legal_entities(chunk_text)
                    }
                    chunks.append(chunk_data)
                
                # شروع chunk جدید با overlap
                overlap_lines = current_lines[-2:] if len(current_lines) > 2 else []
                current_lines = overlap_lines + [line]
                current_word_count = sum(len(l.split()) for l in current_lines)
            else:
                current_lines.append(line)
                current_word_count += line_words
        
        # اضافه کردن آخرین chunk
        if current_lines:
            chunk_text = '\n'.join(current_lines)
            if len(chunk_text.split()) >= self.min_chunk_size:
                chunk_data = {
                    'chunk_id': self._generate_chunk_id(chunk_text, len(chunks)),
                    'chunk_index': len(chunks),
                    'text': chunk_text,
                    'word_count': len(chunk_text.split()),
                    'char_count': len(chunk_text),
                    'sentence_count': len([s for s in chunk_text.split('.') if s.strip()]),
                    'structures': ['simple_paragraph'],
                    'start_position': 0,
                    'boundaries_info': [],
                    'quality_score': self._calculate_simple_quality_score(chunk_text),
                    'keywords': self._extract_keywords(chunk_text),
                    'legal_entities': self._extract_legal_entities(chunk_text)
                }
                chunks.append(chunk_data)
        
        return chunks
    
    def _calculate_simple_quality_score(self, text: str) -> float:
        """محاسبه امتیاز کیفیت ساده"""
        score = 5.0  # امتیاز پایه
        
        # امتیاز بر اساس طول
        words = text.split()
        if self.min_chunk_size <= len(words) <= self.chunk_size:
            score += 2.0
        
        # امتیاز بر اساس محتوای حقوقی
        legal_indicators = ['ماده', 'بند', 'تبصره', 'قانون', 'مقرر']
        legal_count = sum(1 for indicator in legal_indicators if indicator in text)
        score += min(legal_count * 0.5, 3.0)
        
        return min(score, 10.0)
    
    def _create_chunk_data(self, text: str, metadata: Dict, chunk_index: int) -> Dict:
        """ایجاد داده‌های کامل برای یک chunk"""
        
        # پاک‌سازی متن chunk
        clean_text = self._clean_chunk_text(text)
        
        # ایجاد ID یکتا
        chunk_id = self._generate_chunk_id(clean_text, chunk_index)
        
        # آمارگیری
        words = clean_text.split()
        sentences = re.split(r'[.!?؟۔]', clean_text)
        
        return {
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'text': clean_text,
            'word_count': len(words),
            'char_count': len(clean_text),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'structures': list(set(metadata.get('structures', []))),
            'start_position': metadata.get('start_position', 0),
            'boundaries_info': metadata.get('boundaries_included', []),
            'quality_score': self._calculate_quality_score(clean_text, metadata),
            'keywords': self._extract_keywords(clean_text),
            'legal_entities': self._extract_legal_entities(clean_text)
        }
    
    def _clean_chunk_text(self, text: str) -> str:
        """پاک‌سازی متن chunk"""
        # حذف خطوط خالی اضافی
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # ترکیب مجدد
        text = '\n'.join(lines)
        
        # حذف فاصله‌های اضافی در هر خط
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _generate_chunk_id(self, text: str, index: int) -> str:
        """تولید ID یکتا برای chunk"""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        return f"chunk_{index:04d}_{text_hash}"
    
    def _calculate_quality_score(self, text: str, metadata: Dict) -> float:
        """محاسبه امتیاز کیفیت chunk"""
        score = 0.0
        
        # امتیاز بر اساس طول
        words = text.split()
        if self.min_chunk_size <= len(words) <= self.chunk_size:
            score += 3.0
        elif len(words) < self.min_chunk_size:
            score += 1.0
        else:
            score += 2.0
        
        # امتیاز بر اساس ساختار
        structures = metadata.get('structures', [])
        structure_bonus = sum(self.boundary_weights.get(s, 0) for s in structures)
        score += min(structure_bonus / 10, 3.0)
        
        # امتیاز بر اساس محتوای حقوقی
        legal_indicators = ['ماده', 'بند', 'تبصره', 'قانون', 'مقرر', 'موضوع']
        legal_count = sum(1 for indicator in legal_indicators if indicator in text)
        score += min(legal_count * 0.5, 2.0)
        
        # امتیاز بر اساس تنوع جملات
        sentences = text.split('.')
        if len(sentences) > 2:
            score += 1.0
        
        return min(score, 10.0)  # حداکثر امتیاز 10
    
    def _extract_keywords(self, text: str) -> List[str]:
        """استخراج کلیدواژه‌های مهم از chunk"""
        legal_keywords = [
            'قانون', 'ماده', 'بند', 'تبصره', 'فصل', 'مقرر', 'موضوع',
            'مجلس', 'هیئت وزیران', 'شورای عالی', 'وزارت', 'مؤسسه',
            'دانشگاه', 'پژوهش', 'فناوری', 'تحقیقات', 'علوم',
            'انتظامی', 'تخلف', 'مجازات', 'تعهد', 'مسئولیت'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in legal_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_legal_entities(self, text: str) -> List[Dict]:
        """استخراج موجودیت‌های حقوقی"""
        entities = []
        
        # تشخیص ارجاعات به مواد
        article_refs = re.findall(r'ماده\s*[\u06F0-\u06F9\u0660-\u0669\d]+', text)
        for ref in article_refs:
            entities.append({
                'type': 'article_reference',
                'value': ref
            })
        
        # تشخیص تاریخ‌ها
        dates = re.findall(r'(\d{1,2}/\d{1,2}/\d{4})', text)
        for date in dates:
            entities.append({
                'type': 'date',
                'value': date
            })
        
        return entities
    
    def chunk_document(self, text: str, document_metadata: Dict) -> List[Dict]:
        """
        تقسیم یک سند کامل به chunk ها - نسخه بهینه شده
        """
        print(f"🔄 شروع chunking سند: {document_metadata.get('title', 'بدون عنوان')}")
        
        # بررسی طول متن
        total_words = len(text.split())
        total_lines = len(text.split('\n'))
        
        print(f"📄 ورودی: {total_words} کلمه، {total_lines} خط")
        
        # شناسایی boundaries
        boundaries = self.identify_structure_boundaries(text)
        print(f"📍 تعداد boundaries یافت شده: {len(boundaries)}")
        
        # ایجاد chunk ها
        chunks = self.create_semantic_chunks(text, boundaries)
        
        # اضافه کردن metadata سند به هر chunk
        for chunk in chunks:
            chunk['document_metadata'] = document_metadata
            chunk['document_title'] = document_metadata.get('title', '')
            chunk['document_type'] = document_metadata.get('document_type', '')
            chunk['authority'] = document_metadata.get('authority', '')
            chunk['approval_date'] = document_metadata.get('approval_date', '')
        
        if chunks:
            avg_chunk_size = sum(c['word_count'] for c in chunks) / len(chunks)
            avg_quality = sum(c['quality_score'] for c in chunks) / len(chunks)
            print(f"✅ تولید {len(chunks)} chunk برای سند")
            print(f"📊 میانگین طول chunk: {avg_chunk_size:.1f} کلمه")
            print(f"⭐ میانگین کیفیت: {avg_quality:.1f}/10")
        else:
            print("❌ هیچ chunk تولید نشد!")
        
        return chunks
    
    def save_chunks(self, chunks: List[Dict], output_dir: str, document_name: str):
        """
        ذخیره chunk ها در فایل‌های جداگانه
        """
        if not chunks:
            print("⚠️ هیچ chunk برای ذخیره وجود ندارد!")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # ذخیره تک تک chunk ها
        chunks_dir = os.path.join(output_dir, f"{document_name}_chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        for chunk in chunks:
            chunk_file = os.path.join(chunks_dir, f"{chunk['chunk_id']}.json")
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
        
        # ذخیره فایل خلاصه
        summary = {
            'document_name': document_name,
            'total_chunks': len(chunks),
            'total_words': sum(c['word_count'] for c in chunks),
            'average_chunk_size': sum(c['word_count'] for c in chunks) / len(chunks),
            'min_chunk_size': min(c['word_count'] for c in chunks),
            'max_chunk_size': max(c['word_count'] for c in chunks),
            'quality_scores': [c['quality_score'] for c in chunks],
            'average_quality': sum(c['quality_score'] for c in chunks) / len(chunks),
            'structures_found': list(set(
                structure for chunk in chunks 
                for structure in chunk['structures']
            )),
            'chunking_date': datetime.now().isoformat(),
            'chunking_parameters': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'min_chunk_size': self.min_chunk_size
            }
        }
        
        summary_file = os.path.join(output_dir, f"{document_name}_chunks_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"💾 chunk ها ذخیره شدند در: {chunks_dir}")
        print(f"📋 خلاصه ذخیره شد در: {summary_file}")


def main():
    """تابع اصلی برای تست chunker"""
    # خواندن یک فایل پردازش شده نمونه
    processed_dir = "data/processed_chunks"
    metadata_dir = "data/metadata"
    
    # پیدا کردن فایل‌های پردازش شده
    text_files = [f for f in os.listdir(processed_dir) if f.endswith('_cleaned.txt')]
    
    if not text_files:
        print("❌ هیچ فایل پردازش شده‌ای یافت نشد!")
        print("ابتدا document_extractor.py را اجرا کنید.")
        return
    
    # انتخاب اولین فایل برای تست
    test_file = text_files[0]
    document_name = test_file.replace('_cleaned.txt', '')
    
    # خواندن متن
    with open(os.path.join(processed_dir, test_file), 'r', encoding='utf-8') as f:
        text = f.read()
    
    # خواندن metadata
    metadata_file = os.path.join(metadata_dir, f"{document_name}_metadata.json")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        doc_metadata = json.load(f)
    
    # ایجاد chunker
    chunker = LegalTextChunker(chunk_size=400, chunk_overlap=50)
    
    # chunking
    chunks = chunker.chunk_document(text, doc_metadata['metadata'])
    
    # ذخیره
    output_dir = "data/chunks"
    chunker.save_chunks(chunks, output_dir, document_name)
    
    print(f"\n🎉 Chunking سند {document_name} با موفقیت انجام شد!")
    print(f"📊 تعداد chunk های تولید شده: {len(chunks)}")


if __name__ == "__main__":
    main()