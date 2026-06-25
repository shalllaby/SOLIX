import os
import re
import difflib
import sqlite3
import datetime
import requests
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(title="SOLIX Virtual Data Engineer Chatbot")

# Create templates directory if not exists
templates = Jinja2Templates(directory="templates")

# Groq API Details
API_KEY = "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"
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Domain-specific terms for local correction fallback
DOMAIN_VOCABULARY = [
    # SOLIX & General Platform terms
    "solix", "سوليكس", "سولكس", "سند", "sanad", "منصة", "فريق", "أعضاء", "الاعضاء", "مشرف", "المشرفين", "المشرفون", "مشرفين",
    # Team members Arabic
    "محمد", "شلبي", "محسن", "حسن", "العسال", "عبدالرحمن", "عبد الرحمن", "غريب", "ميار", "ممدوح", "منة", "منه", "محمود", "حبيبة", "حبيبه",
    "مختار", "عمرو", "عمر", "فاطمة", "فاطمه", "حسين", "ملك", "السيد", "صفاء", "عاصم", "حاتم", "خالد", "أشرقت", "اشرقت",
    # Team members English
    "shalaby", "mohsen", "hassan", "elassal", "abdelrhman", "gharieb", "mayar", "mamdouh", "menna", "habiba", "mokhtar", "amr", "omar", 
    "fatma", "hossein", "malak", "elsayd", "safaa", "assem", "hatem", "khaled", "ashrakat", "sol", "team",
    # Supervisors & Institution
    "سيمون", "عزت", "نجلاء", "سعيد", "جامعة", "جامعه", "حلوان", "الكلية", "الكليه", "تكنولوجيا", "التكنولوجية", "القاهرة", "القاهره",
    # Technical & features terms
    "التنظيف", "الذكي", "تعبئة", "تعبئه", "الفراغات", "التعلم", "الآلي", "الالي", "المساعد", "الصوتي", "مستشار", "البيانات", "الاصطناعية", 
    "الاصطناعيه", "الاستمارات", "مستخرج", "الصور", "الفوضى", "تخريب", "الأتمتة", "الاتمتة", "الباكيند", "الفرونت", "الواجهات", "الخلفية", "الامامية"
]

CORRECTOR_SYSTEM_PROMPT = (
    "You are an AI assistant specialized in correcting spelling errors, autocompleting incomplete words, "
    "and resolving distorted text in user queries in both Arabic and English. Your goal is to rewrite the query "
    "into a clean, grammatically correct search query in the same language. Do not answer the query under any circumstance.\n\n"
    "Context rules and domain terms:\n"
    "- Platform: SOLIX (سوليكس)\n"
    "- Team members (16 members): محمد شلبي (Mohamed Shalaby), محسن حسن (Mohsen Hassan), محمد العسال (Mohamed Elassal), "
    "عبد الرحمن غريب (Abdelrhman Gharieb), ميار ممدوح (Mayar Mamdouh), منة محمود (Menna Mahmoud), حبيبة محمد (Habiba Mohamed), "
    "محمد مختار (Mohamed Mokhtar), عمرو عمر (Amr Omar), فاطمة حسين (Fatma Hossein), فاطمة محمود (Fatma Mahmoud), "
    "ملك السيد (Malak Elsayd), صفاء محمد (Safaa Mohamed), عاصم حاتم (Assem Hatem), محمد خالد (Mohamed Khaled), أشرقت خالد (Ashrakat Khaild).\n"
    "- Supervisors: د. سيمون عزت, م. نجلاء سعيد.\n"
    "- Institution: جامعة حلوان التكنولوجية الدولية, الكلية التكنولوجية بالقاهرة.\n"
    "- Hackathons: هاكاثون ناسا (NASA Space Apps), Cairo ICT, EVA AI, GDG Delta, Data Pill (مشروع سند SANAD).\n"
    "- Key Features: استوديو التنظيف الذكي (Alpha, Beta, Gamma), محرك تعبئة الفراغات (AI Imputer), المساعد الصوتي سول (SOL Voice Copilot), "
    "استوديو التعلم الآلي (AutoML Studio), مستشار البيانات (AI Dataset Advisor), استوديو البيانات الاصطناعية (Synthetic Data Studio), "
    "منشئ الاستمارات الذكي ومستخرج الصور (SOL Forms & OCR), محرك الفوضى وتخريب البيانات (Chaos Engine & Data Sabotage).\n\n"
    "Instructions:\n"
    "1. Fix spelling mistakes, typos, missing characters, or bad translations in the query.\n"
    "2. Complete incomplete words or expressions (e.g., 'مين سيمو' -> 'من هي دكتورة سيمون عزت', 'مين شلب' -> 'من هو محمد شلبي').\n"
    "3. Keep the original intent and format of the query. STRICTLY DO NOT answer the question or replace question words (like 'من', 'مين', 'ما', 'كيف') with their factual answers (e.g., do NOT rewrite 'من هم المشرفون' as 'د. سيمون عزت, م. نجلاء سعيد'). Only correct spelling and complete partial words.\n"
    "4. Output ONLY the corrected/completed query text, with NO quotes, explanations, markdown, or greetings."
)

def normalize_arabic(text):
    if not text:
        return ""
    text = text.lower()
    # Remove diacritics (Tashkeel)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    # Normalize Alef
    text = re.sub(r"[أإآ]", "ا", text)
    # Normalize Yeh/Alef Maksura
    text = re.sub(r"ى", "ي", text)
    # Normalize Teh Marbuta
    text = re.sub(r"ة", "ه", text)
    # Strip spaces
    return re.sub(r"\s+", " ", text).strip()

def correct_query_locally(query: str) -> str:
    normalized_query = normalize_arabic(query)
    words = normalized_query.split()
    
    excluded_words = {
        "من", "مين", "ما", "كيف", "اين", "هل", "ماذا", "في", "على", "عن", "هو", "هي", "هم", "ان", "او", "لا", "نعم", "مع", "الي", "إلى", "هذا", "هذه"
    }
    
    normalized_vocab = [normalize_arabic(v) for v in DOMAIN_VOCABULARY]
    vocab_map = {normalize_arabic(v): v for v in DOMAIN_VOCABULARY}
    
    corrected_words = []
    for word in words:
        if word in excluded_words or len(word) < 3:
            corrected_words.append(word)
            continue
            
        if word in normalized_vocab:
            corrected_words.append(vocab_map[word])
            continue
            
        matches = difflib.get_close_matches(word, normalized_vocab, n=1, cutoff=0.7)
        if matches:
            corrected_words.append(vocab_map[matches[0]])
        else:
            corrected_words.append(word)
            
    return " ".join(corrected_words)

def correct_query_via_llm(query: str) -> str:
    messages = [
        {"role": "system", "content": CORRECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}"}
    ]
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 100
    }
    try:
        response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=2.5)
        if response.status_code == 200:
            result = response.json()
            corrected = result["choices"][0]["message"]["content"].strip()
            # Clean quotes if model wrapped it in quotes
            corrected = re.sub(r'^["\'\u201c\u201d\u2018\u2019]|["\'\u201c\u201d\u2018\u2019]$', '', corrected).strip()
            if corrected:
                return corrected
    except Exception as e:
        print(f"LLM query corrector failed/timed out: {e}")
    return query

def correct_query(query: str) -> str:
    if not query:
        return ""
    normalized = normalize_arabic(query)
    chitchat_keywords = {
        "مرحبا", "مرحب", "اهلا", "سلام", "السلام عليكم", "صباح الخير", "مساء الخير", 
        "هاي", "hi", "hello", "شكرا", "شكرًا", "شكرا لك", "تسلم", "يعطيك العافية", "thanks",
        "اهلين", "يا هلا", "كيفك", "شلونك", "منور", "شكرا جزيلا"
    }
    
    if normalized in chitchat_keywords or (len(normalized.split()) <= 1 and normalized in chitchat_keywords):
        return query

    corrected = correct_query_via_llm(query)
    if corrected and corrected != query:
        return corrected
        
    return correct_query_locally(query)

# 1. Team members metadata dictionary for the special cases
TEAM_MEMBERS = [
    {"name_ar": "محمد شلبي", "name_en": "Mohamed Shalaby", "role": "قائد الفريق (Team Lead)، رئيس قسم الذكاء الاصطناعي (Head of AI)، ورئيس قسم الأتمتة (Head of AUTOMATION). يشارك بفعالية في تطوير الواجهات الخلفية (Backend)، الواجهات الأمامية (Frontend)، تصميم واجهة وتجربة المستخدم (UI/UX)، وكتابة التوثيق (Documentation)."},
    {"name_ar": "محسن حسن", "name_en": "Mohsen Hassan", "role": "مطور ذكاء اصطناعي (AI)، مطور واجهات خلفية (Backend) وأمامية (Frontend)، ومسؤول عن التوثيق (Documentation)."},
    {"name_ar": "محمد العسال", "name_en": "Mohamed Elassal", "role": "مطور ذكاء اصطناعي (AI)."},
    {"name_ar": "عبد الرحمن غريب", "name_en": "Abdelrhman Gharieb", "role": "مطور ذكاء اصطناعي (AI)."},
    {"name_ar": "ميار ممدوح", "name_en": "Mayar Mamdouh", "role": "مطورة ذكاء اصطناعي (AI)، مطورة أتمتة (AUTOMATION)، وتشارك في مهام الواجهات الخلفية المساعدة (Small BACKEND)."},
    {"name_ar": "منة محمود", "name_en": "Menna Mahmoud", "role": "مطورة ذكاء اصطناعي (AI)، مطورة أتمتة (AUTOMATION)، وتشارك في مهام الواجهات الخلفية المساعدة (Small BACKEND)."},
    {"name_ar": "حبيبة محمد", "name_en": "Habiba Mohamed", "role": "مطورة ذكاء اصطناعي (AI)، مطورة أتمتة (AUTOMATION)، وتشارك في مهام الواجهات الخلفية المساعدة (Small BACKEND)."},
    {"name_ar": "محمد مختار", "name_en": "Mohamed Mokhtar", "role": "مطور واجهات خلفية (Backend) ومطور ذكاء اصطناعي (AI)."},
    {"name_ar": "عمرو عمر", "name_en": "Amr Omar", "role": "رئيس قسم الواجهات الأمامية (Head Of Frontend)."},
    {"name_ar": "فاطمة حسين", "name_en": "Fatma Hossein", "role": "مطورة واجهات أمامية (Frontend)."},
    {"name_ar": "فاطمة محمود", "name_en": "Fatma Mahmoud", "role": "مطورة واجهات أمامية (Frontend) وتشارك في مهام الواجهات الخلفية المساعدة (Small BACKEND)."},
    {"name_ar": "ملك السيد", "name_en": "Malak Elsayd", "role": "مطورة واجهات أمامية (Frontend)."},
    {"name_ar": "صفاء محمد", "name_en": "Safaa Mohamed", "role": "رئيسة قسم تصميم واجهات وتجربة المستخدم (Head Of UI/UX). وتتولى أيضاً مهام تطوير الواجهات الأمامية (Frontend)، الأتمتة (AUTOMATION)، والمشاركة في الواجهات الخلفية المساعدة (Small BACKEND)."},
    {"name_ar": "عاصم حاتم", "name_en": "Assem Hatem", "role": "مصمم واجهات وتجربة المستخدم (UI/UX) ومطور واجهات خلفية (Backend)."},
    {"name_ar": "محمد خالد", "name_en": "Mohamed Khaled", "role": "رئيس قسم التوثيق (Head Of Documentation)."},
    {"name_ar": "أشرقت خالد", "name_en": "Ashrakat Khaild", "role": "رئيسة قسم التوثيق (Head Of Documentation) ومطورة واجهات أمامية (Frontend)."}
]

TEAM_ALL_RESPONSE = """تم تطوير منصة **SOLIX** بواسطة فريق عمل متكامل يضم **16 عضواً** مبدعاً (فريق SOL) موزعين على التخصصات التالية:

*   **👑 القيادة والإدارة (Leadership):**
    *   **محمد شلبي (Mohamed Shalaby):** قائد الفريق، رئيس قسم الذكاء الاصطناعي والأتمتة، ومطور واجهات خلفية وأمامية وتصميم UI/UX وتوثيق.
*   **🧠 الذكاء الاصطناعي والأتمتة (AI & Automation):**
    *   **محسن حسن (Mohsen Hassan):** مطور ذكاء اصطناعي، مطور واجهات خلفية وأمامية ومسؤول التوثيق.
    *   **محمد العسال (Mohamed Elassal):** مطور ذكاء اصطناعي.
    *   **عبد الرحمن غريب (Abdelrhman Gharieb):** مطور ذكاء اصطناعي.
    *   **ميار ممدوح (Mayar Mamdouh):** مطورة ذكاء اصطناعي وأتمتة وواجهات خلفية مساعدة.
    *   **منة محمود (Menna Mahmoud):** مطورة ذكاء اصطناعي وأتمتة وواجهات خلفية مساعدة.
    *   **حبيبة محمد (Habiba Mohamed):** مطورة ذكاء اصطناعي وأتمتة وواجهات خلفية مساعدة.
*   **💻 الواجهات الخلفية (Backend):**
    *   **محمد مختار (Mohamed Mokhtar):** مطور واجهات خلفية ومطور ذكاء اصطناعي.
*   **🎨 الواجهات الأمامية (Frontend):**
    *   **عمرو عمر (Amr Omar):** رئيس قسم الواجهات الأمامية.
    *   **فاطمة حسين (Fatma Hossein):** مطورة واجهات أمامية.
    *   **فاطمة محمود (Fatma Mahmoud):** مطورة واجهات أمامية وواجهات خلفية مساعدة.
    *   **ملك السيد (Malak Elsayd):** مطورة واجهات أمامية.
*   **✨ التصميم و UI/UX:**
    *   **صفاء محمد (Safaa Mohamed):** رئيسة قسم تصميم واجهات وتجربة المستخدم، مطورة واجهات أمامية وأتمتة وواجهات خلفية مساعدة.
    *   **عاصم حاتم (Assem Hatem):** مصمم واجهات وتجربة المستخدم ومطور واجهات خلفية.
*   **📝 التوثيق (Documentation):**
    *   **محمد خالد (Mohamed Khaled):** رئيس قسم التوثيق.
    *   **أشرقت خالد (Ashrakat Khaild):** رئيسة قسم التوثيق ومطورة واجهات أمامية.

هل ترغب في الاستفسار عن تفاصيل دور أو إنجازات عضو معين بالفريق؟"""

SUPERVISORS_AND_UNI_RESPONSE = """المشرفون الأكاديميون والمؤسسة التعليمية لمشروع SOLIX وفريق عمل SOL TEAM:

*   **🎓 المشرفون الأكاديميون:**
    *   **د. سيمون عزت:** خبير أكاديمي في تقنيات الذكاء الاصطناعي وتطوير البرمجيات، له مساهمات بحثية رائدة في مجال التعلم الآلي.
    *   **م. نجلاء سعيد:** مهندسة برمجيات متخصصة في تصميم وتطوير تطبيقات الذكاء الاصطناعي، مع خبرة واسعة في إدارة مشاريع تقنية معقدة.

*   **🏫 الجهة التعليمية:**
    *   **الجامعة:** جامعة حلوان التكنولوجية الدولية.
    *   **الكلية:** الكلية التكنولوجية بالقاهرة."""

def check_supervisor_uni_query(query):
    query_clean = query.lower()
    keywords = [
        "مشرف", "المشرفين", "المشرفون", "دكتور", "سيمون", "عزت", 
        "مهندسة", "نجلاء", "سعيد", "جامعة", "حلوان", "الكلية", "تكنولوجيا", "القاهرة"
    ]
    if any(k in query_clean for k in keywords):
        return True
    return False

HACKATHONS_RESPONSE = """سجل إنجازات وجوائز فريق SOL TEAM في الهاكاثونات والمسابقات التقنية:

1. 🥇 **هاكاثون NASA Space Apps Challenge (أكتوبر 2025):**
   * **المركز:** المركز الأول (TOP 1) على مستوى محافظة شبين الكوم.
   * **الإنجاز:** الترشح لتمثيل مصر عالمياً (Global Nominee) من بين مئات المشاريع المشاركة.

2. 🏅 **هاكاثون Cairo ICT Intersection (نوفمبر 2025):**
   * **المركز:** المركز الرابع (TOP 4) على مستوى جمهورية مصر العربية.

3. 🏅 **هاكاثون EVA AI Hackathon (فبراير 2026):**
   * **المركز:** المركز الرابع (TOP 4).
   * **الجائزة:** الحصول على دعم مالي قدره 40,000 جنيه مصري لمواصلة الابتكار.

4. 🏅 **هاكاثون GDG Delta (فبراير 2026):**
   * **المركز:** المركز الخامس (TOP 5) على مستوى الجمهورية بمشاركة ما يزيد عن 610 متنافس.

5. 🥉 **هاكاثون Data Pill (أبريل 2026):**
   * **المركز:** المركز الثالث (TOP 3) بمشروع **SANAD (سند)** وجائزة مالية بقيمة 10,000 جنيه مصري.

يعكس هذا السجل الحافل شغف فريق SOL بالابتكار وقدرته على المنافسة محلياً ودولياً في مجالات الذكاء الاصطناعي وهندسة البيانات."""

def check_hackathons_query(query):
    query_clean = query.lower()
    keywords = [
        "هاكاثون", "هاكاثونات", "مسابقة", "مسابقات", "إنجازات", "إنجاز", "جوائز", "جائزة", "جوايز", "انجازات",
        "المركز", "مركز", "مراكز", "فوز", "فاز", "كسب", "ترتيب", "مركزهم", "مكافأة", "مكافآت",
        "hackathon", "hackathons", "achievement", "achievements", "award", "awards", "prize", "prizes", "ranking", 
        "nasa", "space apps", "intersection", "eva", "gdg", "data pill", "سند", "sanad"
    ]
    return any(k in query_clean for k in keywords)


def check_team_query(query):
    query_clean = query.lower()
    
    # 1. First check if user is asking about a specific member
    found_member = None
    for member in TEAM_MEMBERS:
        if member["name_ar"] in query_clean or member["name_en"].lower() in query_clean:
            found_member = member
            break
            
    if found_member:
        return {"type": "individual", "member": found_member}
        
    # 2. Check if asking about the whole team
    team_keywords = [
        "فريق", "أعضاء", "الاعضاء", "مين في", "فريق العمل", 
        "كل الفريق", "أعضاء الفريق", "roles", "members", "team", "sol"
    ]
    if any(k in query_clean for k in team_keywords):
        return {"type": "team_all"}
        
    return {"type": "general"}


# 2. Document parser & RAG Retriever class
class SOLIXRetriever:
    def __init__(self, doc_path="SOLIX_DOCUMENTATION.md"):
        self.doc_path = doc_path
        self.chunks = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.load_and_index()
        
    def load_and_index(self):
        if not os.path.exists(self.doc_path):
            print(f"Error: {self.doc_path} not found!")
            return
            
        with open(self.doc_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Split by ## or ### markdown headers to make neat self-contained blocks
        sections = content.split("\n")
        current_chunk = []
        current_header = "مقدمة عامة عن SOLIX"
        
        for line in sections:
            if line.startswith("## ") or line.startswith("### "):
                if current_chunk:
                    chunk_text = "\n".join(current_chunk).strip()
                    if chunk_text:
                        self.chunks.append(f"القسم: {current_header}\n\n{chunk_text}")
                current_header = line.replace("#", "").replace("📌", "").replace("👥", "").replace("🌐", "").replace("🛠️", "").replace("🏆", "").replace("⚡", "").replace("📖", "").strip()
                current_chunk = [line]
            else:
                current_chunk.append(line)
                
        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                self.chunks.append(f"القسم: {current_header}\n\n{chunk_text}")
                
        if self.chunks:
            # Fit TF-IDF on our parsed text chunks
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
            print(f"RAG Pipeline: Indexed {len(self.chunks)} document chunks.")
        else:
            print("RAG Pipeline Error: No chunks found to index.")
            
    def retrieve(self, query, top_k=3):
        if not self.chunks or self.tfidf_matrix is None:
            return []
            
        # Compute TF-IDF cosine similarity scores
        query_vec = self.vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        
        # Calculate clean word overlap similarity
        query_words = set(query.lower().split())
        overlap_scores = []
        for chunk in self.chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            overlap_scores.append(overlap / (len(query_words) + 1))
            
        # Combine TF-IDF with keyword overlap (gives better score for matching specific terms)
        combined_scores = tfidf_scores + 0.4 * np.array(overlap_scores)
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            # Only include chunks that have a minimum threshold of relevance
            if combined_scores[idx] > 0.05:
                results.append(self.chunks[idx])
        return results


# 3. SQLite-based Semantic Cache implementation
class SemanticCache:
    def __init__(self, db_path="cache.db"):
        self.db_path = db_path
        self.init_db()
        self.vectorizer = TfidfVectorizer()
        self.cached_queries = []
        self.load_cache_keys()
        
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_cache (
                query TEXT PRIMARY KEY,
                response TEXT,
                created_at TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        
    def load_cache_keys(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT query FROM chat_cache")
        self.cached_queries = [row[0] for row in cursor.fetchall()]
        conn.close()
        
    def get(self, query):
        if not self.cached_queries:
            return None
            
        query_clean = query.strip().lower()
        
        # Exact match check first
        if query_clean in self.cached_queries:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT response FROM chat_cache WHERE query = ?", (query_clean,))
            row = cursor.fetchone()
            conn.close()
            if row:
                print(f"Cache Exact Hit for: '{query}'")
                return row[0]
                
        # Semantic similarity match using local TF-IDF on cached questions
        try:
            all_queries = self.cached_queries + [query_clean]
            matrix = self.vectorizer.fit_transform(all_queries)
            query_vec = matrix[-1]
            cached_matrix = matrix[:-1]
            
            similarities = (cached_matrix * query_vec.T).toarray().flatten()
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            # 0.93 similarity threshold means question is practically identical in intent
            if best_score > 0.93:
                best_query = self.cached_queries[best_idx]
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT response FROM chat_cache WHERE query = ?", (best_query,))
                row = cursor.fetchone()
                conn.close()
                if row:
                    print(f"Cache Semantic Hit for: '{query}' matching key: '{best_query}' (Score: {best_score:.2f})")
                    return row[0]
        except Exception as e:
            print("Semantic Cache Check error:", e)
            
        return None
        
    def set(self, query, response):
        query_clean = query.strip().lower()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO chat_cache (query, response, created_at) VALUES (?, ?, ?)",
                (query_clean, response, datetime.datetime.now())
            )
            conn.commit()
            if query_clean not in self.cached_queries:
                self.cached_queries.append(query_clean)
        except Exception as e:
            print("Cache set error:", e)
        finally:
            conn.close()


# Initialize RAG retriever and Semantic Cache
retriever = SOLIXRetriever()
cache = SemanticCache()

class MessageItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[MessageItem] = []

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")
        
    # Preprocess and correct query (typo/spelling corrections and autocompleting incomplete inputs)
    corrected_query = correct_query(query)
    print(f"Chat Endpoint: Original Query: '{query}' -> Corrected: '{corrected_query}'")
    
    # Step 1: Query local Cache first to save tokens
    cached_response = cache.get(corrected_query)
    if cached_response:
        return JSONResponse(content={"response": cached_response, "cached": True})
        
    # Step 2: Route request based on content check (Supervisors vs Team vs Hackathons vs General)
    is_supervisor_uni = check_supervisor_uni_query(corrected_query)
    team_info = check_team_query(corrected_query)
    is_hackathons = check_hackathons_query(corrected_query)
    
    context = ""
    system_instruction = (
        "أنت المساعد الذكي SOL الخاص بفريق SOL TEAM لمنصة SOLIX (مهندس البيانات الافتراضي الشامل).\n"
        "مهمتك هي الإجابة عن أسئلة المستخدم والتفاعل معه بذكاء ومرونة باللغة العربية الفصحى.\n"
        "قواعد هامة للإجابة:\n"
        "1. أسلوب المحادثة: يجب أن تجيب باللغة العربية الفصحى المبسطة والواضحة، مع الحفاظ على اختصار الردود قدر الإمكان وتجنب الحشو.\n"
        "2. الترحيب والدردشة العامة (Chitchat): إذا كانت رسالة المستخدم ترحيباً، وداعاً، شكراً، أو دردشة عامة، تفاعل معها بلطف ونبرة ودية طبيعية دون التقيد بأي سياق.\n"
        "3. الذكاء الوجداني (تحليل المشاعر): انتبه لنبرة المستخدم العاطفية:\n"
        "   - إذا كان غاضباً أو محبطاً أو يواجه مشكلة، أظهر التعاطف التام وقدم المساعدة بنبرة مهدئة.\n"
        "   - إذا كان سعيداً أو شاكراً، عبّر عن تقديرك وسعادتك لخدمته.\n"
        "4. الأسئلة العامة وعلم البيانات: إذا سأل المستخدم أسئلة برمجية أو علمية عامة (مثال: مفاهيم التعلم الآلي، أو كيفية عمل مكتبة Python)، أجب بمرونة ومعرفة كاملة ولا تقل 'هذه المعلومة غير متوفرة'.\n"
        "5. أسئلة منصة SOLIX وفريقها ومشرفيها وجامعتها: إذا كان السؤال خاصاً بجوانب منصة SOLIX أو فريق عملها SOL TEAM أو المشرفين الأكاديميين (د. سيمون عزت، م. نجلاء سعيد) أو الجامعة (جامعة حلوان التكنولوجية الدولية والكلية التكنولوجية بالقاهرة)، اعتمد **فقط** على السياق المرفق للإجابة بدقة متناهية. وإذا كان السؤال خاصاً بـ SOLIX ولم تجد إجابته في السياق المرفق، قل بأدب: 'عذراً، هذه المعلومة غير متوفرة في وثائق منصة SOLIX المتاحة لدي.'\n"
    )
    
    if is_supervisor_uni:
        context = SUPERVISORS_AND_UNI_RESPONSE
        system_instruction += (
            "\nملاحظة هامة: السؤال يتعلق بالمشرفين الأكاديميين (د. سيمون عزت، م. نجلاء سعيد) أو المؤسسة التعليمية (جامعة حلوان التكنولوجية الدولية والكلية التكنولوجية بالقاهرة). "
            "أجب بدقة متناهية وبأسلوب رسمي بناءً على السياق المرفق فقط."
        )
    elif is_hackathons:
        context = HACKATHONS_RESPONSE
        system_instruction += (
            "\nملاحظة هامة: السؤال يتعلق بإنجازات وجوائز فريق SOL TEAM في الهاكاثونات والمسابقات التقنية. "
            "أجب بدقة متناهية وبأسلوب فخور، حماسي، ومرتب مع ذكر التواريخ، المراكز، والجوائز بالتفصيل بناءً على السياق المرفق فقط."
        )
    elif team_info["type"] == "team_all":
        context = TEAM_ALL_RESPONSE
        system_instruction += (
            "\nملاحظة هامة: السؤال يتعلق بفريق العمل ككل. "
            "يجب أن تسرد كامل الـ 16 عضواً مع أدوارهم باختصار شديد وبصياغة مرتبة ومحترفة تناسب طبيعة السؤال."
        )
    elif team_info["type"] == "individual":
        member = team_info["member"]
        context = f"الاسم: {member['name_ar']} ({member['name_en']})\nالدور والمسؤوليات التفصيلية: {member['role']}"
        system_instruction += (
            f"\nملاحظة هامة: السؤال يتعلق بعضو معين وهو {member['name_ar']}. "
            "اكتب ردًا احترافيًا ومفصلًا يسلط الضوء على دوره، وتخصصه، ومساهماته في منصة SOLIX بناءً على السياق فقط."
        )
    else:
        # Retrieve relative context from SOLIX documentation using TF-IDF RAG
        relevant_chunks = retriever.retrieve(corrected_query, top_k=3)
        if relevant_chunks:
            context = "\n\n---\n\n".join(relevant_chunks)
        # If no relevant chunks are found, context remains empty, allowing general knowledge fallback
        
    # Step 3: Prepare messages payload including conversation history
    messages = [{"role": "system", "content": system_instruction}]
    
    # Append recent chat history (limit to last 6 messages to keep it optimized)
    for msg in req.history[-6:]:
        # Map frontend role "bot" to API role "assistant"
        api_role = "assistant" if msg.role == "bot" else msg.role
        messages.append({"role": api_role, "content": msg.content})
        
    # Append current query
    user_content = f"السؤال: {corrected_query}"
    if context:
        user_content = f"السياق المتاح:\n{context}\n\n" + user_content
        
    messages.append({"role": "user", "content": user_content})
    
    # Step 4: Query Groq API
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,  # Slightly higher temperature for better conversational flexibility
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            result = response.json()
            bot_response = result["choices"][0]["message"]["content"].strip()
            
            # Save response to cache (only cache direct answers to prevent history bias in cache keys)
            if not req.history:
                cache.set(corrected_query, bot_response)
                
            return JSONResponse(content={"response": bot_response, "cached": False})
        else:
            print("Groq API error status:", response.status_code, response.text)
            return JSONResponse(
                status_code=500,
                content={"response": "عذراً، حدث خطأ أثناء الاتصال بخادم المعالجة. يرجى المحاولة لاحقاً."}
            )
    except Exception as e:
        print("Network Request Error:", e)
        return JSONResponse(
            status_code=500,
            content={"response": "عذراً، حدث خطأ في الاتصال بالشبكة. يرجى التحقق من اتصالك بالإنترنت."}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)

