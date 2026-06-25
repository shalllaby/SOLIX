import os
import re
import sys
import difflib
import sqlite3
import datetime
import requests
import numpy as np
from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup paths relative to the run directory
_router_dir = os.path.dirname(os.path.abspath(__file__)) # .../backend/tools/chatbot
_backend_dir = os.path.dirname(os.path.dirname(_router_dir)) # .../backend
_run_dir = os.path.dirname(_backend_dir) # .../run

sys.path.insert(0, _run_dir)

# Template helper matching main app's locale processing
def add_locale_context(request: Request):
    locale = request.cookies.get("sol_locale", "en")
    if locale not in ("en", "ar"):
        locale = "en"
    return {
        "locale": locale,
        "locale_dir": "rtl" if locale == "ar" else "ltr",
        "locale_lang": locale
    }

templates = Jinja2Templates(
    directory=os.path.join(_run_dir, "frontend", "templates"),
    context_processors=[add_locale_context]
)

# Authentication dependency import
try:
    from backend.auth import get_current_user
    from backend.models import User, ChatSession, ChatMessage
    from backend.database import get_db
    from sqlalchemy.orm import Session
except ImportError:
    # Fallback to a mock dependency if there's any import problem
    def get_current_user():
        return None
    def get_db():
        pass

from backend.middleware.barrier import CredentialsBarrier

router = APIRouter(tags=["Chatbot"])

# Groq API Details
def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY")
    if not key or key == "your_api_key_here":
        return ""
    return key

def get_groq_model():
    return os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# Domain-specific terms for local correction fallback
DOMAIN_VOCABULARY = [
    "solix", "سوليكس", "سولكس", "سند", "sanad", "منصة", "فريق", "أعضاء", "الاعضاء", "مشرف", "المشرفين", "المشرفون", "مشرفين",
    "محمد", "شلبي", "محسن", "حسن", "العسال", "عبدالرحمن", "عبد الرحمن", "غريب", "ميار", "ممدوح", "منة", "منه", "محمود", "حبيبة", "حبيبه",
    "مختار", "عمرو", "عمر", "فاطمة", "فاطمه", "حسين", "ملك", "السيد", "صفاء", "عاصم", "حاتم", "خالد", "أشرقت", "اشرقت",
    "shalaby", "mohsen", "hassan", "elassal", "abdelrhman", "gharieb", "mayar", "mamdouh", "menna", "habiba", "mokhtar", "amr", "omar", 
    "fatma", "hossein", "malak", "elsayd", "safaa", "assem", "hatem", "khaled", "ashrakat", "sol", "team",
    "سيمون", "عزت", "نجلاء", "سعيد", "جامعة", "جامعه", "حلوان", "الكلية", "الكليه", "تكنولوجيا", "التكنولوجية", "القاهرة", "القاهره",
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
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
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
        "Authorization": f"Bearer {get_groq_api_key()}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": get_groq_model(),
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 100
    }
    try:
        response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=2.5)
        if response.status_code == 200:
            result = response.json()
            try:
                from backend.utils.llm_logger import log_groq_response
                log_groq_response(result, module_name="chat")
            except Exception as e_log:
                print(f"Failed to log chatbot query corrector token usage: {e_log}")
            corrected = result["choices"][0]["message"]["content"].strip()
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
    return any(k in query_clean for k in keywords)

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
    
    found_member = None
    for member in TEAM_MEMBERS:
        if member["name_ar"] in query_clean or member["name_en"].lower() in query_clean:
            found_member = member
            break
            
    if found_member:
        return {"type": "individual", "member": found_member}
        
    team_keywords = [
        "فريق", "أعضاء", "الاعضاء", "مين في", "فريق العمل", 
        "كل الفريق", "أعضاء الفريق", "roles", "members", "team", "sol"
    ]
    if any(k in query_clean for k in team_keywords):
        return {"type": "team_all"}
        
    return {"type": "general"}

class SOLIXRetriever:
    def __init__(self, doc_path=None):
        if doc_path is None:
            doc_path = os.path.join(_run_dir, "SOLIX_DOCUMENTATION.md")
        self.doc_path = doc_path
        self.chunks = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.load_and_index()
        
    def load_and_index(self):
        if not os.path.exists(self.doc_path):
            print(f"Error: {self.doc_path} not found!")
            return
            
        with open(self.doc_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
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
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
            print(f"RAG Pipeline: Indexed {len(self.chunks)} document chunks.")
        else:
            print("RAG Pipeline Error: No chunks found to index.")
            
    def retrieve(self, query, top_k=3):
        if not self.chunks or self.tfidf_matrix is None:
            return []
            
        query_vec = self.vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        
        query_words = set(query.lower().split())
        overlap_scores = []
        for chunk in self.chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            overlap_scores.append(overlap / (len(query_words) + 1))
            
        combined_scores = tfidf_scores + 0.4 * np.array(overlap_scores)
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0.05:
                results.append(self.chunks[idx])
        return results

class SemanticCache:
    def __init__(self, db_path=None):
        if db_path is None:
            os.makedirs(os.path.join(_backend_dir, "data"), exist_ok=True)
            db_path = os.path.join(_backend_dir, "data", "chatbot_cache.db")
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
        
    def get(self, query, user_id: int):
        if not self.cached_queries:
            return None
            
        query_clean = f"user_{user_id}:{query.strip().lower()}"
        
        if query_clean in self.cached_queries:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT response FROM chat_cache WHERE query = ?", (query_clean,))
            row = cursor.fetchone()
            conn.close()
            if row:
                print(f"Cache Exact Hit for: '{query}' (User: {user_id})")
                return row[0]
                
        try:
            prefix = f"user_{user_id}:"
            user_cached_queries = [q[len(prefix):] for q in self.cached_queries if q.startswith(prefix)]
            if not user_cached_queries:
                return None
                
            all_queries = user_cached_queries + [query.strip().lower()]
            matrix = self.vectorizer.fit_transform(all_queries)
            query_vec = matrix[-1]
            cached_matrix = matrix[:-1]
            
            similarities = (cached_matrix * query_vec.T).toarray().flatten()
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            if best_score > 0.93:
                best_query = user_cached_queries[best_idx]
                best_query_key = f"user_{user_id}:{best_query}"
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT response FROM chat_cache WHERE query = ?", (best_query_key,))
                row = cursor.fetchone()
                conn.close()
                if row:
                    print(f"Cache Semantic Hit for: '{query}' matching key: '{best_query}' (User: {user_id}, Score: {best_score:.2f})")
                    return row[0]
        except Exception as e:
            print("Semantic Cache Check error:", e)
            
        return None
        
    def set(self, query, response, user_id: int):
        query_clean = f"user_{user_id}:{query.strip().lower()}"
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

# Initialize Retriever and Cache globally
retriever = SOLIXRetriever()
cache = SemanticCache()

class MessageItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[MessageItem] = []

@router.get("/app/chatbot", response_class=HTMLResponse)
@router.get("/app/chatbot/", response_class=HTMLResponse)
def read_chatbot(request: Request, user: Optional[User] = Depends(get_current_user)):
    return templates.TemplateResponse("app/chatbot.html", {"request": request})

@router.post("/api/chat")
async def chat_endpoint(req: ChatRequest, current_user: User = Depends(get_current_user), _barrier = Depends(CredentialsBarrier(["groq_api_key"]))):
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")
        
    corrected_query = correct_query(query)
    print(f"Chat Endpoint: Original Query: '{query}' -> Corrected: '{corrected_query}'")
    
    user_id = current_user.id if current_user else 0
    cached_response = cache.get(corrected_query, user_id=user_id)
    if cached_response:
        return JSONResponse(content={"response": cached_response, "cached": True})
        
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
            f"\nملاحظة هامة: السؤال يتعلق عضو معين وهو {member['name_ar']}. "
            "اكتب ردًا احترافيًا ومفصلًا يسلط الضوء على دوره، وتخصصه، ومساهماته في منصة SOLIX بناءً على السياق فقط."
        )
    else:
        relevant_chunks = retriever.retrieve(corrected_query, top_k=3)
        if relevant_chunks:
            context = "\n\n---\n\n".join(relevant_chunks)
        
    messages = [{"role": "system", "content": system_instruction}]
    
    for msg in req.history[-6:]:
        api_role = "assistant" if msg.role == "bot" else msg.role
        messages.append({"role": api_role, "content": msg.content})
        
    user_content = f"السؤال: {corrected_query}"
    if context:
        user_content = f"السياق المتاح:\n{context}\n\n" + user_content
        
    messages.append({"role": "user", "content": user_content})
    
    headers = {
        "Authorization": f"Bearer {get_groq_api_key()}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": get_groq_model(),
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            result = response.json()
            try:
                from backend.utils.llm_logger import log_groq_response
                log_groq_response(result, module_name="chat")
            except Exception as e_log:
                print(f"Failed to log chatbot endpoint token usage: {e_log}")
            bot_response = result["choices"][0]["message"]["content"].strip()
            
            if not req.history:
                cache.set(corrected_query, bot_response, user_id=user_id)
                
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


@router.get("/api/chat/sessions")
def get_sessions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    sessions = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).order_by(ChatSession.updated_at.desc()).all()
    return [{
        "id": s.id,
        "title": s.title,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None
    } for s in sessions]


@router.post("/api/chat/sessions")
def create_session(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    new_sess = ChatSession(user_id=current_user.id, title="New Chat")
    db.add(new_sess)
    db.commit()
    db.refresh(new_sess)
    return {
        "id": new_sess.id,
        "title": new_sess.title,
        "created_at": new_sess.created_at.isoformat()
    }


@router.get("/api/chat/sessions/{session_id}/messages")
def get_session_messages(session_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc()).all()
    return [{
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "created_at": m.created_at.isoformat()
    } for m in messages]


@router.delete("/api/chat/sessions/{session_id}")
def delete_session(session_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(session)
    db.commit()
    return {"status": "success"}


class SessionChatRequest(BaseModel):
    message: str

@router.post("/api/chat/sessions/{session_id}/chat")
async def session_chat_endpoint(session_id: str, req: SessionChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db), _barrier = Depends(CredentialsBarrier(["groq_api_key"]))):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")
        
    corrected_query = correct_query(query)
    print(f"Session Chat Endpoint: Session: {session_id}, Query: '{query}' -> Corrected: '{corrected_query}'")
    
    # Check Semantic Cache
    cached_response = cache.get(corrected_query, user_id=current_user.id)
    if cached_response:
        # Save user message to database
        user_msg = ChatMessage(session_id=session_id, role="user", content=query)
        db.add(user_msg)
        # Save bot response to database
        bot_msg = ChatMessage(session_id=session_id, role="bot", content=cached_response)
        db.add(bot_msg)
        
        # Update session title if it was default
        if session.title == "New Chat":
            session.title = query[:40] + ("..." if len(query) > 40 else "")
            
        session.updated_at = datetime.datetime.utcnow()
        db.commit()
        return JSONResponse(content={"response": cached_response, "cached": True})
        
    # Check queries for team, supervisors, etc.
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
            f"\nملاحظة هامة: السؤال يتعلق عضو معين وهو {member['name_ar']}. "
            "اكتب ردًا احترافيًا ومفصلًا يسلط الضوء على دوره، وتخصصه، ومساهماته في منصة SOLIX بناءً على السياق فقط."
        )
    else:
        relevant_chunks = retriever.retrieve(corrected_query, top_k=3)
        if relevant_chunks:
            context = "\n\n---\n\n".join(relevant_chunks)
            
    # Load session message history from database
    history_msgs = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.desc()).limit(6).all()
    history_msgs.reverse() # Back to chronological order
    
    messages = [{"role": "system", "content": system_instruction}]
    for msg in history_msgs:
        api_role = "assistant" if msg.role == "bot" else msg.role
        messages.append({"role": api_role, "content": msg.content})
        
    user_content = f"السؤال: {corrected_query}"
    if context:
        user_content = f"السياق المتاح:\n{context}\n\n" + user_content
        
    messages.append({"role": "user", "content": user_content})
    
    headers = {
        "Authorization": f"Bearer {get_groq_api_key()}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": get_groq_model(),
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            result = response.json()
            try:
                from backend.utils.llm_logger import log_groq_response
                log_groq_response(result, module_name="chat")
            except Exception as e_log:
                print(f"Failed to log session chat token usage: {e_log}")
            bot_response = result["choices"][0]["message"]["content"].strip()
            
            # Save user message to database
            user_msg = ChatMessage(session_id=session_id, role="user", content=query)
            db.add(user_msg)
            
            # Save bot response to database
            bot_msg = ChatMessage(session_id=session_id, role="bot", content=bot_response)
            db.add(bot_msg)
            
            # Update session title if it was default
            if session.title == "New Chat":
                session.title = query[:40] + ("..." if len(query) > 40 else "")
                
            session.updated_at = datetime.datetime.utcnow()
            db.commit()
            
            # Set cache
            cache.set(corrected_query, bot_response, user_id=current_user.id)
            
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

