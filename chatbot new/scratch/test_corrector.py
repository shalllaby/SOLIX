import re
import difflib
import requests

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

if __name__ == "__main__":
    test_cases = [
        "مين محد شلبى",
        "دكتوره سيمو",
        "طريقه تثبي سولكس",
        "هاكثون ناسا",
        "اهلين",
        "كيف حالك",
        "ميار ممدوح بتعمل ايه",
        "مين المشرفون على المشروع",
        "اعضا الفري"
    ]
    
    print("Running Spelling Correction Verification Tests...\n")
    for case in test_cases:
        print(f"Input:         '{case}'")
        print(f"Local Fuzzy:   '{correct_query_locally(case)}'")
        print(f"LLM Corrected: '{correct_query_via_llm(case)}'")
        print(f"Final Output:  '{correct_query(case)}'")
        print("-" * 50)
