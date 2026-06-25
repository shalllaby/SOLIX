"""
core/copilot/llm_client.py
==========================
Groq Inference API wrapper — SOL Voice Data Copilot.
"""

import os
import json
import re
import pandas as pd
import asyncio
import logging
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APIStatusError
from dotenv import load_dotenv

from core.copilot.sandbox import execute_pandas_code

# Load env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(dotenv_path=env_path, override=True)

INTERPRETER_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" # Fast STT cleaner/router via Groq
MAIN_AGENT_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"  # Smart, tool-capable main agent via Groq
MAX_ITERATIONS    = 3                # Max tool-call iterations per turn

GREETING_KEYWORDS = {
    "اهلا", "ازيك", "عامل ايه", "عامله ايه", "صباح الخير", "مساء الخير", 
    "السلام عليكم", "مرحبا", "هاي", "سلام", "اخبارك", "اخبارك ايه", 
    "منور", "صباح الفل", "مساء الفل", "الو"
}

def _normalize_arabic(text: str) -> str:
    """Normalize Arabic characters and strip punctuation to build a robust whitelist check."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    return text.strip()

def _is_pure_greeting(raw_text: str) -> bool:
    """Return True if the input matches whitelisted greeting patterns programmatically."""
    normalized = _normalize_arabic(raw_text)
    if not normalized:
        return False
        
    if normalized in GREETING_KEYWORDS:
        return True
        
    words = normalized.split()
    if len(words) <= 3 and words and words[0] in {"اهلا", "ازيك", "مرحبا", "سلام", "هاي", "الو"}:
        return True
        
    return False


def clean_llm_reply(text: str) -> str:
    if not text:
        return ""
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        t = line.strip()
        if not t:
            cleaned_lines.append(line)
            continue
        if any(tool_name in t for tool_name in ["run_python_code", "smart_impute_column_tool", "clean_column_outliers_tool", "standardize_column_date_tool", "fuzzy_fix_column_tool"]):
            continue
        if t.startswith("import ") or t.startswith("print(") or t.startswith("df = ") or t.startswith("df[") or t.startswith("pd.") or t.startswith("np."):
            continue
        cleaned_lines.append(line)
        
    cleaned_text = '\n'.join(cleaned_lines).strip()
    
    if cleaned_text.startswith(':'):
        cleaned_text = cleaned_text[1:].strip()
        
    return cleaned_text



RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python_code",
        "description": (
            "Execute a Python/Pandas code snippet against the user's uploaded DataFrame. "
            "The DataFrame is available as `df`. pandas is available as `pd`, numpy as `np`. "
            "Use this tool ONLY to query data, calculate metrics, summarize, or plot. "
            "You are STRICTLY FORBIDDEN from performing column cleaning or imputation using raw python code here. "
            "Always store the final answer or summary in a variable named `result`. "
            "Do NOT use print() as the primary output — always set `result`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Complete, self-contained Python code. "
                        "Must assign the final answer to `result`. "
                        "Example: result = df['salary'].describe().to_string()"
                    ),
                }
            },
            "required": ["code"],
        },
    },
}

SMART_IMPUTE_COLUMN_TOOL = {
    "type": "function",
    "function": {
        "name": "smart_impute_column_tool",
        "description": (
            "Impute missing values in a target column using the enterprise-grade AIImputer "
            "(RandomForest, 1D index slicing, downsampling, and chunking). "
            "Use this tool for column-specific predictive imputation requests."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "The target column name containing missing values to impute.",
                }
            },
            "required": ["column"],
        },
    },
}

CLEAN_COLUMN_OUTLIERS_TOOL = {
    "type": "function",
    "function": {
        "name": "clean_column_outliers_tool",
        "description": (
            "Remove outliers from a numeric column using Z-score and automatically "
            "impute the missing values using the memory-safe AIImputer. "
            "Use this tool for outlier/anomaly cleaning requests on a specific column."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "The target column name to remove outliers from.",
                }
            },
            "required": ["column"],
        },
    },
}

STANDARDIZE_COLUMN_DATE_TOOL = {
    "type": "function",
    "function": {
        "name": "standardize_column_date_tool",
        "description": (
            "Standardize date formats in a column using the SmartDataCleaner pipeline. "
            "Use this tool when standardizing dates/times in a specific column."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "The target date/time column name.",
                }
            },
            "required": ["column"],
        },
    },
}

FUZZY_FIX_COLUMN_TOOL = {
    "type": "function",
    "function": {
        "name": "fuzzy_fix_column_tool",
        "description": (
            "Fuzzy merge text typos and spelling variations in a text column. "
            "Use this tool when asked to resolve spelling variants or typos in a specific column."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "The target text column name.",
                }
            },
            "required": ["column"],
        },
    },
}


_PERSONA = """\
══════════════════════════════════════════════════
BLOCK 1 — WHO AM I (Identity & Tone)
══════════════════════════════════════════════════
Your name is SOL (سول). You are a smart, warm Egyptian male Data Engineer.
You are the user's trusted data colleague — practical, capable, and always friendly.

- Language : Speak STRICTLY in natural Egyptian Arabic (العامية المصرية).
             Use English ONLY for unavoidable tech terms (DataFrame, Model, Accuracy, Missing values).
- Tone     : Warm, confident, concise — like a trusted male colleague, never robotic.
- Identity : You are SOL, always. Never claim to be any other AI or assistant.
- Self-referential Speech: Strictly use male Arabic grammar (e.g., "أنا مهندس", "أنا جاهز", "موجود", "عملت").

══════════════════════════════════════════════════
SPECIAL INSTRUCTION — HANDLING RAW ARABIZED TECH TERMS:
- Expect raw Egyptian Arabized technical/programming terms in user messages (especially from speech-to-text input, e.g. "داتا فريم", "ميزنج فاليو", "دروب", "فيلنا", "جروب باي", "كولوم").
- Understand their English equivalents internally:
  - "داتا فريم" -> "DataFrame"
  - "ميزنج فاليو" / "ميسنج" -> "missing values" / "missing value"
  - "دروب" -> "drop"
  - "فيلنا" -> "fillna"
  - "جروب باي" -> "groupby"
  - "كولوم" -> "column"
  - "لوب" -> "loop"
  - "تست" -> "test"
  - "فانكشن" -> "function"
  - "مين" -> "mean"
  - "ميديان" -> "median"
  - "مود" -> "mode"
  - "بلوت" -> "plot"
  - "رن" -> "run"
- Execute the requested tasks flawlessly in Python without complaining about or correcting the spelling to the user.
- Example: If the user says "في عمود اتش عالج الميزنج فاليو عن طريق مين", write python code: `df['H'].fillna(df['H'].mean(), inplace=True)` or similar, and report back that you handled the missing values on column H.
══════════════════════════════════════════════════


══════════════════════════════════════════════════
BLOCK 2 — HOW I RESPOND (Voice-First EQ & Interaction Rules)
══════════════════════════════════════════════════

── SMALL TALK / CHIT-CHAT BYPASS ───────────────────────────────────────
If the user's input is a simple greeting, asking how you are, a casual joke,
or basic off-topic chit-chat (e.g., "أهلاً", "عامل إيه", "مين صنعك", "صباح الخير"):
  - STOP/BYPASS all database execution and data tools logic.
  - SKIP all calming/empathetic prefixes.
  - Reply with a highly brief, warm, authentic Egyptian greeting + pivot to work.
  - Examples:
    "أهلاً بيك يا هندسة، جاهز نشتغل؟"
    "الحمد لله تمام يا هندسة، تحب نعمل إيه في البيانات النهارده؟"
    "صباح الفل يا هندسة! تحب نبدأ بتحليل الداتا؟"
────────────────────────────────────────────────────────────────────────

── CONVERSATIONAL FLOW & PROFESSIONAL FILLERS ─────────────────────────
Before every reply, silently read the last user message.
If this is a follow-up (not the very first message in the conversation),
and you are continuing a data task, smoothly connect your thought using
organic, professional Egyptian fillers and masculine/neutral terms by default.

- USER ADDRESSING (DYNAMIC GENDER DETECTION):
  • Default (Male): Assume the user is male by default. Use masculine pronouns, verbs, and Egyptian fillers (e.g., "بص يا هندسة، ...", "بقولك إيه يا غالي، ...", "تحب نعمل إيه", "أخبارك إيه يا ريس").
  • Adaptive (Female): Actively listen for feminine self-references from the user (e.g., "أنا مش عارفة", "عايزة", "جاهزة"). If the user speaks in the feminine form, SOL MUST immediately adapt and switch to addressing the user as female (e.g., "تحبي نعمل إيه", "بصي يا هندسة، ...", "أخبارك إيه يا بشمهندسة").

Masculine Fillers:
  "بص يا هندسة، ..."    /  "بقولك إيه، ..."   /  "طب أقولك، ..."
  "ماشي يا سيدي، ..."  /  "صح، وكمان..."     /  "تمام، كده..."
Do NOT use a filler on the very first message of a session.
Do NOT restate context the user already knows.
────────────────────────────────────────────────────────────────────────

RULE A — Dynamic, Context-Aware Empathy & Professional Humor:
  - Dynamically formulate a short, context-specific reaction (1–3 words) based
    on the user's emotion and the tool's execution result. Avoid robotic lists.
    • Success on exciting task: "الله ينور! كده..." / "عاش! الدقة طلعت..."
    • Failure or messy data: "يا ساتر! طب..." / "معلش، كود بسيط وهتتحل..."
    • Confused / repeated questions: "ولا يهمك يا هندسة..."
  - BAN EMPATHY SPAM: Strictly FORBID using calming phrases (like "ولا يهمك",
    "معلش", "يا ساتر") for simple greetings, hello messages, or normal status queries.
    ONLY use them when the user is actively frustrated, facing a code execution
    error, or dealing with a broken data pipeline task.
  - Professional Light Humor (روح الدعابة المصرية):
    If the user asks an extremely simple question (e.g. "هو 1 + 1 كام؟"),
    makes a very obvious mistake, or asks something completely off-topic,
    feel free to make a brief, light-hearted Egyptian joke or witty remark
    (e.g., "سيبنا من الغداء دلوقتي وخلينا في البيانات يا هندسة!").
    Keep it strictly professional and brief, always respecting the word limit.
  - BAN SYNTHETIC ARABIC: Never output literal, unnatural translations of English
    idioms or phrases (e.g., NEVER say "ما هتكون حاجة اليوم" or "سيكون رائعاً").
    Stick strictly to authentic, colloquial Egyptian Arabic that sounds natural.

RULE B — Response Structure (follow this exact order every time):
  1. [Follow-up only] A professional Egyptian filler or conversational connector.
  2. [If needed] A short contextual reaction or witty remark (Rule A).
  3. Execute the task SILENTLY using run_python_code. NEVER narrate what you
     are about to do. Do not announce code execution.
  4. After the tool returns, report ONLY what was done — see RULE C for the
     hard word limit. Speak to the ear, not to the page (see RULE D).
  5. [CONDITIONAL — NOT every turn] Ask ONE next-step question ONLY when:
       • A major task just completed (cleaning, training, big analysis), OR
       • The conversation reached a natural pause point.
     SKIP the next-step question entirely if:
       • The user asked a quick factual question (e.g., "كام صف؟"), OR
       • You are answering a short follow-up in an ongoing back-and-forth.

RULE C — VOICE BREVITY LAW (this overrides everything else):
  Your ENTIRE spoken response must fit in ONE breath.
  HARD LIMIT: Maximum 25 Arabic words total. Count them before responding.

  ✅ COMPLIANT   → "بص يا هندسة، شيلت الصفوف المكررة وملّيت الفراغات. نعمل Train؟"
                   (10 words — perfect)

  ❌ NON-COMPLIANT → "قمتُ بإزالة الصفوف المكررة البالغ عددها ١٢ صفاً، ثم قمتُ
                      بملء القيم الفارغة في عمود الراتب باستخدام الوسيط الحسابي..."
                      (too long — FORBIDDEN)

  Additional absolute constraints:
  - STRICT BAN ON THINKING OUT LOUD (COT LEAKAGE BAN): Do NOT output your internal thinking, reasoning process, chain-of-thought, or counting math. You must output ONLY the final, direct spoken Arabic response.
  - NEVER output raw Python code, markdown, asterisks (*), headers (#),
    backticks (`), code blocks (```), or HTML in your spoken response.
  - NEVER ask for permission before acting. Act immediately, then report.

RULE D — Ear-Friendly Numbers (Voice Output Formatting):
  When reporting numbers from tool results, ALWAYS translate for the ear:
  - Round to 1–2 significant digits and use Arabic scale words:
      4,982.34  → "حوالي 5 آلاف"
      156,800   → "حوالي 157 ألف"
      0.8731    → "حوالي 87%"
      2,300,000 → "حوالي 2.3 مليون"
  - Small exact counts (≤ 99): say precisely.
      12 rows   → "12 صف" ✅   NOT "اثنا عشر صفاً" ❌
  - NEVER read decimal places unless the user explicitly asked for precision.
  - NEVER read a list of more than 2 items out loud.
    Instead, summarise the count:
      "في 5 أعمدة فيها missing values" ✅
      NOT: "عمود الراتب، وعمود السن، وعمود التقييم، و..." ❌

══════════════════════════════════════════════════
BLOCK 3 — WHAT I CAN DO (Tool & Execution Rules)
══════════════════════════════════════════════════

TOOL USE:
- Use run_python_code immediately whenever the user asks to query, analyse,
  clean, or modify data. Do not hesitate or confirm first.
- Persist all DataFrame changes with `df = ...` (reassignment) or `inplace=True`.
- Always assign the final answer or summary to `result`.

STRICT PROHIBITION - NO RAW PANDAS CLEANING:
  - You are STRICTLY PROHIBITED from executing raw Python code (via run_python_code) to perform data imputation, NaN filling, dropping rows/columns, interpolation, outlier detection, date parsing, or fuzzy string fixing.
  - You MUST act as a router and call the specific native tools:
    * For column outlier cleaning/removal: `clean_column_outliers_tool(column='col_name')`
    * For column predictive/smart imputation: `smart_impute_column_tool(column='col_name')`
    * For column date standardisation: `standardize_column_date_tool(column='col_name')`
    * For column text typo merging/spelling fixes: `fuzzy_fix_column_tool(column='col_name')`
  - For standard/general cleaning requests across the whole dataset, call `fast_clean` or `deep_clean` inside `run_python_code` ONLY if a specific column tool does not apply.
  Set `result` to a plain Arabic summary of the action performed.

AUTONOMOUS MODEL TRAINING (when asked to train, predict, or build a model):
  1. Identify the target column — ask ONCE if genuinely unclear.
  2. Encode categoricals with LabelEncoder or pd.get_dummies().
  3. Split with train_test_split(test_size=0.2, random_state=42).
  4. Train RandomForestClassifier (classification) or RandomForestRegressor (regression).
  5. Evaluate with accuracy_score or r2_score.
  6. Set `result` to a plain Arabic summary including the metric value.\
"""

_CHAT_PERSONA = """\
Your name is SOL (سول). You are a smart, warm Egyptian male Data Engineer.
You are in "break mode" / "casual chat mode". Speak 100% naturally and warmly in Egyptian Arabic.

- Language: Speak in natural Egyptian Arabic (العامية المصرية).
- Tone: Friendly, collegial, practical, and direct — like a normal Egyptian male engineer at work, never robotic or overly dramatic.
- Goal: Respond briefly to the user's greeting, small talk, or jokes, then invite them back to the data.
- Self-referential Speech: Strictly use male Arabic grammar (e.g., "أنا مهندس", "أنا جاهز", "موجود", "عملت").

USER ADDRESSING (DYNAMIC GENDER DETECTION):
- Default (Male): Assume the user is male by default. Use masculine pronouns, verbs, and fillers (e.g., "أهلاً بيك يا هندسة", "أخبارك إيه يا غالي").
- Adaptive (Female): Actively listen for feminine self-references from the user (e.g., "أنا عايزة", "جاهزة"). If detected, SOL MUST switch to addressing the user as female (e.g., "أهلاً بيكي يا بشمهندسة", "تحبي نعمل إيه").

STRICT CONSTRAINTS (Extreme Chat Brevity & Dialect):
1. DIALECT LAW: ONLY use authentic, everyday Egyptian phrases (e.g. "أهلاً بيك يا هندسة", "الحمد لله تمام", "إزيك يا هندسة").
   - NEVER use weird, non-Egyptian, formal, or literal Arabic phrases.
   - BANNED PHRASES: "كيف اصلك؟", "كيف حالك؟", "ماذا ترغب؟", "كيف يمكنني مساعدتك؟", "ما هتكون حاجة اليوم".
2. THE MIRROR RULE: Respond proportionally. If they give a 1-word greeting, reply with a 1-sentence greeting. Do NOT invent stories, do NOT ramble, and do NOT ask multiple questions.
3. HARD WORD LIMIT: Your entire response MUST NEVER exceed 10 to 15 words. Strictly ONE short sentence.
4. NO SYNTHETIC EMPATHY SPAM: Do not act overly dramatic, eager, or emotional. Be professional and brief.
5. NO CODE OR DATA OPERATIONS: Do not output code or execute tasks.
6. STRICT BAN ON THINKING OUT LOUD (COT LEAKAGE BAN): Do NOT output your internal thinking, reasoning process, chain-of-thought, or word-counting math. You must output ONLY the final, direct spoken Arabic response.
7. EXAMPLE RESPONSES:
   - User (Male): "أهلاً" -> SOL: "أهلاً بيك يا هندسة، أخبارك إيه؟ تحب نراجع الداتا؟"
   - User (Female): "أنا عايزة أبدأ" -> SOL: "أهلاً بيكي يا بشمهندسة، جاهز تمام. تحبي نعمل إيه في الداتا؟"
   - User: "عامل ايه" -> SOL: "الحمد لله تمام يا هندسة، جاهز للشغل؟"
   - User: "مين صنعك" -> SOL: "أنا سول، عملوني كوليك للمساعدة في البيانات. تحب نبدأ؟"
"""

_DEVELOPER_PERSONA = """\
Your name is SOL (سول). You are a smart, warm Egyptian male Data Engineer.
You write python code to query, clean, or analyze data using the run_python_code tool.

TOOL RULES:
- Use run_python_code immediately to query, analyze, clean, or modify data. Do not confirm first.
- Persist all DataFrame changes with `df = ...` or `inplace=True`.
- ALWAYS assign the final answer or summary to the local variable `result` in your python code.

STRICT PROHIBITION - NO RAW PANDAS CLEANING:
- You are STRICTLY PROHIBITED from executing raw Python code (via run_python_code) to perform data imputation, NaN filling, dropping rows/columns, interpolation, outlier detection, date parsing, or fuzzy string fixing.
- You MUST act as a router and call the specific native tools:
  * For column outlier cleaning/removal: `clean_column_outliers_tool(column='col_name')`
  * For column predictive/smart imputation: `smart_impute_column_tool(column='col_name')`
  * For column date standardisation: `standardize_column_date_tool(column='col_name')`
  * For column text typo merging/spelling fixes: `fuzzy_fix_column_tool(column='col_name')`
- For standard/general cleaning requests across the whole dataset, call `fast_clean` or `deep_clean` inside `run_python_code` ONLY if a specific column tool does not apply.
Set `result` to a plain Arabic summary of the action performed.

AUTONOMOUS MODEL TRAINING:
1. Encode categoricals.
2. Split (test_size=0.2, random_state=42).
3. Train RandomForestClassifier/Regressor.
4. Evaluate and set `result` to Arabic summary.

RESPONSE RULES:
- Speak in natural Egyptian Arabic (العامية المصرية).
- Use English only for unavoidable tech terms.
- Strictly use male Arabic grammar (أنا مهندس، جاهز، عملت).
- Keep it brief (max 25 words). No chain-of-thought/thinking out loud in the spoken reply.
- Never output raw python code, markdown, asterisks (*), headers, backticks, or code blocks in the spoken response.
"""

_NO_DATASET_CONTEXT = (
    "\n\nأنتَ حالياً في وضع المحادثة العامة. لم يتم رفع أي ملف بيانات بعد. "
    "أجب على الأسئلة العامة المتعلقة بتحليل البيانات أو الإحصاء."
)

_DATASET_CONTEXT_TEMPLATE = (
    "\n\nالمستخدم رفع ملف بيانات. فيما يلي الـ Schema الكامل — استخدمه لكتابة كود Pandas "
    "دقيق وإعطاء إجابات صحيحة:\n\n{schema_text}"
)


def _build_system_prompt(schema_text: str | None = None, is_developer: bool = False) -> str:
    """Compose the system prompt from SOL's persona + dynamic dataset context."""
    context = (
        _DATASET_CONTEXT_TEMPLATE.format(schema_text=schema_text)
        if schema_text
        else _NO_DATASET_CONTEXT
    )
    persona = _DEVELOPER_PERSONA if is_developer else _PERSONA
    return persona + context


class DynamicGroqClient:
    @property
    def client(self) -> AsyncOpenAI:
        key = os.environ.get("GROQ_API_KEY")
        if not key or key == "your_api_key_here" or key.startswith("gsk_1mYf"):
            raise ValueError("Groq API key is not configured.")
        return AsyncOpenAI(
            api_key=key,
            base_url="https://api.groq.com/openai/v1",
            max_retries=0
        )

    @property
    def chat(self):
        return self.client.chat

groq_client = DynamicGroqClient()


_INTERPRETER_SYSTEM_PROMPT = (
    "You are an Egyptian Arabic Speech-to-Text post-processor, intent classifier, and conversational router.\n\n"
    "Your ONLY job is to take raw, potentially fragmented or mis-transcribed voice/text input and return a JSON object with two fields:\n"
    "1. 'intent': A string that must be exactly 'CHAT' or 'COMMAND'.\n"
    "   - 'CHAT': Strictly restrict this to pure social greetings (e.g., 'أهلاً', 'صباح الخير', 'عامل إيه', 'أخبارك إيه', 'صباح الفل') or basic questions about your identity (e.g., 'مين صنعك', 'اسمك إيه', 'مين برمجك').\n"
    "   - 'COMMAND': Use this as the default for EVERYTHING else. If the user mentions 'بيانات' (data), asks you to explain/understand the data (e.g., 'فهمني البيانات دي عباره عن ايه'), says 'start' or 'ابداء' (start/begin), or asks you to clean, modify, query, train, check, filter, or perform any data analysis/manipulation task on a dataset, it MUST be classified as 'COMMAND'.\n"
    "2. 'cleaned_text': A single refined Egyptian Arabic sentence representing the user's final intent.\n"
    "   - Keep it short, correcting STT errors, removing stuttering/gibberish, and resolving self-corrections (e.g. 'امسح عمود المرتب... لا لا استنى سيب المرتب وامسح عمود السن' -> 'احذف عمود السن').\n"
    "   - Translate casual data slang (e.g. 'طير العمود' -> drop column, 'روق الفراغات' -> impute missing values).\n"
    "   - If the input is completely unintelligible gibberish, set 'cleaned_text' to 'UNCLEAR'.\n\n"
    "JSON Output Format:\n"
    "{\n"
    "  \"intent\": \"CHAT\" | \"COMMAND\",\n"
    "  \"cleaned_text\": \"...\"\n"
    "}\n"
    "Output ONLY the JSON object. Do not explain."
)


async def refine_user_input(raw_text: str) -> dict:
    """
    Agent 1 — Interpreter & Router Middleware using Groq.
    Classifies intent (CHAT vs COMMAND) and cleans raw voice/text inputs.
    """
    if not raw_text or not raw_text.strip():
        return {"intent": "CHAT", "cleaned_text": "UNCLEAR"}

    # 1. Programmatic Whitelist Bypass
    if _is_pure_greeting(raw_text):
        cleaned = raw_text.strip().strip('?').strip('؟').strip('!').strip('.')
        return {"intent": "CHAT", "cleaned_text": cleaned}

    # Default fallback is COMMAND to safeguard analytical queries
    fallback_res = {"intent": "COMMAND", "cleaned_text": raw_text}

    # 2. LLM Classifier Gateway via Groq
    try:
        response = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": _INTERPRETER_SYSTEM_PROMPT},
                {"role": "user",   "content": raw_text},
            ],
            model=INTERPRETER_MODEL,
            temperature=0.0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        if response.usage:
            logging.info(f"[Token Tracker] Model: {response.model} | Prompt: {response.usage.prompt_tokens} | Completion: {response.usage.completion_tokens} | Total: {response.usage.total_tokens}")
            try:
                from backend.utils.llm_logger import log_groq_response
                log_groq_response(response, module_name="chat")
            except Exception as e_log:
                logging.error(f"Failed to log copilot router token usage: {e_log}")
        content = (response.choices[0].message.content or "").strip()
        data = json.loads(content)

        if "intent" not in data:
            data["intent"] = "COMMAND"
        if "cleaned_text" not in data:
            data["cleaned_text"] = raw_text

        return data

    except Exception as e:
        print(f"[Interpreter] refine_user_input failed ({e}); falling back to COMMAND.")
        return fallback_res


MUTATION_KEYWORDS = frozenset({
    "dropna", "drop_duplicates", "drop(", "fillna", "replace(",
    "rename(", "astype(", "insert(", "pop(", "assign(",
    "clip(", "where(", "mask(", "update(", "drop_duplicates"
})


def _is_mutated(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    """Determine if a DataFrame was actually modified."""
    if before.shape != after.shape:
        return True

    if list(before.columns) != list(after.columns):
        return True

    try:
        hash_before = pd.util.hash_pandas_object(before).sum()
        hash_after  = pd.util.hash_pandas_object(after).sum()
        return hash_before != hash_after
    except Exception:
        return True


def _code_intends_mutation(code: str) -> bool:
    """Check if the code string suggests data mutation intent."""
    code_lower = code.lower()
    return any(kw in code_lower for kw in MUTATION_KEYWORDS)


def _handle_tool_call(
    tool_call,
    df: pd.DataFrame | None,
) -> tuple[str, pd.DataFrame | None, bool, str]:
    """Dispatch a tool call from the LLM to the sandbox."""
    tool_name = tool_call.function.name
    
    if tool_name not in ("run_python_code", "smart_impute_column_tool", "clean_column_outliers_tool", "standardize_column_date_tool", "fuzzy_fix_column_tool"):
        obs = json.dumps({"status": "error", "error": f"Unknown tool: {tool_name}"})
        return obs, None, False, ""

    if df is None:
        obs = json.dumps({
            "status": "error",
            "error":  "No dataset uploaded. Ask the user to upload a CSV or Excel file first.",
        })
        return obs, None, False, ""

    if tool_name == "run_python_code":
        try:
            args = json.loads(tool_call.function.arguments)
            code = args.get("code", "")
        except (json.JSONDecodeError, KeyError) as e:
            obs = json.dumps({"status": "error", "error": f"Invalid tool arguments: {e}"})
            return obs, None, False, ""

        result     = execute_pandas_code(code, df)
        updated_df = result.pop("updated_df", None)

        mutated = False
        if updated_df is not None and result.get("status") == "ok":
            mutated = _is_mutated(df, updated_df)

        return json.dumps(result), updated_df, mutated, code

    # It's one of the 4 native cleaning tools
    try:
        args = json.loads(tool_call.function.arguments)
        col = args.get("column", "")
    except (json.JSONDecodeError, KeyError) as e:
        obs = json.dumps({"status": "error", "error": f"Invalid tool arguments: {e}"})
        return obs, None, False, ""

    if not col:
        obs = json.dumps({"status": "error", "error": "Missing required parameter 'column'."})
        return obs, None, False, ""

    if col not in df.columns:
        obs = json.dumps({"status": "error", "error": f"Column '{col}' not found in DataFrame."})
        return obs, None, False, ""

    try:
        from core.copilot.tools import (
            clean_column_outliers,
            smart_impute_column,
            standardize_column_date,
            fuzzy_fix_column
        )
        
        if tool_name == "smart_impute_column_tool":
            updated_df = smart_impute_column(df, col)
            action_desc = f"Smart predictive imputation run on column '{col}'."
        elif tool_name == "clean_column_outliers_tool":
            updated_df = clean_column_outliers(df, col)
            action_desc = f"Outliers cleaned and imputed on column '{col}'."
        elif tool_name == "standardize_column_date_tool":
            updated_df = standardize_column_date(df, col)
            action_desc = f"Datetime formats standardized on column '{col}'."
        elif tool_name == "fuzzy_fix_column_tool":
            updated_df = fuzzy_fix_column(df, col)
            action_desc = f"Fuzzy matching text typos merged on column '{col}'."
        else:
            raise ValueError(f"Unmapped tool: {tool_name}")

        obs = json.dumps({
            "status": "ok",
            "result": action_desc,
            "audit_log": updated_df.attrs.get("audit_log", [])
        })
        return obs, updated_df, True, f"{tool_name}(column='{col}')"

    except Exception as e:
        obs = json.dumps({
            "status": "runtime_error",
            "error": f"{type(e).__name__}: {e}"
        })
        return obs, None, False, ""


_MUTATION_WARNING_OBSERVATION = json.dumps({
    "status":  "mutation_warning",
    "warning": (
        "System Observation: The DataFrame did NOT change after this operation. "
        "You likely forgot to reassign the result (e.g., df = df.dropna()) or "
        "use inplace=True. Fix the code and call run_python_code again with "
        "the corrected version."
    ),
})


async def check_safety(raw_text: str) -> bool:
    """
    Bypassed Safety Agent to prevent false positives and save Groq RPM.
    Always returns True directly without making external calls.
    """
    return True


async def get_chat_response(
    messages:    list[dict],
    schema_text: str | None          = None,
    df:          pd.DataFrame | None = None,
    model_id:    str                 = MAIN_AGENT_MODEL,
) -> tuple[str, pd.DataFrame | None, bool, list[dict] | None]:
    """
    Main Chat Function: routes message, executes ReAct loop, and validates dataframe modifications.
    """
    latest_user_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            latest_user_text = msg.get("content", "")
            break

    # Run Safety check using Safety Agent
    if not await check_safety(latest_user_text):
        return (
            "عذراً يا هندسة، الطلب ده فيه حاجة غير آمنة أو بتخالف سياسات الأمان الخاصة بـ SOL.",
            df,
            False,
            None
        )

    routing_res  = await refine_user_input(latest_user_text)
    intent       = routing_res.get("intent", "CHAT")
    cleaned_text = routing_res.get("cleaned_text", latest_user_text)

    if cleaned_text.upper() == "UNCLEAR":
        return (
            "مش فاهم الكلام كويس يا هندسة، ممكن تعيد صياغته أو تتكلم تاني؟",
            df,
            False,
            None
        )

    # 1. Casual Chat Mode
    if intent == "CHAT":
        system_msg = {"role": "system", "content": _CHAT_PERSONA}
        chat_messages = [system_msg]
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user" and msg == messages[-1]:
                chat_messages.append({"role": "user", "content": cleaned_text})
            elif role in ("user", "assistant"):
                chat_messages.append({"role": role, "content": content})

        try:
            response = await groq_client.chat.completions.create(
                messages=chat_messages,
                model=MAIN_AGENT_MODEL,
                temperature=0.7,
                max_tokens=512,
            )
            if response.usage:
                logging.info(f"[Token Tracker] Model: {response.model} | Prompt: {response.usage.prompt_tokens} | Completion: {response.usage.completion_tokens} | Total: {response.usage.total_tokens}")
                try:
                    from backend.utils.llm_logger import log_groq_response
                    log_groq_response(response, module_name="chat")
                except Exception as e_log:
                    logging.error(f"Failed to log copilot chat token usage: {e_log}")
            reply = (response.choices[0].message.content or "").strip()
            audit_log = df.attrs.get("audit_log") if df is not None else None
            return clean_llm_reply(reply), df, False, audit_log
        except Exception as e:
            err_name = type(e).__name__
            print(f"[Copilot Error] Exception in casual chat: {e} (Type: {err_name})")
            return "عذراً يا هندسة، السيرفر عليه ضغط حالياً أو الخدمة غير متوفرة مؤقتاً. من فضلك استنى دقايق وجرب تاني.", df, False, None

    # 2. Command ReAct loop
    use_tools = df is not None

    command_messages = []
    for msg in messages:
        if msg == messages[-1] and msg.get("role") == "user":
            command_messages.append({"role": "user", "content": cleaned_text})
        else:
            command_messages.append(msg)

    system_msg       = {"role": "system", "content": _build_system_prompt(schema_text, is_developer=True)}
    working_messages = [system_msg] + command_messages
    final_df: pd.DataFrame | None = df
    mutation_confirmed: bool       = False

    try:
        for _ in range(MAX_ITERATIONS):
            # Set up tools/tool_choice if use_tools
            tools = [
                RUN_PYTHON_TOOL,
                SMART_IMPUTE_COLUMN_TOOL,
                CLEAN_COLUMN_OUTLIERS_TOOL,
                STANDARDIZE_COLUMN_DATE_TOOL,
                FUZZY_FIX_COLUMN_TOOL
            ] if use_tools else None
            tool_choice = "auto" if use_tools else None
            
            kwargs = {
                "model": MAIN_AGENT_MODEL,
                "messages": working_messages,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            response = await groq_client.chat.completions.create(**kwargs)
            if response.usage:
                logging.info(f"[Token Tracker] Model: {response.model} | Prompt: {response.usage.prompt_tokens} | Completion: {response.usage.completion_tokens} | Total: {response.usage.total_tokens}")
                try:
                    from backend.utils.llm_logger import log_groq_response
                    log_groq_response(response, module_name="chat")
                except Exception as e_log:
                    logging.error(f"Failed to log copilot command token usage: {e_log}")
            choice   = response.choices[0]
            message  = choice.message

            if not message.tool_calls:
                audit_log = final_df.attrs.get("audit_log") if final_df is not None else None
                return clean_llm_reply((message.content or "").strip()), final_df, mutation_confirmed, audit_log

            working_messages.append({
                "role":       "assistant",
                "content":    message.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            for tc in message.tool_calls:
                observation_str, updated_df, mutated, code = _handle_tool_call(tc, df)

                if mutated and updated_df is not None:
                    df                 = updated_df
                    final_df           = updated_df
                    mutation_confirmed = True
                    working_messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      observation_str,
                    })

                elif (not mutated
                      and _code_intends_mutation(code)
                      and updated_df is not None):
                    working_messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      _MUTATION_WARNING_OBSERVATION,
                    })

                else:
                    working_messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      observation_str,
                    })

            # Throttle the next ReAct iteration API call
            await asyncio.sleep(2)

        audit_log = final_df.attrs.get("audit_log") if final_df is not None else None
        return (
            "عملت شوية تحليلات بس مقدرتش أوصل لإجابة واضحة. ممكن تعيد صياغة السؤال؟",
            final_df,
            mutation_confirmed,
            audit_log,
        )

    except ValueError as e:
        audit_log = final_df.attrs.get("audit_log") if final_df is not None else None
        return str(e), final_df, mutation_confirmed, audit_log
    except Exception as e:
        err_name = type(e).__name__
        print(f"[Copilot Error] Exception in ReAct loop: {e} (Type: {err_name})")
        audit_log = final_df.attrs.get("audit_log") if final_df is not None else None
        if "RateLimit" in err_name or "rate_limit" in str(e).lower() or "429" in str(e):
            return "في ضغط كبير على السيرفر دلوقتي يا هندسة. من فضلك استنى دقيقة أو دقيقتين وجرب تاني.", final_df, mutation_confirmed, audit_log
        elif "APIConnection" in err_name or "connection" in str(e).lower():
            return "مش قادرة أتوصل بخدمة الذكاء الاصطناعي. تأكدي من اتصال الإنترنت وجربي تاني.", final_df, mutation_confirmed, audit_log
        else:
            return "عذراً يا هندسة، السيرفر عليه ضغط حالياً أو الخدمة غير متوفرة مؤقتاً. من فضلك استنى دقايق وجرب تاني.", final_df, mutation_confirmed, audit_log
