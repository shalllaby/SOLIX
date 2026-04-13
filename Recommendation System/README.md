# ML Advisor API

هذا المشروع عبارة عن واجهة برمجة تطبيقات (API) مبنية باستخدام **FastAPI**، وظيفتها تحليل البيانات (CSV) واستخراج خصائصها (Metadata)، ثم استخدام الذكاء الاصطناعي عبر **Groq API** (نموذج Llama 3) لتقديم ترشيحات لأفضل 3 نماذج تعلم آلي (Machine Learning Models) تناسب هذه البيانات.

## المميزات (Features)
- 📊 **تحليل البيانات (Data Profiling):** استخراج معلومات مفصلة عن البيانات مثل نسبة القيم المفقودة، نوع تصنيف البيانات (تصنيف ثنائي، متعدد، أو انحدار)، التواء البيانات (Skewness)، ونسبة الأعمدة النصية للرقمية.
- 🧠 **الاستنتاج بالذكاء الاصطناعي (AI Reasoning):** معالجة خصائص البيانات باستخدام نموذج Llama 3 القوي عبر Groq API.
- 🚀 **توصيات دقيقة (Model Recommendations):** ترشيح أفضل 3 خوارزميات تعلم آلي للبيانات المرفوعة، مع ذكر مميزات وعيوب كل خوارزمية.
- 🌐 **واجهة برمجة تطبيقات سريعة (FastAPI):** توفير نقطة اتصال (Endpoint) سريعة وفعّالة تدعم رفع الملفات.

## الأدوات والتقنيات المستخدمة (Tech Stack)
- **لغة البرمجة:** [Python 3.x](https://www.python.org/)
- **إطار عمل الـ API:** [FastAPI](https://fastapi.tiangolo.com/)
- **معالجة البيانات:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)
- **الذكاء الاصطناعي:** [Groq API](https://groq.com/) (Llama-3.3-70b-versatile)
- **أدوات أخرى:** `urllib` للاتصال بالـ API بدون مكتبات خارجية، `uvicorn` لتشغيل الخادم (Server).

## هيكل الملفات (File Structure)

```text
TASK project/
│
├── main.py                # الملف الرئيسي لتشغيل الخادم (FastAPI Application) وتعريف الـ Endpoints
├── ml_advisor.py          # النواة الأساسية للمشروع، يحتوي على منطق تحليل البيانات والاتصال بـ Groq API
├── test 1.csv             # ملف بيانات تجريبي
├── titanic_clean.csv      # ملف بيانات تجريبي جاهز ونظيف (Titanic Dataset)
└── README.md              # هذا الملف (دليل المشروع)
```

## تفاصيل الملفات

### 1. `main.py`
هو المدخل الرئيسي لتطبيق الويب (FastAPI)، يحتوي على نقطتي اتصال (Endpoints):
- `GET /`: نقطة فحص حالة السيرفر (Health Check).
- `POST /recommend-models`: النقطة الخاصة برفع ملف הـ `CSV` وتحديد اسم العمود المستهدف (`target_column`). تقوم بقراءة الملف وتحويله إلى `DataFrame` ثم إرساله إلى `ml_advisor` وإرجاع التوصيات بصيغة JSON.

### 2. `ml_advisor.py`
يتكون من عدة أجزاء رئيسية:
- **Config:** إعدادات الـ API الخاصة بـ Groq.
- **The Metadata Profiler (`_profile_metadata`):** وظيفة لتحليل البيانات إحصائيًا من مكتبات `pandas` و `scipy` وتجهيز (Metadata Packet).
- **The Reasoning Engine:** إعداد الـ Prompt وإرسال الـ Metadata إلى الـ Groq API باستخدام `urllib`.
- **Public API (`get_recommendations`):** الدالة الرئيسية التي يتم استدعاؤها في `main.py` والتي تدير العملية بأكملها.

## كيفية التشغيل (How to Run)

1. **تثبيت المكتبات المطلوبة:**
   تحتاج إلى تثبيت المكتبات الأساسية اذا لم تكن مثبتة:
   ```bash
   pip install fastapi uvicorn pandas numpy scipy
   ```

2. **تشغيل الخادم (Run the Server):**
   افتح موجه الأوامر (Terminal) في نفس مسار المشروع واكتب:
   ```bash
   uvicorn main:app --reload
   ```

3. **تجربة الـ API:**
   - يمكنك زيارة التوثيق التفاعلي الخاص بـ FastAPI عبر المتصفح عن طريق الرابط:  
     `http://127.0.0.1:8000/docs`
   - من خلال هذا الرابط يمكنك رفع ملف الطيران (test 1.csv أو titanic_clean.csv) وتجربة الـ Endpoint الخاصة بـ `/recommend-models`.
   - تأكد من ادخال اسم العمود المستهدف (Target Column) الصحيح (مثلاً `Survived` في بيانات التيتانيك).

## ملاحظات (Notes)
- تأكد من أن مفتاح الـ API الخاص بـ (Groq) يعمل بشكل سليم داخل مصفوفة `CONFIG` في ملف `ml_advisor.py`.
- يمكن تعديل درجة الحرارة (Temperature) للموديل من نفس الإعدادات بناءً على مدى دقة الترشيحات المطلوبة.
