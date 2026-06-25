import pandas as pd
from groq import Groq
import json
import traceback

def test():
    df = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Price': [10000, -15000000, 30000, 40000, 50000],
        'Year': [2010, -10100, 2020, 2015, 2022],
        'Name': ['Ahmed', 'Ali', 'Sara', 'Mona', 'Omar'],
        'Brand': ['Toyota', 'Honda', 'Toyota', 'Ford', 'BMW'],
        'Phone': ['010123456', '011234567', '012345678', '015123456', '010987654']
    })
    sample_df = df.head(5).to_csv(index=False)
    prompt = f"""أنت خبير في علم البيانات وتحليل هيكل قواعد البيانات.
يوجد لديك عينة من 5 صفوف لبيانات تريد تدريب شبكة عصبية عليها.
احذر من الخلط بين الأرقام (مثل سنوات الصنع أو الأسعار) وبين الهويات. لا تقم أبداً بتحويل الأرقام الحسابية إلى بيانات حساسة.

عينة البيانات (بصيغة CSV):
{sample_df}

المطلوب:
لكل عمود في هذه البيانات، حدد نوعه بناءً على محتواه ومعناه، وليس فقط اسمه.
يجب أن يكون النوع واحداً من التالي حصراً:
- "numerical": للبيانات الرقمية (أسعار، سعة محرك، مسافة، رواتب، سنوات صنع، وأي أرقام لها معنى رياضي).
- "categorical": للبيانات النصية القابلة للتصنيف (مثل الماركة، موقع، لون، ناقل حركة).
- "sensitive_name": لأسماء الأشخاص الحقيقية.
- "sensitive_email": للبريد الإلكتروني.
- "sensitive_phone": لأرقام الهواتف.
- "sensitive_address": للعناوين التفصيلية.
- "sensitive_id": لأرقام الهوية أو الحسابات.

أرجع النتيجة بصيغة JSON فقط، حيث المفتاح هو اسم العمود والقيمة هي النوع.
لا تكتب أي نص آخر غير كائن الـ JSON.
"""

    client = Groq(api_key='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')
    try:
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            response_format={'type': 'json_object'}
        )
        print("Response:")
        print(response.choices[0].message.content)
    except Exception as e:
        print("Error:")
        traceback.print_exc()

if __name__ == "__main__":
    test()
