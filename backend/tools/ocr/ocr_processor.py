import os
import concurrent.futures
import time
import io
import pandas as pd
from functools import partial

try:
    import pytesseract
    from PIL import Image
    import fitz  # PyMuPDF لا تحتاج لأداة Poppler أو برامج خارجية
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def _ocr_single_page_worker(page_info):
    """دالة مساعدة تعمل في عملية منفصلة لمعالجة صفحة واحدة بالتوازي."""
    page_index, pdf_bytes, lang, tesseract_cmd, tessdata_prefix = page_info
    
    # إعداد Tesseract داخل العملية الجديدة
    import pytesseract
    from PIL import Image
    import fitz
    import os
    import io

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    if tessdata_prefix:
        os.environ["TESSDATA_PREFIX"] = tessdata_prefix

    try:
        # فتح المستند داخل العملية لمعالجة الصفحة المطلوبة
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(page_index)
        
        # تحويل الصفحة لصورة بجودة عالية
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # تنفيذ الـ OCR واستخراج طبقة الـ PDF
        pdf_page_bytes = pytesseract.image_to_pdf_or_hocr(img, extension='pdf', lang=lang)
        doc.close()
        return page_index, pdf_page_bytes
    except Exception as e:
        return page_index, None


class OCRProcessor:
    """
    مُعالج استخراج النصوص (OCR) من الصور والملفات.
    يدعم الصور وتنسيقات PDF واللغتين العربية والإنجليزية.
    """

    def __init__(self, tesseract_cmd=None):
        """
        تهيئة المعالج وتحديد مسار Tesseract بصورة ذكية.
        """
        # 1. Fallback stack: User param -> ENV Var -> Standard Windows paths -> Custom path
        best_cmd = (
            tesseract_cmd or 
            os.getenv("TESSERACT_PATH") or 
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
        if not os.path.exists(best_cmd):
             best_cmd = r"C:\مشروع التخرج\tesseract.exe"
             
        tesseract_cmd = best_cmd
        
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            # معالجة مشكلة المسارات العربية
            import ctypes
            try:
                buf = ctypes.create_unicode_buffer(260)
                ctypes.windll.kernel32.GetShortPathNameW(tesseract_cmd, buf, 260)
                tesseract_cmd = buf.value or tesseract_cmd
            except Exception:
                pass
                
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            # تحديد مسار ملفات اللغة تلقائياً حتى لو تم تثبيته في مسار مختلف
            os.environ["TESSDATA_PREFIX"] = os.path.join(os.path.dirname(tesseract_cmd), "tessdata")

    # ==========================
    # 1. دوال استخراج النصوص
    # ==========================

    def extract_text_from_image(self, image_bytes, lang='eng+ara'):
        """استخراج النص من صورة (Bytes)."""
        if not OCR_AVAILABLE:
            return "OCR dependencies (PyMuPDF or Tesseract) are not installed on the server."
        try:
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image, lang=lang)
            return text.strip()
        except Exception as e:
            return f"Error extracting text from image: {e}"

    def extract_text_from_pdf(self, pdf_bytes, lang='eng+ara'):
        """تحويل الـ PDF لصور واستخراج النص من كل الصفحات (بدون استخدام Poppler)."""
        if not OCR_AVAILABLE:
            return "OCR dependencies (PyMuPDF or Tesseract) are not installed on the server."
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = []
            for i in range(len(doc)):
                page = doc.load_page(i)
                # تحويل صفحة الـ PDF إلى صورة بدقة مضاعفة (لتحسين دقة استخراج النص)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                
                # تحويل الصورة إلى تنسيق يتعرف عليه Tesseract
                mode = "RGBA" if pix.alpha else "RGB"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                
                page_text = pytesseract.image_to_string(img, lang=lang)
                full_text.append(f"--- Page {i+1} ---\n{page_text.strip()}")
            return "\n\n".join(full_text)
        except Exception as e:
            return f"Error extracting text from PDF: {e}."

    def create_searchable_pdf(self, pdf_bytes, output_path, lang='eng+ara'):
        """النسخة الاحترافية: تحويل الـ PDF لنسخة قابلة للبحث."""
        if not OCR_AVAILABLE:
            return "OCR dependencies (PyMuPDF or Tesseract) are not installed on the server."
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            num_pages = len(doc)
            
            tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
            tessdata_prefix = os.environ.get("TESSDATA_PREFIX", "")

            print(f"[OCR] Processing {num_pages} pages...")
            
            final_pdf = fitz.open()
            
            for i in range(num_pages):
                page = doc.load_page(i)
                # تحويل الصفحة لصورة بجودة عالية
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # تنفيذ الـ OCR واستخراج طبقة الـ PDF
                pdf_page_bytes = pytesseract.image_to_pdf_or_hocr(img, extension='pdf', lang=lang)
                
                if pdf_page_bytes:
                    page_doc = fitz.open(stream=pdf_page_bytes, filetype="pdf")
                    final_pdf.insert_pdf(page_doc)
                    page_doc.close()
            
            doc.close()
            final_pdf.save(output_path)
            final_pdf.close()
            return True
        except Exception as e:
            return f"Error creating searchable PDF: {e}"

    def process_file(self, file_bytes, filename, lang='eng+ara'):
        """دالة للتعامل مع أي ملف واختيار الطريقة المناسبة بناءً على النوع."""
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_bytes, lang=lang)
        else:
            return self.extract_text_from_image(file_bytes, lang=lang)

    # ==========================
    # 2. دوال معالجة البيانات
    # ==========================

    def text_to_dataframe(self, text, separator=None):
        """تحويل النص المستخرج (لو كان جدولاً) إلى DataFrame."""
        try:
            # تنظيف السطور
            lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
            if not lines:
                return pd.DataFrame()

            # تحديد الفاصل تلقائياً في حال لم يتم تحديده
            if not separator:
                first_line = lines[0]
                if '\t' in first_line: separator = '\t'
                elif '|' in first_line: separator = '|'
                elif ',' in first_line: separator = ','
                else: separator = ' '

            data = [line.split(separator) for line in lines]
            
            if len(data) > 1:
                header = data[0]
                clean_data = []
                for row in data[1:]:
                    # موازنة الأعمدة مع الرأس (Header)
                    if len(row) < len(header):
                        row.extend([''] * (len(header) - len(row)))
                    else:
                        row = row[:len(header)]
                    clean_data.append(row)
                df = pd.DataFrame(clean_data, columns=header)
            else:
                df = pd.DataFrame(data)
            return df
        except Exception as e:
            return pd.DataFrame([{"Error": str(e), "Raw_Text": text}])

# مثال على الاستخدام (Example Usage)
if __name__ == "__main__":
    # تهيئة المعالج
    ocr = OCRProcessor()
    print("OCR Processor is ready.")