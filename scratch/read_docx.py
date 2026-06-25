import zipfile
import xml.etree.ElementTree as ET
import os

def read_docx(file_path):
    if not os.path.exists(file_path):
        return f"File {file_path} not found."
    
    try:
        with zipfile.ZipFile(file_path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # XML Namespaces
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            text_runs = []
            for paragraph in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                p_text = []
                for run in paragraph.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                    if run.text:
                        p_text.append(run.text)
                text_runs.append("".join(p_text))
            
            return "\n".join(text_runs)
    except Exception as e:
        return f"Error reading {file_path}: {e}"

if __name__ == "__main__":
    backend_path = r"e:\run-20260221T125607Z-1-001\run\old doce\Backend Development.docx"
    db_path = r"e:\run-20260221T125607Z-1-001\run\old doce\Database Design.docx"
    
    backend_text = read_docx(backend_path)
    db_text = read_docx(db_path)
    
    os.makedirs(r"e:\run-20260221T125607Z-1-001\run\scratch", exist_ok=True)
    
    with open(r"e:\run-20260221T125607Z-1-001\run\scratch\backend_development.txt", "w", encoding="utf-8") as f:
        f.write(backend_text)
        
    with open(r"e:\run-20260221T125607Z-1-001\run\scratch\database_design.txt", "w", encoding="utf-8") as f:
        f.write(db_text)
        
    print("Done extracting docx content!")
