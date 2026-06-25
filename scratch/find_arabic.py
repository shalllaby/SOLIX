import os, re

arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
files_to_check = [
    'frontend/templates/app/_layout.html',
    'frontend/templates/app/dashboard.html',
    'frontend/templates/app/chat_with_data.html',
    'frontend/templates/app/cleaning_studio.html',
    'frontend/templates/app/synthetic_studio.html',
    'frontend/templates/app/viz_report.html',
    'frontend/templates/app/automl_studio.html'
]

os.makedirs('scratch', exist_ok=True)
with open('scratch/arabic_lines.txt', 'w', encoding='utf-8') as out:
    for path in files_to_check:
        out.write(f'=== {path} ===\n')
        if not os.path.exists(path):
            out.write('FILE NOT FOUND\n')
            continue
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for idx, line in enumerate(f, 1):
                if arabic_pattern.search(line):
                    out.write(f'{idx}: {line.strip()[:150]}\n')
print("Successfully scanned and wrote to scratch/arabic_lines.txt")
