import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('frontend/templates/app/automl_studio.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if '{% block' in line or '{% endblock' in line:
        print(f"Line {i+1}: {line.strip()}")
