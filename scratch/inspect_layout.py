import os

layout_path = 'frontend/templates/app/_layout.html'
if os.path.exists(layout_path):
    with open(layout_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if 'href="/app/' in line or "href='/app/" in line:
                print(f"{i+1}: {line.strip()}")
else:
    print("Layout not found")
