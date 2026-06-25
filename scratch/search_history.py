import os

search_paths = [
    r"C:\Users\Mohamed Shalaby\AppData\Roaming\Antigravity\User\History",
    r"C:\Users\Mohamed Shalaby\AppData\Local\History",
    r"C:\Users\Mohamed Shalaby\AppData\Roaming\Code\User\History",
    r"C:\Users\Mohamed Shalaby\AppData\Roaming\Cursor\User\History",
    r"C:\Users\Mohamed Shalaby\.gemini\antigravity"
]

print("Starting search...")
found = []
for base_path in search_paths:
    if not os.path.exists(base_path):
        continue
    print(f"Searching in: {base_path}")
    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            # check size to avoid reading massive files
            try:
                size = os.path.getsize(file_path)
                if size < 500000: # under 500KB
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if 'downloadPdfReport' in content and 'automl' in content.lower():
                            print(f"Found match: {file_path} (Size: {size})")
                            found.append((file_path, size))
            except Exception as e:
                pass

print("Search completed. Found:", len(found))
