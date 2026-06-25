import os

search_dir = r"E:\run-20260221T125607Z-1-001\run"
target = "more operations"

for root, dirs, files in os.walk(search_dir):
    if ".git" in root or ".pytest_cache" in root or "__pycache__" in root:
        continue
    for file in files:
        if file.endswith((".py", ".js", ".html", ".txt")):
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if target in content.lower():
                    print(f"FOUND in: {path}")
                    # Print matching lines
                    lines = content.splitlines()
                    for i, line in enumerate(lines):
                        if target in line.lower():
                            print(f"  Line {i+1}: {line.strip()}")
            except Exception as e:
                pass
print("Search complete.")
