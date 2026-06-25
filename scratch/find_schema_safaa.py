import os

file_path = "Genreted data safaa/Genreted data safaa/synthetic .py"
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    print(f"suggest_schema_from_prompt in file: {'suggest_schema_from_prompt' in content}")
    print(f"suggest_schema_from_prompt count: {content.count('suggest_schema_from_prompt')}")
    
    # Let's find all function definitions starting with def
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if "def " in line:
            print(f"{idx+1}: {line.strip()}")
else:
    print("File not found")
