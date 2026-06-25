file_path = "backend/tools/synthetic_data/engine.py"
with open(file_path, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

print(f"suggest_schema in file: {'suggest_schema' in content}")
print(f"suggest_schema count: {content.count('suggest_schema')}")
