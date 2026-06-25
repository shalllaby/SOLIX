file_path = "backend/tools/synthetic_data/engine.py.bak"
with open(file_path, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# Let's search backwards from line 1140 (index 1139) to find the first def statement
for idx in range(1139, 0, -1):
    if "def " in lines[idx]:
        print(f"Found def at line {idx+1}: {lines[idx].strip()}")
        # Print 5 lines before and after
        for i in range(max(0, idx - 5), min(len(lines), idx + 10)):
            print(f"{i+1}: {lines[i].strip()}")
        break
