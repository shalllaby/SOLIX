file_path = "backend/tools/synthetic_data/engine.py"
with open(file_path, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

output_path = "scratch/inspect_output.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Total lines: {len(lines)}\n")
    start = 1150
    end = 1350
    for idx in range(start, min(end, len(lines))):
        f.write(f"{idx+1}: {lines[idx]}")
