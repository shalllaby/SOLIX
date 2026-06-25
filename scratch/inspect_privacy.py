file_path = "Genreted data safaa/Genreted data safaa/synthetic .py"
with open(file_path, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

output_path = "scratch/inspect_privacy_safaa.txt"
with open(output_path, "w", encoding="utf-8") as f:
    start = 980
    end = min(1030, len(lines))
    for idx in range(start, end):
        f.write(f"{idx+1}: {lines[idx]}")
