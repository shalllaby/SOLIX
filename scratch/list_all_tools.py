import os

tools_path = r"e:\run-20260221T125607Z-1-001\run\backend\tools"
out_path = r"e:\run-20260221T125607Z-1-001\run\scratch\tools_structure.txt"

with open(out_path, "w", encoding="utf-8") as out:
    out.write("--- RECURSIVE TOOLS LIST ---\n")
    for root, dirs, files in os.walk(tools_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, tools_path)
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    out.write(f"Tool: {rel_path} | Lines: {len(lines)}\n")
                except Exception as e:
                    out.write(f"Tool: {rel_path} | Error: {e}\n")

print("Done listing all tools!")
