import os

path = 'backend/tools/synthetic_data/engine.py'
bak_path = 'backend/tools/synthetic_data/engine.py.bak'
out_path = 'scratch/output_inspect.txt'

res = []

if os.path.exists(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    res.append(f"engine.py total lines: {len(lines)}")
    res.append(f"Line 1115-1125 of engine.py:")
    for i in range(1114, min(1125, len(lines))):
        res.append(f"  {i+1}: {lines[i].rstrip()}")
    res.append(f"Line 1210-1225 of engine.py:")
    for i in range(1209, min(1225, len(lines))):
        res.append(f"  {i+1}: {lines[i].rstrip()}")
else:
    res.append("engine.py not found")

if os.path.exists(bak_path):
    with open(bak_path, 'r', encoding='utf-8', errors='ignore') as f:
        bak_lines = f.readlines()
    res.append(f"bak total lines: {len(bak_lines)}")
    res.append(f"Line 1115-1125 of bak:")
    for i in range(1114, min(1125, len(bak_lines))):
        res.append(f"  {i+1}: {bak_lines[i].rstrip()}")
    res.append(f"Line 1210-1225 of bak:")
    for i in range(1209, min(1225, len(bak_lines))):
        res.append(f"  {i+1}: {bak_lines[i].rstrip()}")
else:
    res.append("bak not found")

with open(out_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(res))
