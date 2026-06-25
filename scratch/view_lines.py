import sys

with open('backend/tools/synthetic_data/engine.py', 'rb') as f:
    content = f.read()

# Try to decode with utf-8, replace errors to find where the bad bytes are
try:
    decoded = content.decode('utf-8')
    print("Decoded successfully with UTF-8")
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}")

# Let's print lines 1200 to 1230 by decoding line-by-line with replacement
lines = content.split(b'\n')
print(f"Total lines: {len(lines)}")
start_line = 1200
end_line = 1235
for idx in range(start_line - 1, min(end_line, len(lines))):
    line_bytes = lines[idx]
    try:
        line_str = line_bytes.decode('utf-8')
    except UnicodeDecodeError:
        line_str = line_bytes.decode('utf-8', errors='replace')
        print(f"WARNING: Line {idx+1} has invalid UTF-8 bytes: {line_bytes}")
    print(f"{idx+1}: {line_str.rstrip()}")
