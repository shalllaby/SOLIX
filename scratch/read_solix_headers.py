import os

path = r"e:\run-20260221T125607Z-1-001\run\SOLIX_DOCUMENTATION.md"
out_path = r"e:\run-20260221T125607Z-1-001\run\scratch\solix_doc_headers.txt"

with open(out_path, "w", encoding="utf-8") as out:
    if os.path.exists(path):
        out.write("--- SOLIX_DOCUMENTATION.md Headers ---\n")
        try:
            # Let's read with errors='replace' to avoid codec errors
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            headers = [line.strip() for line in lines if line.strip().startswith("#")]
            out.write(f"Headers ({len(headers)}):\n")
            for h in headers:
                out.write(f"  {h}\n")
        except Exception as e:
            out.write(f"Error: {e}\n")
    else:
        out.write("File does not exist.\n")

print("Done scanning SOLIX_DOCUMENTATION.md!")
