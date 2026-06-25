import os
import sys

docs = [
    "README.md",
    "SOLIX_DOCUMENTATION.md",
    "PROJECT_ARCHITECTURE_DEEP_DIVE.md",
    "DATASET_ADVISOR.md",
    "FRONTEND_DOCUMENTATION.md",
    "RAG_REFERENCE.md",
    "BACKEND.md",
    "SYNTHETIC_STUDIO_GUIDE.md",
    "WORKFLOW.md"
]

out_path = r"e:\run-20260221T125607Z-1-001\run\scratch\doc_headers.txt"

with open(out_path, "w", encoding="utf-8") as out:
    out.write("--- SCANNING DOCUMENTATION FILES ---\n")
    for doc in docs:
        path = os.path.join(r"e:\run-20260221T125607Z-1-001\run", doc)
        if os.path.exists(path):
            size = os.path.getsize(path)
            out.write(f"File: {doc} | Size: {size} bytes\n")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                headers = [line.strip() for line in lines if line.strip().startswith("#")]
                out.write(f"  Headers ({len(headers)}):\n")
                for h in headers:
                    out.write(f"    {h}\n")
            except Exception as e:
                out.write(f"  Error reading headers: {e}\n")
        else:
            out.write(f"File {doc} does not exist.\n")
        out.write("-" * 50 + "\n")

print("Done writing headers report!")
