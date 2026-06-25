import sys
import os

# Add parent directory to path so we can import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from app import correct_query
    print("SUCCESS: Successfully imported 'correct_query' from app.py!")
except Exception as e:
    print(f"FAILED to import from app.py: {e}")
    sys.exit(1)

test_cases = [
    "مين محد شلبى",
    "دكتوره سيمو",
    "طريقه تثبي سولكس"
]

print("\nRunning Verification on Imported Function...\n")
for case in test_cases:
    try:
        corrected = correct_query(case)
        print(f"Input:     '{case}'")
        print(f"Corrected: '{corrected}'")
        print("-" * 30)
    except Exception as e:
        print(f"Error correcting '{case}': {e}")
