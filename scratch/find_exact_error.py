import json

candidates = ["nan", "undefined", "None", "", "N/A", "null"]
keys = ["privacy_score", "fidelity_score", "error", "status", "message"]

for k in keys:
    for val in candidates:
        for spaces in range(10):
            indent = " " * spaces
            # Try with and without newline and different styling
            text = f"{{\n{indent}\"{k}\": {val}\n}}"
            try:
                json.loads(text)
            except json.JSONDecodeError as e:
                err_str = str(e)
                if "line 2 column 19 (char 20)" in err_str:
                    print(f"MATCH! key: '{k}', val: '{val}', spaces: {spaces}, text:")
                    print(repr(text))
                    print(err_str)
                    print("-" * 40)
