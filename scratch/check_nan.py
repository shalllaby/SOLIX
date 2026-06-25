import json
for val in ["NaN", "nan", "null", "undefined", "", "None"]:
    try:
        res = json.loads(f'{{\n  "privacy_score": {val}\n}}')
        print(f"Succeeded for {val}: {res}")
    except Exception as e:
        print(f"Failed for {val}: {type(e)} {e}")
