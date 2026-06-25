import json
try:
    json.loads('{\n    "dcr_values": ')
except Exception as e:
    print("Error:", type(e), e)
