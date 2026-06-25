import sys
sys.path.append("E:/run-20260221T125607Z-1-001/run")
from backend.auth import create_access_token
token = create_access_token({'sub': 'test@test.com'})
print("TOKEN:", token)
