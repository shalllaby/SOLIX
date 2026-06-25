import shutil

src = r"C:\Users\Mohamed Shalaby\AppData\Roaming\Antigravity\User\History\27b37031\inXH.html"
dst = r"frontend/templates/app/automl_studio.html"

try:
    shutil.copy(src, dst)
    print("SUCCESS: Copied backup to frontend/templates/app/automl_studio.html")
except Exception as e:
    print("ERROR:", e)
