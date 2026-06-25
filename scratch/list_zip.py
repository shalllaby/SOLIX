import zipfile

zip_path = "frontend_project.zip"
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print("Total files in zip:", len(zip_ref.namelist()))
        for name in zip_ref.namelist()[:30]:
            print(name)
except Exception as e:
    print("Error:", e)
