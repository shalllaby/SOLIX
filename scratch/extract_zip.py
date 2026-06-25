import zipfile
import os

zip_path = 'frontend_project.zip'
if os.path.exists(zip_path):
    print("ZIP file exists!")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        namelist = zip_ref.namelist()
        matching = [name for name in namelist if 'automl_studio.html' in name]
        print("Matching in ZIP:", matching)
        if matching:
            content = zip_ref.read(matching[0]).decode('utf-8')
            print("Length of content in ZIP:", len(content))
            # print last 30 lines
            lines = content.splitlines()
            print("Last 30 lines:")
            for line in lines[-30:]:
                print(line)
else:
    print("ZIP file does NOT exist!")
