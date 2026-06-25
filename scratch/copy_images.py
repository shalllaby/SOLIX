import os
import shutil

src_dir = r"e:\run-20260221T125607Z-1-001\run\صور التيم"
dest_dir = r"e:\run-20260221T125607Z-1-001\run\frontend\static\team"

os.makedirs(dest_dir, exist_ok=True)
for file_name in os.listdir(src_dir):
    src_path = os.path.join(src_dir, file_name)
    dest_path = os.path.join(dest_dir, file_name)
    if os.path.isfile(src_path):
        shutil.copy2(src_path, dest_path)
        print(f"Copied {file_name} to {dest_dir}")
