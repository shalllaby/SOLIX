import os

for root, dirs, files in os.walk('.'):
    for file in files:
        if 'automl_studio.html' in file:
            print(os.path.join(root, file))
