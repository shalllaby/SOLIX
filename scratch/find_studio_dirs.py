import os

found = []
for root, dirs, files in os.walk("app"):
    for f in files:
        found.append(os.path.join(root, f))
    for d in dirs:
        found.append(os.path.join(root, d))

print("Total items in app:", len(found))
for item in sorted(found)[:100]:
    print(item)
