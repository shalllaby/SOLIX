import os
import re

templates_dir = r"e:\run-20260221T125607Z-1-001\run\frontend\templates\app"
files = [f for f in os.listdir(templates_dir) if f.endswith(".html")]

print(f"{'File Name':<30} | {'Locale == count':<15} | {'Has Arabic Characters':<25}")
print("-" * 80)

arabic_pattern = re.compile(r"[\u0600-\u06ff]")

for filename in sorted(files):
    path = os.path.join(templates_dir, filename)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    locale_count = content.count("locale ==") + content.count("locale == 'ar'") + content.count('locale == "ar"') + content.count("locale== 'ar'") + content.count("locale=='ar'")
    # Let's do a regex check for locale == (any spacing/quotes)
    locale_regex_count = len(re.findall(r"locale\s*==\s*['\"]ar['\"]", content))
    
    # Also look for any general "locale" reference (to catch getLocale() or variables)
    locale_total_count = len(re.findall(r"\blocale\b", content))
    
    has_arabic = bool(arabic_pattern.search(content))
    
    # Check if the file has Arabic characters outside comments
    # (Just a simple heuristic: count of Arabic characters)
    arabic_char_count = len(arabic_pattern.findall(content))
    
    print(f"{filename:<30} | {locale_regex_count:<15} | {has_arabic} ({arabic_char_count} chars)")
