import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ner_cars import (
    load_env_file,
    clean_pakwheels,
    normalize_output,
    extract_categorical_by_regex,
)

load_env_file()

raw_text = Path(str(Path(__file__).parent) + "/scraped_output.txt").read_text(encoding="utf-8")

cleaned = clean_pakwheels(raw_text)
print("=== CLEANED TEXT ===")
print(cleaned)
print(f"\n--- ({len(cleaned)} chars) ---\n")

regex_result = extract_categorical_by_regex(cleaned)
print("=== REGEX EXTRACTION ===")
print(json.dumps(regex_result, indent=2))

fallback = normalize_output({}, cleaned)
print("\n=== FALLBACK RESULT ===")
print(json.dumps(fallback, indent=2))

expected = {
    "model_year": 2004,
    "mileage_km": 160000,
    "engine_capacity_cc": 1500,
    "fuel_type": "petrol",
    "transmission": "automatic",
    "assembly": "local",
    "brand": "honda",
    "model_name": "civic",
}
print("\n=== EXPECTED ===")
print(json.dumps(expected, indent=2))

print("\n=== COMPARISON ===")
all_ok = True
for field, exp_val in expected.items():
    got = fallback.get(field)
    match = "✅" if got == exp_val else "❌"
    if got != exp_val:
        all_ok = False
    print(f"  {match} {field}: expected={exp_val}, got={got}")

if all_ok:
    print("\n🎉 All fields matched!")
else:
    print("\n⚠️  Some fields did not match")
