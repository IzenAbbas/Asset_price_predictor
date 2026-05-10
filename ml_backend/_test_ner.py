"""Test NER extraction endpoints."""
import requests
import json

BASE = "http://127.0.0.1:8001"

# Test vehicle NER extraction
print("Testing vehicle extraction...")
try:
    r = requests.post(
        f"{BASE}/extract/vehicle-fields",
        json={"url": "https://www.pakwheels.com/used-cars/toyota-corolla-2021-for-sale-in-multan-11468975"},
        timeout=60,
    )
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  [OK] Vehicle fields extracted: brand={data.get('brand')}, model={data.get('model_name')}")
    else:
        print(f"  [FAIL] Response: {r.text[:500]}")
except Exception as e:
    print(f"  [ERROR] {e}")

# Test house NER extraction
print("\nTesting house extraction...")
try:
    r = requests.post(
        f"{BASE}/extract/house-fields",
        json={"url": "https://www.zameen.com/Property/dha_defence_dha_phase_6_7_marla_house_for_sale_in_dha_phase_6_lahore-54192402-1448-1.html"},
        timeout=60,
    )
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  [OK] House fields extracted: city={data.get('city')}, type={data.get('property_type')}")
    else:
        print(f"  [FAIL] Response: {r.text[:500]}")
except Exception as e:
    print(f"  [ERROR] {e}")

print("\nDone!")
