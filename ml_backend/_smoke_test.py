"""Test all API endpoints against the running server."""
import requests
import json
import sys

BASE = "http://127.0.0.1:8001"

# 1. Root
r = requests.get(f"{BASE}/")
assert r.status_code == 200, f"Root failed: {r.status_code}"
print("[OK] GET / -- status", r.status_code)

# 2. Health
r = requests.get(f"{BASE}/health")
data = r.json()
assert data["car_model_loaded"] is True
assert data["house_model_loaded"] is True
print("[OK] GET /health -- both models loaded")

# 3. Options
r = requests.get(f"{BASE}/options")
opts = r.json()
assert len(opts["car"]["brands"]) > 0
assert len(opts["house"]["cities"]) > 0
print(f"[OK] GET /options -- {len(opts['car']['brands'])} car brands, {len(opts['house']['cities'])} house cities")

# 4. Car prediction
r = requests.post(f"{BASE}/predict/car", json={
    "mileage": 50000, "engine_capacity": 1800, "model_year": 2020,
    "fuel_type": "petrol", "transmission": "automatic", "assembly": "local",
    "brand": "toyota", "model_name": "corolla altis"
})
assert r.status_code == 200, f"Car predict failed: {r.text}"
print(f"[OK] POST /predict/car -- {r.json()['formatted_price']}")

# 5. House prediction
r = requests.post(f"{BASE}/predict/house", json={
    "Total_Area": 10, "bedrooms": 3, "baths": 2,
    "latitude": 31.5, "longitude": 74.3,
    "listing_year": 2024, "listing_month": 6,
    "property_type": "house", "location": "dha phase 5",
    "city": "lahore", "province_name": "punjab", "purpose": "for sale"
})
assert r.status_code == 200, f"House predict failed: {r.text}"
print(f"[OK] POST /predict/house -- {r.json()['formatted_price']}")

print("\nAll 5 endpoints verified -- backend is working correctly!")
