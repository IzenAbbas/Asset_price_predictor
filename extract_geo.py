import pandas as pd
import json

try:
    df = pd.read_csv('House_Details.csv')
    unique_rows = df[['province_name', 'city', 'location']].dropna().drop_duplicates()

    geo = {}
    for _, row in unique_rows.iterrows():
        p = row['province_name']
        c = row['city']
        l = row['location']
        if p not in geo:
            geo[p] = {}
        if c not in geo[p]:
            geo[p][c] = []
        geo[p][c].append(l)

    for p in geo:
        for c in geo[p]:
            geo[p][c].sort()

    with open('extracted_geo.json', 'w') as f:
        json.dump(geo, f, indent=4)
    print("Saved to extracted_geo.json")
    print(f"Provinces: {list(geo.keys())}")
except Exception as e:
    print(f"Error: {e}")
