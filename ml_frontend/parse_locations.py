import csv
import json
import os

geography = {}

# We need to map column indexes:
# 'location' = 6, 'city' = 7, 'province_name' = 8
# Wait, let's just use csv.DictReader
with open('../House_Details.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        prov = row.get('province_name', '').strip()
        city = row.get('city', '').strip()
        loc = row.get('location', '').strip()
        
        if not prov or not city or not loc:
            continue
            
        if prov not in geography:
            geography[prov] = {}
        if city not in geography[prov]:
            geography[prov][city] = set()
            
        geography[prov][city].add(loc)

for prov in geography:
    for city in geography[prov]:
        geography[prov][city] = sorted(list(geography[prov][city]))

with open('lib/constants/pakistan_locations.dart', 'w', encoding='utf-8') as f:
    f.write('const Map<String, Map<String, List<String>>> pakistanGeography = {\n')
    for prov in sorted(geography.keys()):
        prov_escaped = prov.replace("'", "\\'")
        f.write(f"  '{prov_escaped}': {{\n")
        
        for city in sorted(geography[prov].keys()):
            city_escaped = city.replace("'", "\\'")
            f.write(f"    '{city_escaped}': [\n")
            
            for loc in geography[prov][city]:
                loc_escaped = loc.replace("'", "\\'")
                f.write(f"      '{loc_escaped}',\n")
                
            f.write("    ],\n")
        f.write("  },\n")
    f.write("};\n")

print("Done parsing and writing to dart file.")
