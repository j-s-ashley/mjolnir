import re
from collections import defaultdict

first_file  = "digi-no-bib.txt"
second_file = "digi-with-bib.txt"
num_events  = 10

# Define the 4 regions of interest
regions = [
    "EcalBarrelCollectionDigi",
    "EcalEndcapCollectionDigi",
    "HcalBarrelCollectionDigi",
    "HcalEndcapCollectionDigi"
]

# Structure: region -> { 'count': int, 'max_energy': float, 'max_pos': [x, y, z] }
results = defaultdict(lambda: {'count': 0, 'max_energy': 0.0, 'max_pos': [0.0, 0.0, 0.0]})

with open(second_file, "r") as f:
    for line in f:
        # Match hit count lines
        match_count = re.match(r"\s*(\d+)\s+hits in (\w+)", line)
        if match_count:
            hits = int(match_count.group(1))
            region = match_count.group(2)
            if region in regions:
                results[region]['count'] += hits
            continue

        # Match CalorimeterHit lines
        match_hit = re.match(
            r"CalorimeterHit \d+ in (\w+): Energy ([\d.]+) Position \(([^,]+), ([^,]+), ([^)]+)\)",
            line
        )
        if match_hit:
            region = match_hit.group(1)
            if region in regions:
                energy = float(match_hit.group(2))
                pos = list(map(lambda x: abs(float(x)), match_hit.group(3, 4, 5)))

                results[region]['max_energy'] = max(results[region]['max_energy'], energy)
                results[region]['max_pos'] = [
                    max(results[region]['max_pos'][i], pos[i]) for i in range(3)
                ]

# Print results
for region in regions:
    data = results[region]
    print(f"{region}:")
    print(f"  Total hits: {data['count']}")
    print(f"  Max energy: {data['max_energy']}")
    print(f"  Max |position|: x={data['max_pos'][0]}, y={data['max_pos'][1]}, z={data['max_pos'][2]}")

