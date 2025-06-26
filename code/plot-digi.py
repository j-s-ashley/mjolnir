import re
from collections import defaultdict
import matplotlib.pyplot as plt

filename = "digi-with-bib.txt"
regions  = [
    "EcalBarrelCollectionDigi",
    "EcalEndcapCollectionDigi",
    "HcalBarrelCollectionDigi",
    "HcalEndcapCollectionDigi"
]

# Store energies and positions per region
data = defaultdict(lambda: {'energies': [], 'positions': [[], [], []]})  # x, y, z

with open(filename, "r") as f:
    for line in f:
        match_hit = re.match(
            r"CalorimeterHit \d+ in (\w+): Energy ([\d.]+) Position \(([^,]+), ([^,]+), ([^)]+)\)",
            line
        )
        if match_hit:
            region = match_hit.group(1)
            if region in regions:
                energy = float(match_hit.group(2))
                x, y, z = map(float, match_hit.group(3, 4, 5))
                data[region]['energies'].append(energy)
                data[region]['positions'][0].append(x)
                data[region]['positions'][1].append(y)
                data[region]['positions'][2].append(z)

fig, axes = plt.subplots(len(regions), 4, figsize=(20, 4 * len(regions)))

for i, region in enumerate(regions):
    reg_data = data[region]

    # Energy histogram
    axes[i, 0].hist(reg_data['energies'], bins=30, color='orange')
    axes[i, 0].set_title(f"{region} - Energy")

    # X, Y, Z position histograms
    for j, axis in enumerate(['X', 'Y', 'Z']):
        axes[i, j + 1].hist(reg_data['positions'][j], bins=30)
        axes[i, j + 1].set_title(f"{region} - {axis} Position")

plt.tight_layout()
plt.savefig("digi-with-bib-dist.pdf")
