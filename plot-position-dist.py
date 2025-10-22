import math
from pathlib import Path
from collections import defaultdict
import statistics
import numpy as np
import ROOT
import argparse
import pyLCIO
from pyLCIO import EVENT, UTIL
import matplotlib.pyplot as plt

COLLECTIONS = [
        "ITBarrelHits",
        "ITEndcapHits",
        "OTBarrelHits",
        "OTEndcapHits",
        "VXDBarrelHits",
        "VXDEndcapHits"
        ]

def options():
    parser = argparse.ArgumentParser(description="Generate BIB output hits TTree root file from input slcio file.")
    parser.add_argument("-i", required=True, type=Path, help="Input LCIO file")
    parser.add_argument(
        "--nhits",
        default=0,
        type=int,
        help="Max number of hits to dump for each collection",
    )
    return parser.parse_args()

def get_collection(event, name):
    names = event.getCollectionNames()
    if name in names:
        return event.getCollection(name)
    return []

def get_r(x, y):
    r = (x*x + y*y) ** 0.5
    return r

def main():
    ops = options()
    in_file = ops.i
    print(f"Reading file {in_file}")

    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(str(in_file))

    data = defaultdict(list) # Dynamic list for positional information per hit collection

    # Start of event loop
    for i_event, event in enumerate(reader):
        cols = {} # Temporary dictionary storage for collections
        
        # Within each event, get hit collections
        for col in COLLECTIONS:
            cols[col] = get_collection(event, col)

        # Start of hit collection loop
        for col_name in COLLECTIONS:
            collection = cols[col_name]
            enc = collection.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding) # Get CellID info
            dec = UTIL.BitField64(enc) # Create 64-bit CellID decoder

            # Start loop over each hit in collection
            for i_hit, hit in enumerate(collection):
                # Limit hit number based on input at runtime
                if i_hit < ops.nhits:
                    break
                dec.setValue(hit.getCellID0() | (hit.getCellID1() << 32)) # 32-bit to 64-bit, then shift
                position = hit.getPosition() # Structure: position[x, y, z]
                # Get z-values from endcaps, r-values from barrel
                if "Endcap" in col_name:
                    z_pos = position[2]
                    side_val = dec["side"].value()
                    if i_hit != 0:
                        d_z = z_pos - data[col_name][i_hit-1]
                        data[f"{col_name}-d_z"].append(d_z)
                    data[col_name].append(z_pos * side_val)
                else:
                    r_pos = get_r(position[0], position[1])
                    if i_hit != 0:
                        d_r = r_pos - data[col_name][i_hit-1]
                        data[f"{col_name}-d_r"].append(d_r)
                    data[col_name].append(r_pos)
                    # End of loop over each hit
    # End of event loop

    # Pull and plot aggregated data per subdetector
    for col, vals in data.items():
        print(col)
        if "Endcap" in col:
            if "d_z" in col:
                avg_dz   = statistics.mean(vals)
                sigma_dz = statistics.variance(vals)
                print(f"{col} average distance between hits: {avg_dz}")
                print(f"{col} variance in distance between hits: {sigma_dz}")
            else:
                z_min = min(vals)
                z_max = max(vals)
                print(f"{col} minimum z: {z_min}")
                print(f"{col} maximum z: {z_max}")
                bins = np.arange(z_min-1, z_max+1, 1)
                plt.hist(vals, bins=bins)
                plt.xlabel("z")
                plt.ylabel("hits")
        else:
            if "d_r" in col:
                avg_dr   = statistics.mean(vals)
                sigma_dr = statistics.variance(vals)
                print(f"{col} average distance between hits: {avg_dr}")
                print(f"{col} variance in distance between hits: {sigma_dr}")
            else:
                r_min = min(vals)
                r_max = max(vals)
                print(f"{col} minimum r: {r_min}")
                print(f"{col} maximum r: {r_max}")
                bins = np.arange(r_min-1, r_max+1, 1)
                plt.hist(vals, bins=bins)
                plt.xlabel("r")
                plt.ylabel("hits")
        plt.grid()
        plt.title(f"{col}-position-dist-signal")
        plt.savefig(f"{col}-position-dist-signal.pdf")
        plt.clf()

if __name__ == "__main__":
    main()
