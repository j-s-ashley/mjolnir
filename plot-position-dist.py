import math
from pathlib import Path
import statistics
import argparse
import pyLCIO
from pyLCIO import EVENT, UTIL
import numpy as np
import ROOT
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

    for i_event, event in enumerate(reader):
        cols = {}
        for col in COLLECTIONS:
            cols[col] = get_collection(event, col)

        print(f"Event {i_event} has")
        for col in cols:
            print(f"  {len(cols[col]):5} hits in {col}")

        for col_name in COLLECTIONS:
            collection = cols[col_name]
            enc = collection.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
            dec   = UTIL.BitField64(enc)
            r     = []
            r_dif = []
            z     = []
            z_dif = []

            for i_hit, hit in enumerate(collection):
                if i_hit < ops.nhits:
                    break
                dec.setValue(hit.getCellID0() | (hit.getCellID1() << 32))
                position = hit.getPosition()
                if "Endcap" in col_name:
                    z_pos = position[2]
                    side_val = dec["side"].value()
                    if i_hit != 0:
                        d_z = z_pos - z[i_hit-1]
                        z_dif.append(d_z)
                    z.append(z_pos * side_val)
                else:
                    r_pos = get_r(position[0], position[1])
                    if i_hit != 0:
                        d_r = r_pos - r[i_hit-1]
                        r_dif.append(d_r)
                    r.append(r_pos)

            if "Endcap" in col_name:
                z_min = min(z)
                z_max = max(z)
                print(col_name)
                print(f"           minimum z: {z_min}")
                print(f"           maximum z: {z_max}")
                print(f"          average dz: {z_max}")
                print(f"          z variance: {statistics.variance(z_dif)}")
                bins = np.arange(z_min-1, z_max+1, 1)
                plt.hist(z, bins=bins)
                plt.xlabel("z")
                plt.ylabel("hits")
            else:
                r_min = min(r)
                r_max = max(r)
                print(f"           minimum r: {r_min}")
                print(f"           maximum r: {r_max}")
                print(f"          average dr: {r_max}")
                print(f"          r variance: {statistics.variance(r_dif)}")
                bins = np.arange(r_min-1, r_max+1, 1)
                plt.hist(r, bins=bins)
                plt.xlabel("r")
                plt.ylabel("hits")

            plt.grid()
            plt.title(f"{col_name}-position-dist.pdf")
            plt.savefig(f"{col_name}-layer-vs-position.pdf")
            plt.clf()

if __name__ == "__main__":
    main()
