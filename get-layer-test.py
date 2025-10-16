import re
import os
import math
from pathlib import Path
import argparse
import pyLCIO
from pyLCIO import EVENT, UTIL
import ROOT, array
from array import array

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

def main():

    ops = options()
    in_file = ops.i
    print(f"Reading file {in_file}")

    stem = in_file.stem #gets the name without extension
    in_ind = stem.removeprefix("digiGNN_") #in_file.stem.split('_')[-1]
    out_dir = "/global/cfs/projectdirs/atlas/jashley/mjolnir/data/beta"
    out_file = f"{out_dir}/Hits_TTree_{in_ind}.root"
    tree_name = "HitTree"
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
            print(f"CellIDEncoding for {col_name}: {enc}")
            dec = UTIL.BitField64(enc)

            for i_hit, hit in enumerate(collection):
                if i_hit < ops.nhits:
                    break
                dec.setValue(hit.getCellID0() | (hit.getCellID1() << 32))
                print("Decoder:", dec)
                print("System (subdetector):", dec["system"].value())
                print("Layer:", dec["layer"].value())

if __name__ == "__main__":
    main()
