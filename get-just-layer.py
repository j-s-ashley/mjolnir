import re
import os
import math
from pathlib import Path
import argparse
import pyLCIO
from pyLCIO import EVENT, UTIL
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
            print(f"CellIDEncoding for {col_name}: {enc}")
            dec = UTIL.BitField64(enc)
            layer = []

            for i_hit, hit in enumerate(collection):
                if i_hit < ops.nhits:
                    break
                dec.setValue(hit.getCellID0() | (hit.getCellID1() << 32))
                layer_val = dec["layer"].value()
                side_val  = dec["side"].value()
                #layer.append(layer_val)
                print(f"{col_name} hit {i_hit} layer value: {layer_val}, side value {side_val}")

            #layer_min = min(layer)
            #layer_max = max(layer)
            #print(f"{col_name} minimum layer: {min(layer)}")
            #print(f"           maximum layer: {max(layer)}")

if __name__ == "__main__":
    main()
