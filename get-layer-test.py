import re
import os
import math
from pathlib import Path
import argparse
import pyLCIO
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

def parse_cellid_encoding(enc_str):
    """Parse an encoding string like 'system:5,side:-2,layer:6,module:11,sensor:8'"""
    fields = []
    for part in enc_str.split(','):
        name, width = part.split(':')
        fields.append((name.strip(), abs(int(width))))
    return fields

def decode_cellid(cellid, fields):
    """
    Decode integer cellid into dict of field values (LSB-first).
    fields: list of (name,width), in order given by CellIDEncoding string.
    """
    out = {}
    shift = 0
    for name, width in fields:
        mask = (1 << width) - 1
        value = (cellid >> shift) & mask
        out[name] = value
        shift += width
    return out

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
            cellid_encoding = collection.getParameters().getStringVal("CellIDEncoding")
            fields = parse_cellid_encoding(cellid_encoding)
            print(f"CellIDEncoding for {col_name}: {cellid_encoding}")

            for i_hit, hit in enumerate(collection):
                if i_hit < ops.nhits:
                    break

                cellid = hit.getCellID0()
                decoded = decode_cellid(cellid, fields)
                print("Decoded fields:", decoded)
                print("System (subdetector):", decoded['system'])
                print("Layer:", decoded['layer'])

def get_collection(event, name):
    names = event.getCollectionNames()
    if name in names:
        return event.getCollection(name)
    return []


if __name__ == "__main__":
    main()
