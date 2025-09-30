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
    """
    Parse encoding string like "system:5,side:-2,layer:6,..." into list of (name,width).
    Negative widths are allowed in strings (sometimes used to indicate signed), we use abs(width).
    """
    parts = [p.strip() for p in enc_str.split(',') if p.strip()]
    fields = []
    for p in parts:
        m = re.match(r'^([^:]+)\s*:\s*(-?\d+)\s*$', p)
        if not m:
            raise ValueError(f"Can't parse token: {p!r}")
        name = m.group(1)
        width = abs(int(m.group(2)))
        fields.append((name, width))
    return fields

def decode_cellid(cellid, fields):
    """
    Decode integer cellid into dict by interpreting fields as MSB->LSB order.
    fields is list of (name,width).
    """
    total_bits = sum(w for _, w in fields)
    rem_bits = total_bits
    out = {}
    for name, width in fields:
        rem_bits -= width
        mask = (1 << width) - 1
        value = (cellid >> rem_bits) & mask
        out[name] = value
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
                info   = decode_cellid(cellid, fields)
                subdet = info.get('system')
                layer  = info.get('layer')

                print(f"{col_name} hit {i_hit} in layer {layer} of {subdet}")

def get_collection(event, name):
    names = event.getCollectionNames()
    if name in names:
        return event.getCollection(name)
    return []


if __name__ == "__main__":
    main()
