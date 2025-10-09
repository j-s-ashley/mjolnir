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

def get_theta(x, y, z):
    r = (x*x + y*y) ** 0.5
    angle = math.atan2(r, z)
    return angle

def get_cluster_size(x_local, y_local):
    ymax = -1e6
    xmax = -1e6
    ymin = 1e6
    xmin = 1e6

    if y_local < ymin:
        ymin = y_local
    if y_local > ymax:
        ymax = y_local

    if x_local < xmin:
        xmin = x_local
    if x_local > xmax:
        xmax = x_local

    cluster_size_y = (ymax - ymin) + 1
    cluster_size_x = (xmax - xmin) + 1

    return cluster_size_x, cluster_size_y

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
    # Create a new ROOT file and TTree
    root_file = ROOT.TFile(out_file, "RECREATE")
    tree = ROOT.TTree(tree_name, "Tree storing hit information")
    
    # Define variables to be stored in the tree
    x = ROOT.std.vector('float')()
    y = ROOT.std.vector('float')()
    z = ROOT.std.vector('float')()
    t =	ROOT.std.vector('float')()
    e =	ROOT.std.vector('float')()
    theta =	ROOT.std.vector('float')()
    cluster_size_x   = ROOT.std.vector('float')()
    cluster_size_y   = ROOT.std.vector('float')()
    cluster_size_tot = ROOT.std.vector('float')()
    subdetector = ROOT.std.vector('int')()
    layer       = ROOT.std.vector('int')()

    # Create branches
    tree.Branch("Cluster_x", x)
    tree.Branch("Cluster_y", y)
    tree.Branch("Cluster_z", z)
    tree.Branch("Cluster_ArrivalTime", t)
    tree.Branch("Cluster_EnergyDeposited", e)
    tree.Branch("Incident_Angle", theta)
    tree.Branch("Cluster_Size_x", cluster_size_x)
    tree.Branch("Cluster_Size_y", cluster_size_y)
    tree.Branch("Cluster_Size_tot", cluster_size_tot)
    tree.Branch("Subdetector", subdetector)
    tree.Branch("Layer", layer)
    
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

            for i_hit, hit in enumerate(collection):
                if i_hit < ops.nhits:
                    break
                
                x.clear()
                y.clear()
                z.clear()
                t.clear()
                e.clear()
                theta.clear()
                cluster_size_x.clear()
                cluster_size_y.clear()
                cluster_size_tot.clear()
                subdetector.clear()
                layer.clear()

                hits = hit.getRawHits()

                position = hit.getPosition()
                x_pos = position[0]
                y_pos = position[1]
                z_pos = position[2]
                x.push_back(x_pos)
                y.push_back(y_pos)
                z.push_back(z_pos)
                t.push_back(hit.getTime())
                e.push_back(hit.getEDep())
                theta.push_back(get_theta(x_pos, y_pos, z_pos))
            
                cluster_x, cluster_y = get_cluster_size(x_pos, y_pos)
                cluster_size_x.push_back(cluster_x)
                cluster_size_y.push_back(cluster_y)
                cluster_size_tot.push_back(len(hits))

                cellid = hit.getCellID0()
                fields = parse_cellid_encoding(cellid_encoding)
                decoded = decode_cellid(cellid, fields)
                subdetector.push_back(decoded['system'])
                layer.push_back(decoded['layer'])

                tree.Fill()
                    
    # Write and close
    tree.Write()
    root_file.Close()

    print(f"ROOT file '{out_file}' with tree '{tree_name}' created.")

def get_collection(event, name):
    names = event.getCollectionNames()
    if name in names:
        return event.getCollection(name)
    return []


if __name__ == "__main__":
    main()
