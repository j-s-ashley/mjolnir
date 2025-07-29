import os
import math
from pathlib import Path
import argparse
import pyLCIO
import ROOT, array

TRKHIT_COLLECTIONS = [
        "ITBarrelHits",
        "ITEndcapHits",
        "OTBarrelHits",
        "OTEndcapHits",
        "VXDBarrelHits",
        "VXDEndcapHits"
        ]

SIMTRKHIT_COLLECTIONS = [
        "InnerTrackerBarrelCollection_HTF",
        "InnerTrackerEndcapCollection_HTF",
        "OuterTrackerBarrelCollection_HTF",
        "OuterTrackerEndcapCollection_HTF",
        "VertexBarrelCollection_HTF",
        "VertexEndcapCollection_HTF"
        ]

REL_COLLECTIONS = [
        "ITBarrelHitsRelations_HTF",
        "ITEndcapHitsRelations_HTF",
        "OTBarrelHitsRelations_HTF",
        "OTEndcapHitsRelations_HTF",
        "VXDBarrelHitsRelations_HTF",
        "VXDEndcapHitsRelations_HTF"
        ]

PIXEL_COLLECTIONS = [
        "IBPixels_HTF",
        "IEPixels_HTF",
        "OBPixels_HTF",
        "OEPixels_HTF",
        "VBPixels_HTF",
        "VEPixels_HTF"
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
    subdetector = ROOT.std.vector('float')()
    layer       = ROOT.std.vector('float')()
    
    # Create branches
    tree.Branch("Hit_x", x)
    tree.Branch("Hit_y", y)
    tree.Branch("Hit_z", z)
    tree.Branch("Hit_ArrivalTime", t)
    tree.Branch("Hit_EnergyDeposited", e)
    tree.Branch("Incident_Angle", theta)
    tree.Branch("Cluster_Size_x", cluster_size_x)
    tree.Branch("Cluster_Size_y", cluster_size_y)
    tree.Branch("Cluster_Size_tot", cluster_size_tot)
    tree.Branch("Subdetector", subdetector)
    tree.Branch("Layer", layer)
    
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(str(in_file))

    for i_event, event in enumerate(reader):
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

        rel_cols = {}
        for rel_col in REL_COLLECTIONS:
            rel_cols[rel_col] = get_collection(event, rel_col)

        print(f"Event {i_event} has")
        for rel_col in rel_cols:
            print(f"  {len(rel_cols[rel_col]):5} hits in {rel_col}")

        for rel_col_name in REL_COLLECTIONS:
            collection = rel_cols[rel_col_name]
            cell_encoding = collection.getParameters().getStringVal("CellIDEncoding")
            print(f"Cell encoding for {rel_col_name}: {cell_encoding}")
            print(f"Type: {type(cell_encoding)}")
            #decoder = pyLCIO.UTIL.CellIDDecoder(collection)
            for i_hit, hit in enumerate(collection):
                if i_hit < ops.nhits:
                    break

                digi_hit, sim_hit = hit.getFrom(), hit.getTo()
                position = digi_hit.getPosition()
                x_pos = position[0]
                y_pos = position[1]
                z_pos = position[2]
                x.push_back(x_pos)
                y.push_back(y_pos)
                z.push_back(z_pos)
                t.push_back(digi_hit.getTime())
                e.push_back(digi_hit.getEDep())
                theta.push_back(get_theta(x_pos, y_pos, z_pos))
                
                #subdet = decoder(digi_hit)["subdet"]
                #layer = decoder(digi_hit)["layer"]
                #subdetector.push_back(subdet)
                #layer.push_back(layer)

        pix_cols = {}
        for pix_col in PIXEL_COLLECTIONS:
            pix_cols[pix_col] = get_collection(event, pix_col)

        for pix_col in pix_cols:
            print(f"  {len(pix_cols[pix_col]):5} hits in {pix_col}")

        for pix_col_name in PIXEL_COLLECTIONS:
            for i_hit, hit in enumerate(pix_cols[pix_col_name]):
                if i_hit < ops.nhits:
                    break

                digi_hit, sim_hit = hit.getFrom(), hit.getTo()
                position = digi_hit.getPosition()
                hits     = digi_hit.getRawHits()
                x_pos = position[0]
                y_pos = position[1]
                cluster_x, cluster_y = get_cluster_size(x_pos, y_pos)
                cluster_size_x.push_back(cluster_x)
                cluster_size_y.push_back(cluster_y)
                cluster_size_tot.push_back(len(hits))

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
