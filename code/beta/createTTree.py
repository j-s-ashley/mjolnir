import os
import math
from pathlib import Path
import argparse
import pyLCIO
import ROOT, array

COLLECTIONS = [
        "EcalBarrelCollectionDigi",
        "EcalBarrelCollectionRec",
        "EcalEndcapCollectionDigi",
        "EcalEndcapCollectionRec",
        "HcalBarrelCollectionDigi",
        "HcalBarrelCollectionRec",
        "HcalEndcapCollectionDigi",
        "HcalEndcapCollectionRec",
        "IBPixels_HTF",
        "IEPixels_HTF",
        "ITBarrelHits",
        "ITBarrelHitsRelations_HTF",
        "ITEndcapHits",
        "ITEndcapHitsRelations_HTF",
        "InnerTrackerBarrelCollection_HTF",
        "InnerTrackerEndcapCollection_HTF",
        "MCParticle",
        "MuonHits",
        "OBPixels_HTF",
        "OEPixels_HTF",
        "OTBarrelHits",
        "OTBarrelHitsRelations_HTF",
        "OTEndcapHits",
        "OTEndcapHitsRelations_HTF",
        "OuterTrackerBarrelCollection_HTF",
        "OuterTrackerEndcapCollection_HTF",
        "VBPixels_HTF",
        "VEPixels_HTF",
        "VXDBarrelHits",
        "VXDBarrelHitsRelations_HTF",
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
    angle = math.atan2(r/z)
    return angle


def get_cluster_size(trkhit):
    raw_hits = trkhit.getRawHits()
    
    ymax = -1e6
    xmax = -1e6
    ymin = 1e6
    xmin = 1e6

    for hit in raw_hits:
        local_pos = hit.getPosition()  # Assuming (x, y, z)
        x_local   = local_pos[0]
        y_local   = local_pos[1]

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

        mcparticles = event.getCollection("MCParticle")
        for i_mcparticle, mcparticle in enumerate(mcparticles):
            if mcparticle.getPDG()==13 and abs(mcparticle.getVertex()[0])==0 and abs(mcparticle.getVertex()[1])==0 and abs(mcparticle.getVertex()[2])==0:
                mc_px.push_back(mcparticle.getMomentum()[0])
                mc_py.push_back(mcparticle.getMomentum()[1])
                mc_pz.push_back(mcparticle.getMomentum()[2])
                mc_vx.push_back(mcparticle.getVertex()[0])
                mc_vy.push_back(mcparticle.getVertex()[1])
                mc_vz.push_back(mcparticle.getVertex()[2])
                mc_id.push_back(i_mcparticle)
                mc_pdgid.push_back(mcparticle.getPDG())
                mc_charge.push_back(mcparticle.getCharge())
            #print(f"{i_mcparticle}\t(Vx,Vy,Vz)=({mcparticle.getVertex()[0]},{mcparticle.getVertex()[1]},{mcparticle.getVertex()[2]})\tPdgID={mcparticle.getPDG()}\tCharge={mcparticle.getCharge()}")
            
        cols = {}
        for col in COLLECTIONS:
            cols[col] = get_collection(event, col)

        print(f"Event {i_event} has")
        for col in cols:
            print(f"  {len(cols[col]):5} hits in {col}")

        for col_name in COLLECTIONS:
            for i_hit, hit in enumerate(cols[col_name]):
                if i_hit < ops.nhits:
                    break

                digi_hit, sim_hit = hit.getFrom(), hit.getTo()
                position = digi_hit.getPosition()
                getMC = sim_hit.getMCParticle()
                if not getMC: #all the BIB particles
                    isSec.push_back(1)
                else:
                    isSec.push_back(sim_hit.isProducedBySecondary())
                #print(f"(Vx,Vy,Vz)=({getMC.getVertex()[0]},{getMC.getVertex()[1]},{getMC.getVertex()[2]}), charge={getMC.getCharge()}, pdgID={getMC.getPDG()}")
                x.push_back(position[0])
                y.push_back(position[1])
                z.push_back(position[2])
                t.push_back(digi_hit.getTime())
                e.push_back(digi_hit.getEDep())
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
