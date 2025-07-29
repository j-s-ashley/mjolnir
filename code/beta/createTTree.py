import os
from pathlib import Path
import argparse
import pyLCIO
import ROOT, array

COLLECTIONS = [
    "ITBarrelHitsRelations_HTF",
    "VXDBarrelHitsRelations_HTF",
    "VXDEndcapHitsRelations_HTF",
    "ITEndcapHitsRelations_HTF",
    "OTBarrelHitsRelations_HTF",
    "OTEndcapHitsRelations_HTF"
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
    isSec = ROOT.std.vector('float')()
    mc_px = ROOT.std.vector('float')()
    mc_py = ROOT.std.vector('float')()
    mc_pz = ROOT.std.vector('float')()
    mc_vx = ROOT.std.vector('float')()
    mc_vy = ROOT.std.vector('float')()
    mc_vz = ROOT.std.vector('float')()
    mc_id = ROOT.std.vector('float')()
    mc_pdgid = ROOT.std.vector('float')()
    mc_charge = ROOT.std.vector('float')()
    
    # Create branches
    tree.Branch("Hit_x", x)
    tree.Branch("Hit_y", y)
    tree.Branch("Hit_z", z)
    tree.Branch("Hit_ArrivalTime", t)
    tree.Branch("Hit_EnergyDeposited", e)
    tree.Branch("Hit_isFromSecondary", isSec)
    tree.Branch("MCP_Vx", mc_vx)
    tree.Branch("MCP_Vy", mc_vy)
    tree.Branch("MCP_Vz", mc_vz)
    tree.Branch("MCP_Px", mc_px)
    tree.Branch("MCP_Py", mc_py)
    tree.Branch("MCP_Pz", mc_pz)
    tree.Branch("MCP_ID", mc_id)
    tree.Branch("MCP_PDGID", mc_pdgid)
    tree.Branch("MCP_Charge", mc_charge)

    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(str(in_file))

    for i_event, event in enumerate(reader):
        x.clear()
        y.clear()
        z.clear()
        t.clear()
        e.clear()
        mc_px.clear()
        mc_py.clear()
        mc_pz.clear()
        mc_vx.clear()
        mc_vy.clear()
        mc_vz.clear()
        mc_id.clear()
        mc_charge.clear()
        mc_pdgid.clear()

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
