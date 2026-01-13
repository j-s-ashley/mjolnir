import ROOT
from ROOT import TMVA
from dataclasses import dataclass
from array import array

# Manual normalization
def normalize_in_place(h):
    integ = h.Integral()
    if integ > 0:
        h.Scale(1.0 / integ)

# Create output file
output_file = ROOT.TFile("TMVA_output.root", "RECREATE")

# Initialize TMVA
TMVA.Tools.Instance()
factory = TMVA.Factory("TMVAClassification", output_file,
        "!V:!Silent:Color:DrawProgressBar:Transformations=I;:AnalysisType=Classification")
dataloader = TMVA.DataLoader("dataset")

@dataclass(frozen=True)
class Variable:
    label: str
    ymax: float
    xmin: float
    xmax: float

variables = {
        "Cluster_ArrivalTime": Variable(label="Cluster arrival time [ns]", ymax=0.23, xmin=-0.52, xmax=0.7),
        "Cluster_EnergyDeposited": Variable(label="Cluster energy deposited [KeV]", ymax=1.1, xmin=0, xmax=0.0007),
        "Incident_Angle": Variable(label="Incident angle", ymax=0.077, xmin=0, xmax=3),
        "Cluster_Size_x": Variable(label="Cluster size in x [pixels]", ymax=1.1, xmin=0, xmax=60),
        "Cluster_Size_y": Variable(label="Cluster size in y [pixels]", ymax=1.1, xmin=0, xmax=400),
        "Cluster_Size_tot": Variable(label="Total cluster size [pixels]", ymax=1.1, xmin=0, xmax=400),
        "Cluster_x": Variable(label="Cluster x position [cm]", ymax=0.11, xmin=-35, xmax=35),
        "Cluster_y": Variable(label="Cluster y position [cm]", ymax=0.11, xmin=-35, xmax=35),
        "Cluster_z": Variable(label="Cluster z position [cm]", ymax=0.0055, xmin=-80, xmax=80),
        "Cluster_RMS_x": Variable(label="Cluster RMS in x [cm^2]", ymax=0.22, xmin=0, xmax=90000),
        "Cluster_RMS_y": Variable(label="Cluster RMS in y [cm^2]", ymax=0.22, xmin=0, xmax=350000),
        "Cluster_Skew_x": Variable(label="Cluster skew in x", ymax=0.55, xmin=-1.75, xmax=1.75),
        "Cluster_Skew_y": Variable(label="Cluster skew in y", ymax=0.55, xmin=-1.75, xmax=1.75),
        "Cluster_AspectRatio": Variable(label="Cluster aspect ratio", ymax=1.1, xmin=0, xmax=25000),
        "Cluster_Eccentricity": Variable(label="Cluster eccentricity", ymax=1.1, xmin=0.85, xmax=1)
        }

# Load input variables
for v_id, _ in variables.items():
    dataloader.AddVariable(v_id, "F")
for i in range(9):
    dataloader.AddVariable(f"PixelHits_EnergyDeposited_{i}", "F")
    dataloader.AddVariable(f"PixelHits_ArrivalTime_{i}", "F")

# Load signal and background files
#sig_file = ROOT.TFile("/global/cfs/projectdirs/atlas/jashley/mjolnir/data/beta/MAIA/signal/Hits_TTree_output_digi_light_training.root")
#bkg_file = ROOT.TFile("/global/cfs/projectdirs/atlas/jashley/mjolnir/data/beta/MAIA/bg/Hits_TTree_output_digi_light_training.root")
sig_file = ROOT.TFile("/global/cfs/projectdirs/atlas/jashley/mjolnir/data/beta/MAIA/Hits_TTree_output_digi_noMS_5kMuons_10_170Theta_0_5TeV_1.root")
bkg_file = ROOT.TFile("/global/cfs/projectdirs/atlas/jashley/mjolnir/data/beta/MAIA/Hits_TTree_MAIAvxb75_noMS_1_0-0.root")

sig_tree = sig_file.Get("HitTree")
bkg_tree = bkg_file.Get("HitTree")

dataloader.AddSignalTree(sig_tree, 1.0)
dataloader.AddBackgroundTree(bkg_tree, 1.0)

# Prepare dataset
dataloader.PrepareTrainingAndTestTree(ROOT.TCut(""), ROOT.TCut(""),
    "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V")

# Book a BDT
factory.BookMethod(dataloader, TMVA.Types.kBDT, "BDT",
    "!H:!V:NTrees=200:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20")

# Train, test, evaluate
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

# --- INPUT VARIABLE DISTRIBUTIONS --- #
n_bins = 50
ROOT.gStyle.SetOptStat(0)

for v_id, v in variables.items():
    x_min = v.xmin
    x_max = v.xmax
    y_max = v.ymax

    h_sig_name = f"h_sig_{v_id}"
    h_bkg_name = f"h_bkg_{v_id}"

    h_sig = ROOT.TH1F(h_sig_name, f"Normalized {v.label} (signal vs background)", n_bins, x_min, x_max)
    h_bkg = ROOT.TH1F(h_bkg_name, f"Normalized {v.label} (signal vs background)", n_bins, x_min, x_max)

    # Fill hists from original trees
    sig_tree.Draw(f"{v_id}>>{h_sig_name}", "", "goff")
    bkg_tree.Draw(f"{v_id}>>{h_bkg_name}", "", "goff")

    # Pretty plots
    c = ROOT.TCanvas(f"c_{v_id}", f"{v_id} signal vs background", 800, 600)

    h_sig.SetLineColor(ROOT.kRed)
    h_bkg.SetLineColor(ROOT.kBlue)
    h_sig.SetLineWidth(2)
    h_bkg.SetLineWidth(2)
    
    # Fix axis issues by normalizing manually
    normalize_in_place(h_sig)
    normalize_in_place(h_bkg)

    h_sig.SetMaximum(y_max)
    h_sig.GetXaxis().SetTitle(f"{v.label}")
    h_sig.GetYaxis().SetTitle("Normalized number of clusters")

    h_sig.Draw("HIST")
    h_bkg.Draw("HIST SAME")

    leg = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
    leg.AddEntry(h_sig, "Signal", "l")
    leg.AddEntry(h_bkg, "Background", "l")
    leg.Draw()

    c.Write()

output_file.Close()
print("TMVA training completed. Output saved to 'TMVA_output.root'")
