import ROOT
from ROOT import TMVA
from array import array

# Create output file
output_file = ROOT.TFile("TMVA_output.root", "RECREATE")

# Initialize TMVA
TMVA.Tools.Instance()
factory = TMVA.Factory("TMVAClassification", output_file,
        "!V:!Silent:Color:DrawProgressBar:Transformations=I;:AnalysisType=Classification")
dataloader = TMVA.DataLoader("dataset")

# Define input variables
variables = [
    "Cluster_ArrivalTime",
    "Cluster_EnergyDeposited",
    "Incident_Angle",
    "Cluster_Size_x",
    "Cluster_Size_y",
    "Cluster_Size_tot",
    "Cluster_x",
    "Cluster_y",
    "Cluster_z",
    "Cluster_RMS_x",
    "Cluster_RMS_y",
    "Cluster_Skew_x",
    "Cluster_Skew_y",
    "Cluster_AspectRatio",
    "Cluster_Eccentricity"
]

# Load input variables
for v in variables:
    dataloader.AddVariable(v, "F")
for i in range(9):
    dataloader.AddVariable(f"PixelHits_EnergyDeposited_{i}", "F")
    dataloader.AddVariable(f"PixelHits_ArrivalTime_{i}", "F")

# Load signal and background files
sig_file = ROOT.TFile("/global/cfs/projectdirs/atlas/jashley/mjolnir/data/beta/MAIA/signal/Hits_TTree_output_digi_light_training.root")
bkg_file = ROOT.TFile("/global/cfs/projectdirs/atlas/jashley/mjolnir/data/beta/MAIA/bg/Hits_TTree_output_digi_light_training.root")
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

n_bins = 50

for v in variables:
    # Determine common range from both trees
    sig_min = sig_tree.GetMinimum(v)
    sig_max = sig_tree.GetMaximum(v)
    bkg_min = bkg_tree.GetMinimum(v)
    bkg_max = bkg_tree.GetMaximum(v)

    x_min = min(sig_min, bkg_min)
    x_max = max(sig_max, bkg_max)

    h_sig_name = f"h_sig_{v}"
    h_bkg_name = f"h_bkg_{v}"

    h_sig = ROOT.TH1F(h_sig_name, f"Normalized {v} (signal vs background)", n_bins, x_min, x_max)
    h_bkg = ROOT.TH1F(h_bkg_name, f"Normalized {v} (signal vs background)", n_bins, x_min, x_max)

    # Fill hists from original trees
    sig_tree.Draw(f"{v}>>{h_sig_name}", "", "goff")
    bkg_tree.Draw(f"{v}>>{h_bkg_name}", "", "goff")

    # Pretty plots
    ROOT.gStyle.SetOptStat(0)
    c = ROOT.TCanvas(f"c_{v}", f"{v} signal vs background", 800, 600)

    h_sig.SetLineColor(ROOT.kRed)
    h_bkg.SetLineColor(ROOT.kBlue)
    h_sig.SetLineWidth(2)
    h_bkg.SetLineWidth(2)
    h_sig.DrawNormalized("HIST")
    h_bkg.DrawNormalized("HIST SAME")

    leg = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
    leg.AddEntry(h_sig, "Signal", "l")
    leg.AddEntry(h_bkg, "Background", "l")
    leg.Draw()

    c.Write()

output_file.Close()
print("TMVA training completed. Output saved to 'TMVA_output.root'")
