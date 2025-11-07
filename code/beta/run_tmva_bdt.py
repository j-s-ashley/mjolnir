import ROOT
from ROOT import TMVA
from array import array

# Create output file
output_file = ROOT.TFile("TMVA_output.root", "RECREATE")

# Initialize TMVA
TMVA.Tools.Instance()
factory = TMVA.Factory("TMVAClassification", output_file,"!V:!Silent:Color:DrawProgressBar:Transformations=I;:AnalysisType=Classification")
factory = TMVA.Factory("TMVAClassification", output_file,
        "!Silent:Color:DrawProgressBar:Transformations=I;:AnalysisType=Classification:Correlations:InputCorrelations:VariableImportance")
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

output_file.Close()
print("TMVA training completed. Output saved to 'TMVA_output.root'")
