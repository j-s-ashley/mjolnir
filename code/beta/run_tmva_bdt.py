import ROOT
from ROOT import TMVA

# Create output file
outputFile = ROOT.TFile("TMVA_output.root", "RECREATE")

# Initialize TMVA
TMVA.Tools.Instance()
factory = TMVA.Factory("TMVAClassification", outputFile,"!V:!Silent:Color:DrawProgressBar:Transformations=I;:AnalysisType=Classification")
dataloader = TMVA.DataLoader("dataset")

# Define input variables
dataloader.AddVariable("Cluster_ArrivalTime", "F")
dataloader.AddVariable("Cluster_EnergyDeposited", "F")
dataloader.AddVariable("Incident_Angle", "F")
dataloader.AddVariable("Cluster_Size_x", "F")
dataloader.AddVariable("Cluster_Size_y", "F")
dataloader.AddVariable("Cluster_Size_tot", "F")

# Load signal and background files
sig_file = ROOT.TFile("../../data/beta/digi_signal.root")
bkg_file = ROOT.TFile("../../data/beta/digi_bg.root")
sig_tree = sig_file.Get("tree")
bkg_tree = bkg_file.Get("tree")

dataloader.AddSignalTree(sig_tree)
dataloader.AddBackgroundTree(bkg_tree)

# Prepare dataset
dataloader.PrepareTrainingAndTestTree(ROOT.TCut(""), ROOT.TCut(""),
    "nTrain_Signal=1:nTrain_Background=10000:SplitMode=Random:NormMode=NumEvents:!V")

# Book a BDT
factory.BookMethod(dataloader, TMVA.Types.kBDT, "BDT",
    "!H:!V:NTrees=200:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20")

# Train, test, evaluate
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

outputFile.Close()
print("TMVA training completed. Output saved to 'TMVA_output.root'")
