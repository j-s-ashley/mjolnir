import ROOT
import argparse
from ROOT import TMVA
from dataclasses import dataclass
from array import array

# Accept sensor thickness CLI input
def options():
    parser = argparse.ArgumentParser(description="Train BDT on data from input TTree files.")
    parser.add_argument("-t", required=True, type=int, help="VXB sensor thickness")
    return parser.parse_args()

sensor_thickness = options().t

# Pixel information helper function
def make_pixelhit_vars(prefix, label, *, n=9, ymax, xmin, xmax, legend="right", yscale="log"):
    # Sorry, Tova. Dictionary comprehension just looks so much better here.
    return {
        f"{prefix}_{i}": Variable(
            label=f"{label} for pixel hit {i}",
            ymax=ymax,
            xmin=xmin,
            xmax=xmax,
            legend=legend,
            yscale=yscale,
        )
        for i in range(n)
    }

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
    label: str  # x-axis label for histograms
    ymax: float # max y-value on histograms
    xmin: float # min x-value on histograms
    xmax: float # max x-value on histograms
    legend: str # legend position tag
    yscale: str # linear or log scale for y-axis

variables = {
        "Cluster_ArrivalTime": Variable(
            label="Cluster arrival time [ns]",
            ymax=0.3,
            xmin=-0.52,
            xmax=0.7,
            legend="right",
            yscale="linear",
        ),
        "Cluster_EnergyDeposited": Variable(
            label="Cluster energy deposited [KeV]",
            ymax=1.1,
            xmin=0,
            xmax=0.0007,
            legend="right",
            yscale="log",
        ),
        "Incident_Angle": Variable(
            label="Incident angle [radians]",
            ymax=0.09,
            xmin=0,
            xmax=3,
            legend="center",
            yscale="linear",
        ),
        "Cluster_Size_x": Variable(
            label="Cluster size in x [pixels]",
            ymax=1.1,
            xmin=0,
            xmax=60,
            legend="right",
            yscale="log",
        ),
        "Cluster_Size_y": Variable(
            label="Cluster size in y [pixels]",
            ymax=1.1,
            xmin=0,
            xmax=400,
            legend="right",
            yscale="log",
        ),
        "Cluster_Size_tot": Variable(
            label="Total cluster size [pixels]",
            ymax=1.1,
            xmin=0,
            xmax=400,
            legend="right",
            yscale="log",
        ),
        "Cluster_x": Variable(
            label="Cluster x position [cm]",
            ymax=0.11,
            xmin=-35,
            xmax=35,
            legend="center",
            yscale="linear",
        ),
        "Cluster_y": Variable(
            label="Cluster y position [cm]",
            ymax=0.11,
            xmin=-35,
            xmax=35,
            legend="center",
            yscale="linear",
        ),
        "Cluster_z": Variable(
            label="Cluster z position [cm]",
            ymax=0.55,
            xmin=-80,
            xmax=80,
            legend="right",
            yscale="linear",
        ),
        "Cluster_RMS_x": Variable(
            label="Cluster RMS in x [cm^{2}]",
            ymax=0.22,
            xmin=0,
            xmax=90000,
            legend="right",
            yscale="linear",
        ),
        "Cluster_RMS_y": Variable(
            label="Cluster RMS in y [cm^{2}]",
            ymax=0.22,
            xmin=0,
            xmax=350000,
            legend="right",
            yscale="linear",
        ),
        "Cluster_Skew_x": Variable(
            label="Cluster skew in x",
            ymax=0.55,
            xmin=-1.75,
            xmax=1.75,
            legend="center",
            yscale="log",
        ),
        "Cluster_Skew_y": Variable(
            label="Cluster skew in y",
            ymax=0.55,
            xmin=-1.75,
            xmax=1.75,
            legend="center",
            yscale="log",
        ),
        "Cluster_AspectRatio": Variable(
            label="Cluster aspect ratio",
            ymax=1.1,
            xmin=0,
            xmax=25000,
            legend="right",
            yscale="log",
        ),
        "Cluster_Eccentricity": Variable(
            label="Cluster eccentricity",
            ymax=1.1,
            xmin=0.85,
            xmax=1,
            legend="center",
            yscale="log",
        ),
        }

variables |= make_pixelhit_vars(
    "PixelHits_EnergyDeposited",
    "Pixel hit energy deposited [KeV]",
    ymax=13250,
    xmin=0,
    xmax=62000,
)

variables |= make_pixelhit_vars(
    "PixelHits_ArrivalTime",
    "Pixel hit arrival time [ns]",
    ymax=13500,
    xmin=-2.1,
    xmax=5.2,
)

# Load input variables
for v_id, _ in variables.items():
    dataloader.AddVariable(v_id, "F")

# Load signal and background files
sig_file = ROOT.TFile(f"/global/cfs/projectdirs/atlas/jashley/mjolnir/data/beta/MAIA/signal/{sensor_thickness}_sig_trng_ttree.root")
bkg_file = ROOT.TFile(f"/global/cfs/projectdirs/atlas/jashley/mjolnir/data/beta/MAIA/bg/{sensor_thickness}_bkg_trng_ttree.root")
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

    h_sig = ROOT.TH1F(h_sig_name, f"{sensor_thickness} #mum", n_bins, x_min, x_max)
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
    h_sig.SetFillColorAlpha(ROOT.kRed, 0.35)
    h_bkg.SetFillColorAlpha(ROOT.kBlue, 0.35)
    
    # Fix axis issues by normalizing manually
    normalize_in_place(h_sig)
    normalize_in_place(h_bkg)

    h_sig.SetMaximum(y_max)
    h_sig.GetXaxis().SetTitle(f"{v.label}")
    h_sig.GetYaxis().SetTitle("Normalized number of clusters")

    if v.yscale == "log":
        ROOT.gPad.SetLogy(1)

    h_sig.Draw("HIST F")
    h_bkg.Draw("HIST F SAME")
    h_sig.Draw("HIST SAME")
    h_bkg.Draw("HIST SAME")

    if v.legend == "right":
        leg = ROOT.TLegend(0.7, 0.65, 0.9, 0.8)
    else:
        leg = ROOT.TLegend(0.4, 0.65, 0.6, 0.8)
    leg.AddEntry(h_sig, "Signal", "fl")
    leg.AddEntry(h_bkg, "Background", "fl")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.Draw()

    c.SaveAs(f"{sensor_thickness}_{v_id}_dist.png")
    c.Write()

output_file.Close()
print("TMVA training completed. Output saved to 'TMVA_output.root'")
