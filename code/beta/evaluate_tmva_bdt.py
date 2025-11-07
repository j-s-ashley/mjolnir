import ROOT
import numpy as np
from ROOT import TMVA, TFile, TH1F
from sklearn.metrics import roc_curve, auc
from array import array

ROOT.TMVA.Tools.Instance()
reader    = ROOT.TMVA.Reader("!Color:Silent")
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

for i in range(9):
    variables.append(f"PixelHits_EnergyDeposited_{i}")
    variables.append(f"PixelHits_ArrivalTime_{i}")

buffers = {v: array('f', [0.]) for v in variables}
for v in variables:
    reader.AddVariable(v, buffers[v])

reader.BookMVA("BDT", "dataset/weights/TMVAClassification_BDT.weights.xml")

# Input files
sig_file = ROOT.TFile.Open("../../data/beta/MAIA/signal/Hits_TTree_output_digi_light_eval.root")
bkg_file = ROOT.TFile.Open("../../data/beta/MAIA/bg/Hits_TTree_output_digi_light_eval.root")
sig_tree = sig_file.Get("HitTree")
bkg_tree = bkg_file.Get("HitTree")

out_file = ROOT.TFile("BDT_Eval.root", "RECREATE")
out_file.cd()

def evaluate_flat_tree(flat_tree, scores_list):
    for evt in flat_tree:
        for v in variables:
            buffers[v][0] = getattr(evt, v)
        score = reader.EvaluateMVA("BDT")
        scores_list.append(score)

# Evaluate signal and background
sig_scores = []
bkg_scores = []
evaluate_flat_tree(sig_tree, sig_scores)
evaluate_flat_tree(bkg_tree, bkg_scores)

y_true      = np.array([1]*len(sig_scores) + [0]*len(bkg_scores))
y_score     = np.array(sig_scores + bkg_scores)
fpr, tpr, _ = roc_curve(y_true, y_score)
bkg_rej     = 1 - fpr
sig_eff     = tpr
roc_auc     = auc(sig_eff, bkg_rej)
print(f"ROC AUC (Signal efficiency vs Background rejection) = {roc_auc:.3f}")

# --- Save score histograms and ROC ---
h_sig = ROOT.TH1F("h_sig_score", "Signal BDT Output;BDT Score;Entries", 100, -1, 1)
for s in sig_scores:
    h_sig.Fill(s)
h_sig.Scale(1. / h_sig.Integral())

h_bkg = ROOT.TH1F("h_bkg_score", "Background BDT Output;BDT Score;Entries", 100, -1, 1)
for b in bkg_scores:
    h_bkg.Fill(b)
h_bkg.Scale(1. / h_bkg.Integral())

c1 = ROOT.TCanvas("c1", "BDT Output", 800, 600)
h_sig.SetLineColor(ROOT.kRed)
h_bkg.SetLineColor(ROOT.kBlue)
h_sig.SetLineWidth(2)
h_bkg.SetLineWidth(2)
h_sig.Draw("HIST")
h_bkg.Draw("HIST SAME")
h_sig.SetStats(0)

legend = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
legend.AddEntry(h_sig, "Signal clusters", "l")
legend.AddEntry(h_bkg, "Background clusters", "l")
legend.Draw()

c1.SaveAs("BDT_Evaluation_wpixels.png")
c1.Write()

c2 = ROOT.TCanvas("c2", "ROC Curve (Signal Eff vs Background Rejection)", 600, 600)
g  = ROOT.TGraph(len(sig_eff), array('f', bkg_rej), array('f', sig_eff))

g.SetTitle(f"ROC Curve;Background Rejection;Signal Efficiency")
g.SetLineColor(ROOT.kBlue)
g.SetLineWidth(2)
g.Draw("AL")
g.GetXaxis().SetLimits(0, 1)
g.GetYaxis().SetRangeUser(0, 1)

c2.SaveAs("BDT_ROC_SigEff_vs_BkgRej_wpixels.png")
g.Write("ROC_curve")

# Write all to output file
h_sig.Write()
h_bkg.Write()

out_file.Close()
