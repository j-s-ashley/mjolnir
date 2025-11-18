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

# --- INPUT --- #
# Training files
sig_training_file = ROOT.TFile.Open("../../data/beta/MAIA/signal/Hits_TTree_output_digi_light_training.root")
bkg_training_file = ROOT.TFile.Open("../../data/beta/MAIA/bg/Hits_TTree_output_digi_light_training.root")
sig_training_tree = sig_training_file.Get("HitTree")
bkg_training_tree = bkg_training_file.Get("HitTree")
# Evaluation files
sig_eval_file = ROOT.TFile.Open("../../data/beta/MAIA/signal/Hits_TTree_output_digi_light_eval.root")
bkg_eval_file = ROOT.TFile.Open("../../data/beta/MAIA/bg/Hits_TTree_output_digi_light_eval.root")
sig_eval_tree = sig_eval_file.Get("HitTree")
bkg_eval_tree = bkg_eval_file.Get("HitTree")

combined_out_file = ROOT.TFile("BDT_combined_training_eval.root", "RECREATE")

def evaluate_flat_tree(flat_tree, scores_list):
    for evt in flat_tree:
        for v in variables:
            buffers[v][0] = getattr(evt, v)
        score = reader.EvaluateMVA("BDT")
        scores_list.append(score)

# --- EVALUATE SIGNAL AND BACKGROUND --- #
combined_out_file.cd()

# --- Training  files --- #
sig_training_scores = []
bkg_training_scores = []
evaluate_flat_tree(sig_training_tree, sig_training_scores)
evaluate_flat_tree(bkg_training_tree, bkg_training_scores)

training_y_true      = np.array([1]*len(sig_training_scores) + [0]*len(bkg_training_scores))
training_y_score     = np.array(sig_training_scores + bkg_training_scores)
training_fpr, training_tpr, training_ = roc_curve(training_y_true, training_y_score)
training_bkg_rej     = 1 - training_fpr
training_sig_eff     = training_tpr
training_roc_auc     = auc(training_sig_eff, training_bkg_rej)
print(f"Training  ROC AUC (Signal efficiency vs Background rejection) = {training_roc_auc:.3f}")

# Save score histograms and ROC
h_sig_training = ROOT.TH1F("h_sig_training_score", "Signal BDT Output (Training );BDT Score;Entries", 100, -1, 1)
for s in sig_training_scores:
    h_sig_training.Fill(s)
h_sig_training.Scale(1. / h_sig_training.Integral())

h_bkg_training = ROOT.TH1F("h_bkg_training_score", "Background BDT Output (Training );BDT Score;Entries", 100, -1, 1)
for b in bkg_training_scores:
    h_bkg_training.Fill(b)
h_bkg_training.Scale(1. / h_bkg_training.Integral())

c_hist_train = ROOT.TCanvas("c_hist_train", "BDT Output", 800, 600)
h_sig_training.SetLineColor(ROOT.kRed)
h_bkg_training.SetLineColor(ROOT.kBlue)
h_sig_training.SetLineWidth(2)
h_bkg_training.SetLineWidth(2)
h_sig_training.Draw("HIST")
h_bkg_training.Draw("HIST SAME")
h_sig_training.SetStats(0)

training_hist_legend = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
training_hist_legend.AddEntry(h_sig_training, "Signal clusters", "l")
training_hist_legend.AddEntry(h_bkg_training, "Background clusters", "l")
training_hist_legend.Draw()

c_hist_train.SaveAs("BDT_evaluation_of_training_wpixels.png")
c_hist_train.Write()

c_roc_train = ROOT.TCanvas("c_roc_train", "ROC Curve (Signal Eff vs Background Rejection)", 600, 600)
g_train = ROOT.TGraph(len(training_sig_eff), array('f', training_bkg_rej), array('f', training_sig_eff))

training_roc_legend = ROOT.TLegend(0.15, 0.20, 0.35, 0.35)
g_train.SetTitle(f"ROC Curve;Background Rejection;Signal Efficiency")
g_train.SetLineColor(ROOT.kBlue)
g_train.SetLineWidth(2)
g_train.Draw("AL")
g_train.GetXaxis().SetLimits(0, 1)
g_train.GetYaxis().SetRangeUser(0, 1)

c_roc_train.SaveAs("BDT_ROC_of_training_SigEff_vs_BkgRej_wpixels.png")
g_train.Write("training_ROC_curve")

# --- Evaluation files --- #
sig_eval_scores = []
bkg_eval_scores = []
evaluate_flat_tree(sig_eval_tree, sig_eval_scores)
evaluate_flat_tree(bkg_eval_tree, bkg_eval_scores)

eval_y_true      = np.array([1]*len(sig_eval_scores) + [0]*len(bkg_eval_scores))
eval_y_score     = np.array(sig_eval_scores + bkg_eval_scores)
eval_fpr, eval_tpr, eval_ = roc_curve(eval_y_true, eval_y_score)
eval_bkg_rej     = 1 - eval_fpr
eval_sig_eff     = eval_tpr
eval_roc_auc     = auc(eval_sig_eff, eval_bkg_rej)
print(f"Evaluation ROC AUC (Signal efficiency vs Background rejection) = {eval_roc_auc:.3f}")

# Save score histograms and ROC
h_sig_eval = ROOT.TH1F("h_sig_eval_score", "Signal BDT Output (Evaluation);BDT Score;Entries", 100, -1, 1)
for s in sig_eval_scores:
    h_sig_eval.Fill(s)
h_sig_eval.Scale(1. / h_sig_eval.Integral())

h_bkg_eval = ROOT.TH1F("h_bkg_eval_score", "Background BDT Output (Evaluation);BDT Score;Entries", 100, -1, 1)
for b in bkg_eval_scores:
    h_bkg_eval.Fill(b)
h_bkg_eval.Scale(1. / h_bkg_eval.Integral())

c_hist_eval = ROOT.TCanvas("c_hist_eval", "BDT Output", 800, 600)
h_sig_eval.SetLineColor(ROOT.kRed)
h_bkg_eval.SetLineColor(ROOT.kBlue)
h_sig_eval.SetLineWidth(2)
h_bkg_eval.SetLineWidth(2)
h_sig_eval.Draw("HIST")
h_bkg_eval.Draw("HIST SAME")
h_sig_eval.SetStats(0)

eval_hist_legend = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
eval_hist_legend.AddEntry(h_sig_eval, "Signal clusters", "l")
eval_hist_legend.AddEntry(h_bkg_eval, "Background clusters", "l")
eval_hist_legend.Draw()

c_hist_eval.SaveAs("BDT_evaluation_of_eval_wpixels.png")
c_hist_eval.Write()

c_roc_eval = ROOT.TCanvas("c_roc_eval", "ROC Curve (Signal Eff vs Background Rejection)", 600, 600)
g_eval = ROOT.TGraph(len(eval_sig_eff), array('f', eval_bkg_rej), array('f', eval_sig_eff))

eval_roc_legend = ROOT.TLegend(0.15, 0.20, 0.35, 0.35)
g_eval.SetTitle(f"ROC Curve;Background Rejection;Signal Efficiency")
g_eval.SetLineColor(ROOT.kBlue)
g_eval.SetLineWidth(2)
g_eval.Draw("AL")
g_eval.GetXaxis().SetLimits(0, 1)
g_eval.GetYaxis().SetRangeUser(0, 1)

c_roc_eval.SaveAs("BDT_ROC_of_eval_SigEff_vs_BkgRej_wpixels.png")
g_eval.Write("eval_ROC_curve")

# --- Combined stats --- #
# Save score histograms and ROC
c_hist_comb = ROOT.TCanvas("c_hist_comb", "BDT Output", 800, 600)
h_sig_eval.SetLineColor(ROOT.kRed)
h_bkg_eval.SetLineColor(ROOT.kBlue)
h_sig_training.SetLineColor(ROOT.kMagenta)
h_bkg_training.SetLineColor(ROOT.kCyan)
h_sig_eval.Draw("HIST")
h_bkg_eval.Draw("HIST SAME")
h_sig_training.Draw("HIST SAME")
h_bkg_training.Draw("HIST SAME")
h_sig_eval.SetStats(0)
h_bkg_eval.SetStats(0)
h_sig_training.SetStats(0)
h_bkg_training.SetStats(0)

comb_hist_legend = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
comb_hist_legend.AddEntry(h_sig_eval, "Signal clusters in evaluation data", "l")
comb_hist_legend.AddEntry(h_sig_training, "Signal clusters in training data", "l")
comb_hist_legend.AddEntry(h_bkg_eval, "Background clusters in evaluation data", "l")
comb_hist_legend.AddEntry(h_bkg_training, "Background clusters in training data", "l")
comb_hist_legend.Draw()

c_hist_comb.SaveAs("BDT_evaluation_of_comb_wpixels.png")
c_hist_comb.Write()

c_roc_comb = ROOT.TCanvas("c_roc_comb", "ROC Curve (Signal Eff vs Background Rejection)", 600, 600)

comb_roc_legend = ROOT.TLegend(0.15, 0.20, 0.35, 0.35)
g_train.SetLineColorAlpha(ROOT.kBlue,0.5)
g_eval.SetLineColorAlpha(ROOT.kRed,0.5)
g_train.Draw("AL")
g_eval.Draw("AL")
comb_roc_legend.AddEntry(g_train,"Training Data","l")
comb_roc_legend.AddEntry(g_eval,"Evaluation Data","l")
comb_roc_legend.Draw()

c_roc_comb.SaveAs("BDT_ROC_of_comb_SigEff_vs_BkgRej_wpixels.png")

# Write all to output file
h_sig_eval.Write()
h_bkg_eval.Write()
h_sig_training.Write()
h_bkg_training.Write()

combined_out_file.Close()
