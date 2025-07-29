import ROOT
import array

# Setup TMVA reader
reader = ROOT.TMVA.Reader("!Color:!Silent")
x = array.array('f', [0.])
y = array.array('f', [0.])
reader.AddVariable("x", x)
reader.AddVariable("y", y)

# Load trained weights
reader.BookMVA("BDT", "dataset/weights/TMVAClassification_BDT.weights.xml")

# Input files
signal_file = ROOT.TFile.Open("signal.root")
background_file = ROOT.TFile.Open("background.root")
sig_tree = signal_file.Get("tree")
bkg_tree = background_file.Get("tree")

# Histograms for BDT output
h_sig = ROOT.TH1F("h_sig", "BDT Output;BDT Score;Events", 50, -1, 1)
h_bkg = ROOT.TH1F("h_bkg", "BDT Output;BDT Score;Events", 50, -1, 1)

# Fill histograms
for event in sig_tree:
    x[0], y[0] = event.x, event.y
    score = reader.EvaluateMVA("BDT")
    h_sig.Fill(score)

for event in bkg_tree:
    x[0], y[0] = event.x, event.y
    score = reader.EvaluateMVA("BDT")
    h_bkg.Fill(score)

# Normalize for comparison
h_sig.Scale(1.0 / h_sig.Integral())
h_bkg.Scale(1.0 / h_bkg.Integral())

# Draw BDT Score distributions
canvas = ROOT.TCanvas("c1", "BDT Output", 800, 600)
h_sig.SetLineColor(ROOT.kRed)
h_bkg.SetLineColor(ROOT.kBlue)
h_sig.SetLineWidth(2)
h_bkg.SetLineWidth(2)

h_sig.Draw("HIST")
h_bkg.Draw("HIST SAME")
legend = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
legend.AddEntry(h_sig, "Signal", "l")
legend.AddEntry(h_bkg, "Background", "l")
legend.Draw()

canvas.SaveAs("bdt_score_distribution.png")

# ROC Curve
sig_eff = []
bkg_eff = []
n_bins = h_sig.GetNbinsX()

for i in range(n_bins):
    sig_int = h_sig.Integral(i+1, n_bins)
    bkg_int = h_bkg.Integral(i+1, n_bins)
    sig_eff.append(sig_int)
    bkg_eff.append(bkg_int)

# Draw ROC curve
roc = ROOT.TGraph(len(sig_eff), array.array('f', bkg_eff), array.array('f', sig_eff))
roc.SetTitle("ROC Curve;Background Efficiency;Signal Efficiency")
roc.SetLineWidth(2)
roc.SetLineColor(ROOT.kGreen+2)

c2 = ROOT.TCanvas("c2", "ROC", 600, 600)
roc.Draw("AL")
c2.SaveAs("bdt_roc_curve.png")

print("Plots saved: bdt_score_distribution.png, bdt_roc_curve.png")
