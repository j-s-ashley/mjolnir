import ROOT
import array

def create_file(filename, mean_x, mean_y, n_events):
    file = ROOT.TFile(filename, "RECREATE")
    tree = ROOT.TTree("tree", "data")

    x = array.array('f', [0.])
    y = array.array('f', [0.])
    tree.Branch("x", x, "x/F")
    tree.Branch("y", y, "y/F")

    for _ in range(n_events):
        x[0] = ROOT.gRandom.Gaus(mean_x, 1)
        y[0] = ROOT.gRandom.Gaus(mean_y, 1)
        tree.Fill()

    tree.Write()
    file.Close()

# Generate signal and background ROOT files
create_file("signal.root", mean_x=1, mean_y=1, n_events=100)
create_file("background.root", mean_x=-1, mean_y=-1, n_events=100)
print("Created 'signal.root' and 'background.root'")
