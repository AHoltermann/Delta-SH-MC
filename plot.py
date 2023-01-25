

import ROOT
import Subsamplerunner
import numpy as np



data,data_error,sig_1,sig_1_e,sig_2,sig_2_e,sig_3,sig_3_e,var2,var4,orth = Subsamplerunner.Gauss()

x = data[1]
y = orth[1]
x_err = data_error[1]
y_err = np.zeros(len(orth[1]))

G = ROOT.TGraphErrors(x,y,x_err,y_err)
G.SetMarkerColor(4)
G.SetMarkerStyle(21)

c = ROOT.TCanvas("Title","Title",1000,2000)
c.cd()
G.Draw("ALP")
c.SaveAs("filename.pdf")






