import ROOT
import numpy as np
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
#from mpl_toolkits import mplot3d
import random
import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy.stats import multivariate_normal
np.random.seed(41)

import Quantities
import datagenerate


a,b = datagenerate.Gaussdata(1000000,0.2,0.2,0.05,0.08,-0.4)

print(a)
print(b)
heat = ROOT.TH2D("heat","heat",200,0,1,200,0,1)

for i in range(len(a)):   
    heat.Fill(a[i],b[i])

c = ROOT.TCanvas("sosa","sosa",1000,2000)
heat.Draw("colz");

c.SaveAs("saner.pdf")





