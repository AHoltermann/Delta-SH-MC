import ROOT
import numpy as np
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
#from mpl_toolkits import mplot3d
import random
import matplotlib.pyplot as plt
from matplotlib import pyplot
#%matplotlib inline
from scipy.stats import multivariate_normal
np.random.seed(41)


print("food")

h = ROOT.TH2F("gauss","Example histogram",100,-4,4,-4,4)
h.FillRandom2("gaus")
c2 = ROOT.TCanvas("c2","c2",400,400)
h.Draw()

c2.SaveAs("sane.pdf")

