import ROOT
import numpy as np
import ctypes
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
#from mpl_toolkits import mplot3d
import random
import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy.stats import multivariate_normal
np.random.seed(41)

import subsample

def gauss():

    data_gauss = np.shape(16,21)
    data_gauss_error = np.shape(16,21)
    sig_1 = np.shape(16,21)
    sig_1_e = np.shape(16,21)
    sig_2 = np.shape(16,21)
    sig_2_e = np.shape(16,21)
    sig_3 = np.shape(16,21)
    sig_3_e = np.shape(16,21)
    var2 = np.shape(16,21)
    var4 = np.shape(16,21)
    orth = np.shape(16,21)


    var_v = 0.055
    u_v = 0.1
    u_vp = 0.07

    for i in range(16):
        var_vp = (0.1*i)*(var_v/u_v)*(u_vp)

        for j in range(21):
            p = (10-j)/(10)

            data_gauss[i][j],data_gauss_error[i][j],sig_1[i][j],sig_1_e[i][j],sig_2[i][j],sig_2_e[i][j],sig_3[i][j],sig_3_e[i][j],var2[i][j],var4[i][j] = subsample.Subsample_Gauss("Ratio4_uw",1000,100000,u_v,u_vp,np.sqrt(var_v),np.sqrt(var_vp),p)

            orth[i][j] = p*np.sqrt(var_v)*np.sqrt(var_vp)

    return data_gauss,data_gauss_error,sig_1,sig_1_e,sig_2,sig_2_e,sig_3,sig_3_e,var2,var4,orth

    
def bvge():

    data_bvge = np.shape(16,21)
    data_bvge_error = np.shape(16,21)
    sig_1 = np.shape(16,21)
    sig_1_e = np.shape(16,21)
    sig_2 = np.shape(16,21)
    sig_2_e = np.shape(16,21)
    sig_3 = np.shape(16,21)
    sig_3_e = np.shape(16,21)
    var2 = np.shape(16,21)
    var4 = np.shape(16,21)
    orth = np.shape(16,21)


    a1 = 200

    for i in range(16):
        a2 = i*15

        for j in range(21):
            theta = (10-j)/(10)

            data_bvge[i][j],data_bvge_error[i][j],sig_1[i][j],sig_1_e[i][j],sig_2[i][j],sig_2_e[i][j],sig_3[i][j],sig_3_e[i][j],var2[i][j],var4[i][j] = subsample.Subsample_BVGE("Ratio4_uw",1000,100000,a1,a2,theta)
            orth[i][j] = theta


    return data_bvge,data_bvge_error,sig_1,sig_1_e,sig_2,sig_2_e,sig_3,sig_3_e,var2,var4,orth
        

        


