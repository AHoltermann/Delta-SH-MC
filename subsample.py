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

import datagenerate
import Quantities



def Subsample_Gauss(qty,subsamples,samples,u1,u2,s1,s2,p):
    data = np.zeros(subsamples)
    s1 = np.zeros(subsamples)
    s2 = np.zeros(subsamples)
    s3 = np.zeros(subsamples)

    v2_arr = np.zeros(subsamples)
    v4_arr = np.zeros(subsamples)
    v2p_arr = np.zeros(subsamples)
    v4p_arr = np.zeros(subsamples)

    for i in range(subsamples):
        a,b = datagenerate.Gaussdata(samples,u1,u2,s1,s2,p)
        
        data[i] = Quantities.quantity(qty,a,b)
        s1[i], s2[i], s3[i] = Quantities.Variance_ratios(a,b)
        
        v2_arr[i],v2p_arr[i],v4_arr[i],v4p_arr[i] = Quantities.Cumulants(a,b)

    var_2 = np.var(v2_arr) - np.var(v2p_arr)
    var_4 = np.var(v4_arr) - np.var(v4p_arr)


    return np.mean(data), np.std(data)/(np.sqrt(subsamples)), np.mean(s1), np.std(s1)/(np.sqrt(subsamples)), np.mean(s2), np.std(s2)/(np.sqrt(subsamples)), np.mean(s3), np.std(s3)/(np.sqrt(subsamples)), var_2, var_4


def Subsample_BVGE(qty,subsamples,samples,a1,a2,theta):
    data = np.zeros(subsamples)
    s1 = np.zeros(subsamples)
    s2 = np.zeros(subsamples)
    s3 = np.zeros(subsamples)

    v2_arr = np.zeros(subsamples)
    v4_arr = np.zeros(subsamples)
    v2p_arr = np.zeros(subsamples)
    v4p_arr = np.zeros(subsamples)

    for i in range(subsamples):
        a,b = datagenerate.bvgedata(samples,a1,a2,theta)
        
        data[i] = Quantities.quantity(qty,a,b)
        s1[i], s2[i], s3[i] = Quantities.Variance_ratios(a,b)
        
        v2_arr[i],v2p_arr[i],v4_arr[i],v4p_arr[i] = Quantities.Cumulants(a,b)

    var_2 = np.var(v2_arr) - np.var(v2p_arr)
    var_4 = np.var(v4_arr) - np.var(v4p_arr)


    return np.mean(data), np.std(data)/(np.sqrt(subsamples)), np.mean(s1), np.std(s1)/(np.sqrt(subsamples)), np.mean(s2), np.std(s2)/(np.sqrt(subsamples)), np.mean(s3), np.std(s3)/(np.sqrt(subsamples)), var_2, var_4


