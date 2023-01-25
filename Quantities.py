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
        
def moments(v,vp):

        vs = np.mean(v)
        v_2 = np.mean(v*v)
        v_4 = np.mean(v*v*v*v)
        v_6 = np.mean(v*v*v*v*v*v)
        v_8 = np.mean(v*v*v*v*v*v*v*v)

        vps = np.mean(vp)
        v_2m = np.mean(v*vp)
        v_4m = np.mean(v*v*v*vp)
        v_6m = np.mean(v*v*v*v*v*vp)
        v_8m = np.mean(v*v*v*v*v*v*v*vp)

        v_2p = np.mean(vp*vp)
        v_4p = np.mean(v*v*vp*vp)
        v_6p = np.mean(v*v*v*v*vp*vp)
        v_8p = np.mean(v*v*v*v*v*v*vp*vp)

        ref_arr = np.array([vs,v_2,v_4,v_6,v_8])
        diff_arr = np.array([vps,v_2m,v_4m,v_6m,v_8m])
        diff2_arr = np.array([vps,v_2p,v_4p,v_6p,v_8p])

        return ref_arr, diff_arr, diff2_arr


def SC_u(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return ref_arr[2] - ref_arr[1]*ref_arr[1]

def M3_u(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return ref_arr[3] - 3*ref_arr[2]*ref_arr[1] + 2*ref_arr[1]*ref_arr[1]*ref_arr[1]

def MHC3_u(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return ref_arr[3] - 5*ref_arr[2]*ref_arr[1] + 4*ref_arr[1]*ref_arr[1]*ref_arr[1]
    
def M4_u(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return ref_arr[4] - 4*ref_arr[3]*ref_arr[1] +6*ref_arr[2]*ref_arr[1]*ref_arr[1]- 3*ref_arr[1]*ref_arr[1]*ref_arr[1]*ref_arr[1]
    
def AC4_u(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return ref_arr[4] - 4*ref_arr[3]*ref_arr[1]-3*ref_arr[2]*ref_arr[2] +12*ref_arr[2]*ref_arr[1]*ref_arr[1]- 6*ref_arr[1]*ref_arr[1]*ref_arr[1]*ref_arr[1]
    
def MHC4_u(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return ref_arr[4] - 10*ref_arr[3]*ref_arr[1]-9*ref_arr[2]*ref_arr[2] +54*ref_arr[2]*ref_arr[1]*ref_arr[1]- 36*ref_arr[1]*ref_arr[1]*ref_arr[1]*ref_arr[1]    
    
def SC_w(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff_arr[2] - diff_arr[1]*ref_arr[1]

def M3_w(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff_arr[3] - 2*diff_arr[2]*ref_arr[1] - ref_arr[2]*diff_arr[1] + 2*diff_arr[1]*ref_arr[1]*ref_arr[1]

def MHC3_w(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff_arr[3] - 4*diff_arr[2]*ref_arr[1] - diff_arr[1]*ref_arr[2] + 4*diff_arr[1]*ref_arr[1]*ref_arr[1]
    
def M4_w(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff_arr[4] - 3*diff_arr[3]*ref_arr[1] - ref_arr[3]*diff_arr[1] +3*ref_arr[2]*diff_arr[1]*ref_arr[1] + 3*diff_arr[1]*diff_arr[1]*ref_arr[2]- 3*ref_arr[1]*diff_arr[1]*ref_arr[1]*ref_arr[1]
    
def AC4_w(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff_arr[4] - 3*ref_arr[3]*ref_arr[1] - ref_arr[3]*diff_arr[1]-3*ref_arr[2]*diff_arr[2] +6*diff_arr[2]*ref_arr[1]*ref_arr[1] + 6*ref_arr[2]*diff_arr[1]*diff_arr[1]-6*diff_arr[1]*ref_arr[1]*ref_arr[1]*ref_arr[1]
    
def MHC4_w(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff_arr[4] - 9*ref_arr[3]*ref_arr[1] - ref_arr[3]*diff_arr[1]-9*ref_arr[2]*diff_arr[2] +36*diff_arr[2]*ref_arr[1]*ref_arr[1] + 18*ref_arr[2]*diff_arr[1]*diff_arr[1]-36*diff_arr[1]*ref_arr[1]*ref_arr[1]*ref_arr[1]
    
def SC_d(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff2_arr[2] - diff2_arr[1]*ref_arr[1]

def M3_d(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff2_arr[3] - 2*diff2_arr[2]*ref_arr[1] - ref_arr[2]*diff2_arr[1] + 2*diff2_arr[1]*ref_arr[1]*ref_arr[1]

def MHC3_d(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff2_arr[3] - 4*diff2_arr[2]*ref_arr[1] - diff2_arr[1]*ref_arr[2] + 4*diff2_arr[1]*ref_arr[1]*ref_arr[1]
    
def M4_d(v,vpoi):
    ref_arr,diff2_arr,diff2_arr = moments(v,vpoi)
    return diff2_arr[4] - 3*diff2_arr[3]*ref_arr[1] - ref_arr[3]*diff2_arr[1] +3*ref_arr[2]*diff2_arr[1]*ref_arr[1] + 3*diff2_arr[1]*diff2_arr[1]*ref_arr[2]- 3*ref_arr[1]*diff2_arr[1]*ref_arr[1]*ref_arr[1]
    
def AC4_d(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff2_arr[4] - 3*ref_arr[3]*ref_arr[1] - ref_arr[3]*diff2_arr[1]-3*ref_arr[2]*diff2_arr[2] +6*diff2_arr[2]*ref_arr[1]*ref_arr[1] + 6*ref_arr[2]*diff2_arr[1]*diff2_arr[1]-6*diff2_arr[1]*ref_arr[1]*ref_arr[1]*ref_arr[1]
    
def MHC4_d(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff2_arr[4] - 9*ref_arr[3]*ref_arr[1] - ref_arr[3]*diff2_arr[1]-9*ref_arr[2]*diff2_arr[2] +36*diff2_arr[2]*ref_arr[1]*ref_arr[1] + 18*ref_arr[2]*diff2_arr[1]*diff2_arr[1]-36*diff2_arr[1]*ref_arr[1]*ref_arr[1]*ref_arr[1]
    

def SC_b(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff2_arr[2] - diff_arr[1]*diff_arr[1]

def M3_b(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff2_arr[3] - diff2_arr[2]*ref_arr[1]- 2*diff_arr[1]*diff_arr[2] + 2*diff_arr[1]*diff_arr[1]*ref_arr[1]

def MHC3_b(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff2_arr[3] - 4*diff_arr[2]*diff_arr[1] - diff2_arr[2]*diff_arr[1] + 4*diff_arr[1]*diff_arr[1]*ref_arr[1]

def M4_b(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff2_arr[4] - 2*diff_arr[3]*diff_arr[1] - 2*diff2_arr[3]*ref_arr[1] + 4*diff_arr[2]*diff_arr[1]*ref_arr[1] + diff2_arr[2]*ref_arr[1]*ref_arr[1] + ref_arr[2]*diff_arr[1]*diff_arr[1] - 3*diff_arr[1]*diff_arr[1]*ref_arr[1]*ref_arr[1]

def AC4_b(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff2_arr[4] - 2*diff_arr[3]*diff_arr[1] - 2*diff2_arr[3]*ref_arr[1] - diff2_arr[2]*ref_arr[2] - 2*diff_arr[2]*diff_arr[2] + 8*diff_arr[2]*diff_arr[1]*ref_arr[1] + 2*diff2_arr[2]*ref_arr[1]*ref_arr[1] + 2*ref_arr[2]*diff_arr[1]*diff_arr[1] - 6*diff_arr[1]*diff_arr[1]*ref_arr[1]*ref_arr[1]
       
def MHC4_b(v,vpoi):
    ref_arr,diff_arr,diff2_arr = moments(v,vpoi)
    return diff2_arr[4] - 4*diff_arr[3]*diff_arr[1] - 4*diff2_arr[3]*ref_arr[1] - diff2_arr[2]*ref_arr[2] - 8*diff_arr[2]*diff_arr[2] + 32*diff_arr[2]*diff_arr[1]*ref_arr[1] + 4*diff2_arr[2]*ref_arr[1]*ref_arr[1] + 4*ref_arr[2]*diff_arr[1]*diff_arr[1] - 24*diff_arr[1]*diff_arr[1]*ref_arr[1]*ref_arr[1]
               
        
def gamma_4(a,b,v,vpoi):
    ref_arr, diff_arr, diff2_arr = moments(v,vpoi)
    n1 = 0
    d1 = 0
    
    if(a == "u"):
        n1 = ref_arr[2]
        d1 = ref_arr[1]**2
    if(a == "w"):
        n1 = diff_arr[2]
        d1 = ref_arr[1]*diff_arr[1]
    if(a == "d"):
        n1 = diff2_arr[2]
        d1 = ref_arr[1]*diff2_arr[1]
    if(a == "b"):
        n1 = diff2_arr[2]
        d1 = diff_arr[1]**2
    if(b == "u"):
        n2 = ref_arr[2]
        d2 = ref_arr[1]**2
    if(b == "w"):
        n2 = diff_arr[2]
        d2 = ref_arr[1]*diff_arr[1]
    if(b == "d"):
        n2 = diff2_arr[2]
        d2 = ref_arr[1]*diff2_arr[1]
    if(b == "b"):
        n2 = diff2_arr[2]
        d2 = diff_arr[1]**2
        
    return ((n1/d1) - (n2/d2))

def gamma_6(a,b,v,vpoi):
    ref_arr, diff_arr, diff2_arr = moments(v,vpoi)
    n1 = 0
    d1 = 0
    
    if(a == "u"):
        n1 = ref_arr[3]
        d1 = ref_arr[1]**2*ref_arr[1]
    if(a == "w"):
        n1 = diff_arr[3]
        d1 = ref_arr[1]*diff_arr[1]*ref_arr[1]
    if(a == "d"):
        n1 = diff2_arr[3]
        d1 = ref_arr[1]*diff2_arr[1]*ref_arr[1]
    if(a == "b"):
        n1 = diff2_arr[3]
        d1 = diff_arr[1]**2*ref_arr[1]
    if(b == "u"):
        n2 = ref_arr[3]
        d2 = ref_arr[1]**2*ref_arr[1]
    if(b == "w"):
        n2 = diff_arr[3]
        d2 = ref_arr[1]*diff_arr[1]*ref_arr[1]
    if(b == "d"):
        n2 = diff2_arr[3]
        d2 = ref_arr[1]*diff2_arr[1]*ref_arr[1]
    if(b == "b"):
        n2 = diff2_arr[3]
        d2 = diff_arr[1]**2*ref_arr[1]
        
    return -1*((n1/d1) - (n2/d2))

def gamma_8(a,b,v,vpoi):
    ref_arr, diff_arr, diff2_arr = moments(v,vpoi)
    n1 = 0
    d1 = 0
    
    if(a == "u"):
        n1 = ref_arr[4]
        d1 = ref_arr[1]**2*ref_arr[1]**2
    if(a == "w"):
        n1 = diff_arr[4]
        d1 = ref_arr[1]*diff_arr[1]*ref_arr[1]**2
    if(a == "d"):
        n1 = diff2_arr[4]
        d1 = ref_arr[1]*diff2_arr[1]*ref_arr[1]**2
    if(a == "b"):
        n1 = diff2_arr[4]
        d1 = diff_arr[1]**2*ref_arr[1]**2
    if(b == "u"):
        n2 = ref_arr[4]
        d2 = ref_arr[1]**2*ref_arr[1]**2
    if(b == "w"):
        n2 = diff_arr[4]
        d2 = ref_arr[1]*diff_arr[1]*ref_arr[1]**2
    if(b == "d"):
        n2 = diff2_arr[4]
        d2 = ref_arr[1]*diff2_arr[1]*ref_arr[1]**2
    if(b == "b"):
        n2 = diff2_arr[4]
        d2 = diff_arr[1]**2*ref_arr[1]**2
        
    return ((n1/d1) - (n2/d2))


def ratio_4(a,b,v,vpoi):
    ref_arr, diff_arr, diff2_arr = moments(v,vpoi)
    n1 = 0
    d1 = 0
    
    if(a == "u"):
        n1 = ref_arr[2]
        d1 = ref_arr[1]**2
    if(a == "w"):
        n1 = diff_arr[2]
        d1 = ref_arr[1]*diff_arr[1]
    if(a == "d"):
        n1 = diff2_arr[2]
        d1 = ref_arr[1]*diff2_arr[1]
    if(a == "b"):
        n1 = diff2_arr[2]
        d1 = diff_arr[1]**2
    if(b == "u"):
        n2 = ref_arr[2]
        d2 = ref_arr[1]**2
    if(b == "w"):
        n2 = diff_arr[2]
        d2 = ref_arr[1]*diff_arr[1]
    if(b == "d"):
        n2 = diff2_arr[2]
        d2 = ref_arr[1]*diff2_arr[1]
    if(b == "b"):
        n2 = diff2_arr[2]
        d2 = diff_arr[1]**2
        
    return ((n2/d2)/(n1/d1))
    

def ratio_6(a,b,v,vpoi):
    ref_arr, diff_arr, diff2_arr = moments(v,vpoi)
    n1 = 0
    d1 = 0
    
    if(a == "u"):
        n1 = ref_arr[3]
        d1 = ref_arr[1]**2*ref_arr[1]
    if(a == "w"):
        n1 = diff_arr[3]
        d1 = ref_arr[1]*diff_arr[1]*ref_arr[1]
    if(a == "d"):
        n1 = diff2_arr[3]
        d1 = ref_arr[1]*diff2_arr[1]*ref_arr[1]
    if(a == "b"):
        n1 = diff2_arr[3]
        d1 = diff_arr[1]**2*ref_arr[1]
    if(b == "u"):
        n2 = ref_arr[3]
        d2 = ref_arr[1]**2*ref_arr[1]
    if(b == "w"):
        n2 = diff_arr[3]
        d2 = ref_arr[1]*diff_arr[1]*ref_arr[1]
    if(b == "d"):
        n2 = diff2_arr[3]
        d2 = ref_arr[1]*diff2_arr[1]*ref_arr[1]
    if(b == "b"):
        n2 = diff2_arr[3]
        d2 = diff_arr[1]**2*ref_arr[1]
        
    return ((n2/d2)/(n1/d1))


def ratio_8(a,b,v,vpoi):
    ref_arr, diff_arr, diff2_arr = moments(v,vpoi)
    n1 = 0
    d1 = 0
    
    if(a == "u"):
        n1 = ref_arr[4]
        d1 = ref_arr[1]**2*ref_arr[1]**2
    if(a == "w"):
        n1 = diff_arr[4]
        d1 = ref_arr[1]*diff_arr[1]*ref_arr[1]**2
    if(a == "d"):
        n1 = diff2_arr[4]
        d1 = ref_arr[1]*diff2_arr[1]*ref_arr[1]**2
    if(a == "b"):
        n1 = diff2_arr[4]
        d1 = diff_arr[1]**2*ref_arr[1]**2
    if(b == "u"):
        n2 = ref_arr[4]
        d2 = ref_arr[1]**2*ref_arr[1]**2
    if(b == "w"):
        n2 = diff_arr[4]
        d2 = ref_arr[1]*diff_arr[1]*ref_arr[1]**2
    if(b == "d"):
        n2 = diff2_arr[4]
        d2 = ref_arr[1]*diff2_arr[1]*ref_arr[1]**2
    if(b == "b"):
        n2 = diff2_arr[4]
        d2 = diff_arr[1]**2*ref_arr[1]**2
        
    return (n2/d2)/(n1/d1)

def quantity(string,v,vpoi):
    if(string == "SC_u"):
        return(SC_u(v,vpoi))
    if(string == "M3_u"):
        return(M3_u(v,vpoi))
    if(string == "MHC3_u"):
        return(MHC3_u(v,vpoi))
    if(string == "M4_u"):
        return(M4_u(v,vpoi))
    if(string == "AC4_u"):
        return(AC4_u(v,vpoi))
    if(string == "MHC4_u"):
        return(MHC4_u(v,vpoi))
    if(string == "SC_w"):
        return(SC_w(v,vpoi))
    if(string == "M3_w"):
        return(M3_w(v,vpoi))
    if(string == "MHC3_w"):
        return(MHC3_w(v,vpoi))
    if(string == "M4_w"):
        return(M4_w(v,vpoi))
    if(string == "AC4_w"):
        return(AC4_w(v,vpoi))
    if(string == "MHC4_w"):
        return(MHC4_w(v,vpoi))
    if(string == "SC_d"):
        return(SC_d(v,vpoi))
    if(string == "M3_d"):
        return(M3_d(v,vpoi))
    if(string == "MHC3_d"):
        return(MHC3_d(v,vpoi))
    if(string == "M4_d"):
        return(M4_d(v,vpoi))
    if(string == "AC4_d"):
        return(AC4_d(v,vpoi))
    if(string == "MHC4_d"):
        return(MHC4_d(v,vpoi))
    if(string == "SC_b"):
        return(SC_b(v,vpoi))
    if(string == "M3_b"):
        return(M3_b(v,vpoi))
    if(string == "MHC3_b"):
        return(MHC3_b(v,vpoi))
    if(string == "M4_b"):
        return(M4_b(v,vpoi))
    if(string == "AC4_b"):
        return(AC4_b(v,vpoi))
    if(string == "MHC4_b"):
        return(MHC4_b(v,vpoi))
    
    if(string == "Gamma4_uw"):
        return(gamma_4("u","w",v,vpoi))
    if(string == "Gamma4_ud"):
        return(gamma_4("u","d",v,vpoi))
    if(string == "Gamma4_ub"):
        return(gamma_4("u","b",v,vpoi))
    if(string == "Gamma4_wd"):
        return(gamma_4("w","d",v,vpoi))
    if(string == "Gamma4_wb"):
        return(gamma_4("w","b",v,vpoi))
    if(string == "Gamma4_db"):
        return(gamma_4("d","b",v,vpoi))
    
    if(string == "Gamma6_uw"):
        return(gamma_6("u","w",v,vpoi))
    if(string == "Gamma6_ud"):
        return(gamma_6("u","d",v,vpoi))
    if(string == "Gamma6_ub"):
        return(gamma_6("u","b",v,vpoi))
    if(string == "Gamma6_wd"):
        return(gamma_6("w","d",v,vpoi))
    if(string == "Gamma6_wb"):
        return(gamma_6("w","b",v,vpoi))
    if(string == "Gamma6_db"):
        return(gamma_6("d","b",v,vpoi))
    
    if(string == "Gamma8_uw"):
        return(gamma_6("u","w",v,vpoi))
    if(string == "Gamma8_ud"):
        return(gamma_6("u","d",v,vpoi))
    if(string == "Gamma8_ub"):
        return(gamma_6("u","b",v,vpoi))
    if(string == "Gamma8_wd"):
        return(gamma_6("w","d",v,vpoi))
    if(string == "Gamma8_wb"):
        return(gamma_6("w","b",v,vpoi))
    if(string == "Gamma8_db"):
        return(gamma_6("d","b",v,vpoi))
    
    if(string == "Ratio4_uw"):
        return(ratio_4("u","w",v,vpoi))
    if(string == "Ratio4_ud"):
        return(ratio_4("u","d",v,vpoi))
    if(string == "Ratio4_ub"):
        return(ratio_4("u","b",v,vpoi))
    if(string == "Ratio4_wd"):
        return(ratio_4("w","d",v,vpoi))
    if(string == "Ratio4_wb"):
        return(ratio_4("w","b",v,vpoi))
    if(string == "Ratio4_db"):
        return(ratio_4("d","b",v,vpoi))
    
    if(string == "Ratio6_uw"):
        return(ratio_6("u","w",v,vpoi))
    if(string == "Ratio6_ud"):
        return(ratio_6("u","d",v,vpoi))
    if(string == "Ratio6_ub"):
        return(ratio_6("u","b",v,vpoi))
    if(string == "Ratio6_wd"):
        return(ratio_6("w","d",v,vpoi))
    if(string == "Ratio6_wb"):
        return(ratio_6("w","b",v,vpoi))
    if(string == "Ratio6_db"):
        return(ratio_6("d","b",v,vpoi))
    
    if(string == "Ratio8_uw"):
        return(ratio_6("u","w",v,vpoi))
    if(string == "Ratio8_ud"):
        return(ratio_6("u","d",v,vpoi))
    if(string == "Ratio8_ub"):
        return(ratio_6("u","b",v,vpoi))
    if(string == "Ratio8_wd"):
        return(ratio_6("w","d",v,vpoi))
    if(string == "Ratio8_wb"):
        return(ratio_6("w","b",v,vpoi))
    if(string == "Ratio8_db"):
        return(ratio_6("d","b",v,vpoi))

def Cumulants(a,b): 

    v2 = np.sqrt(np.mean(a*a))
    v2p = np.mean(a*b)/np.sqrt(np.mean(a*a))
        
    v4 =  (2*np.mean(a*a)**2 - np.mean(a*a*a*a))**(1/4)
    v4p = (2*np.mean(a*b)*np.mean(a*a) - np.mean(a*a*a*b))/((2*np.mean(a*a)**2 - np.mean(a*a*a*a))**(3/4))
        
    return v2, v2p, v4, v4p
    
def Variance_ratios(a,b):

    
    sigma1 = np.var(b)/np.var(a) 
    sigma2 = np.var(b*b)/np.var(a*a)
    sigma3 = np.var(a*b)/np.var(a*a) 
    #sigma4 = np.var(np.sqrt(a*a)) - np.var(a*b)
    
    return sigma1, sigma2, sigma3

def Variance_diffs(a,b):
    
    sigma1 = np.var(a) - np.var(b)
    sigma2 = np.var(a*a) - np.var(b*b)
    sigma3 = np.var(a*a) - np.var(a*b)
    #sigma4 = np.var(np.sqrt(a*a)) - np.var(a*b)
    
    return sigma1, sigma2, sigma3
