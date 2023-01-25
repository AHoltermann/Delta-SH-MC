
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


randy = ROOT.TRandom2()
randy.SetSeed(41)



def Gauss2d(x,par):
    f = 1/(2*np.pi*par[2]*par[3]*np.sqrt(1-(par[4]**2)))
    z = (((x[0]-par[0])/par[2])**2) + (((x[1]-par[1])/par[3])**2) - 2*par[4]*(x[0]-par[0])*(x[1]-par[1])/(par[2]*par[3])
    out = f*np.exp(-z/(2*(1-(par[4]**2))))
    return out

def BVGE2d(x,par):
    a0 = par[0]
    a1 = par[1]
    theta = par[2]

    #f = a1*np.exp(-x1)*pow((1-exp(-x1)),a1-1)*a2*exp(-x2)*pow((1-exp(-x2)),a2-1)*(1+theta*((1-2*pow(1-exp(-x1),a1))*(1-2*pow(1-exp(-x2),a2))));
    f = a0*np.exp(-x[0])*((1-np.exp(-x[0]))**(a0-1))*a1*np.exp(-x[1])*((1-np.exp(-x[1]))**(a1-1))
    s = (1+theta*(1-2*((1-np.exp(-x[0]))**a0))*(1-2*((1-np.exp(-x[1]))**a1)))
    return f*s
    


def datasets(num_samples,var_v,var_vpoi,covar_v,mean_v,mean_vpoi):

    method = 'eigenvectors'

    r = np.array([[var_v,abs(covar_v)],[covar_v,var_vpoi]])
         
    # Generate samples from three independent normally distributed random
    # variables (with mean 0 and std. dev. 1).
    x = norm.rvs(size=(2, num_samples))


    if method == 'cholesky':
        # Compute the Cholesky decomposition.
        c = cholesky(r, lower=True)
    else:
        # Compute the eigenvalues and eigenvectors.
        evals, evecs = eigh(r)
        # Construct c, so c*c^T = r.
        c = np.dot(evecs, np.diag(np.sqrt(evals)))

    # Convert the data to correlated random variables. 
    y = np.dot(c, x)

    y[0] += mean_v
    y[1] += mean_vpoi

    return np.array(y[0]),np.array(y[1])

def Gaussdata(n,mu_v,mu_vp,sig_v,sig_vp,rho,):
    
    a=0
    b=0
    
    Gauss = ROOT.TF2("Biv",Gauss2d,-1,2,-1,2,5)
    Gauss.SetParameter(0,mu_v)
    Gauss.SetParameter(1,mu_vp)
    Gauss.SetParameter(2,sig_v)
    Gauss.SetParameter(3,sig_vp)
    Gauss.SetParameter(4,rho)
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        a = ctypes.c_double(0)
        b = ctypes.c_double(0)
        Gauss.GetRandom2(a,b,randy)

        x[i] = a.value
        y[i] = b.value
        
    return np.array(x),np.array(y)

def bvgedata(n,a1,a2,theta):


    bvge = ROOT.TF2("BVGE",BVGE2d,0,10,0,10,3)
    bvge.SetParameter(0,a1)
    bvge.SetParameter(1,a2)
    bvge.SetParameter(2,theta)
    bvge.SetNpx(200)
    bvge.SetNpy(200)
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        a = ctypes.c_double(0)
        b = ctypes.c_double(0)
        bvge.GetRandom2(a,b,randy)

        c = a.value
        d = b.value     

        x[i] = np.sqrt(np.exp(-c))
        y[i] = np.sqrt(np.exp(-d))
    

    return np.array(x),np.array(y)


    