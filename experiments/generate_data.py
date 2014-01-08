import scipy as SP
import pdb
import os
from utils import getVariance,getVarianceKron
import h5py
import scipy.linalg as LA
import utils

import sys
sys.path.append('..')

import core.gp.gp_kronprod as gp_kronprod
import core.covariance.linear as linear
from core.gp.gp_kronprod import ravel,unravel

def sim_linear_kernel(X=None,N=None,n_dim=None,theta=None):
    """
    simulate positive definite kernel
    """
    if X==None:
        X = SP.random.randn(N,n_dim)
    else:
        N = X.shape[0]
        n_dim = X.shape[1]
 
    if theta==None:
        theta = SP.random.randn(1)

    cf = linear.LinearCF(n_dim)
    cf.X = X
    K = cf.K(theta)
    
    return K,X

def generate_multiKron(C,R):
    S_c,U_c = LA.eigh(C+1E-6*SP.eye(C.shape[0]))
    S_r,U_r = LA.eigh(R+1E-6*SP.eye(R.shape[0]))
    US_c = SP.sqrt(S_c) * U_c
    US_r = SP.sqrt(S_r) * U_r
    # kron(US_c,US_r) vec(Y) = vec(US_r.T*Y*US_c)
    Y = SP.random.randn(R.shape[0],C.shape[0])
    Y = SP.dot(US_r,SP.dot(Y,US_c.T))
    return Y


def generate_datasetHIDDEN(fn,f_causal,f_common,f_hidden,N,D,n_r=None,n_omega=None,n_repeats=1):
    """
    generate dataset for simulation studies

    var_observed:  variance explained by observed features
    f_common: fraction of common effects
    f_hidden: fraction of hidden effects
    N: number of samples
    D: number of traits
    n_r: number of features in R
    n_c: number of features in C
    n_sigma: number of features in Sigma
    n_omega: number of features in Omega
    n_repeats: number of repeats

    Y = Yobserved + Yhidden + Ynoise,
    Yobserved = Ycommon + Yindependent
    """
    if n_r==None:
        n_r = N
    if n_omega==None:
        n_omega=N

    f = h5py.File(fn,'w')

    for rep_id in range(n_repeats):
        # simulate observed,common effect
        scaling = SP.diag(SP.random.randn(D))
        X_r = SP.random.randn(N,n_r)
        K_r = SP.dot(X_r,X_r.T)
        w_common = SP.random.randn(n_r,1)
        Ycommon = SP.dot(X_r,w_common)
        Ycommon = SP.tile(Ycommon,(1,D))
        Ycommon = SP.dot(Ycommon,scaling)
        Ycommon *= SP.sqrt(f_common)/SP.sqrt(Ycommon.var(0).mean())
        
        # simulate observed,independent effect
        Yind = SP.zeros((N,D))
        for i in range(D):
            w_ind = SP.random.randn(n_r)
            Yind[:,i] = SP.dot(X_r,w_ind)
            
        Yind *= SP.sqrt(1-f_common)/SP.sqrt(Yind.var(0).mean())

        # merge together to observed effect
        Yobserved = Ycommon + Yind
        Yobserved*=  SP.sqrt(1-f_hidden)/SP.sqrt(Yobserved.var(0).mean())

        # simulate hidden, common effect
        scaling = SP.diag(SP.random.randn(D))
        X_omega = SP.random.randn(N,n_omega)
        w_common = SP.random.randn(n_omega,1)
        Ycommon = SP.dot(X_omega,w_common)
        Ycommon = SP.tile(Ycommon,(1,D))
        Ycommon = SP.dot(Ycommon,scaling)
        Ycommon *= SP.sqrt(f_common)/SP.sqrt(Ycommon.var(0).mean())

        # simulate hidden, independent effect
        Yind = SP.zeros((N,D))
        for i in range(D):
            w_ind = SP.random.randn(n_omega)
            Yind[:,i] = SP.dot(X_omega,w_ind)
        Yind *= SP.sqrt(1-f_common)/SP.sqrt(Yind.var(0).mean())
        

        # merge together
        Yhidden = Ycommon + Yind
        Yhidden*= SP.sqrt(f_hidden)/SP.sqrt(Yhidden.var(0).mean())
        Ycausal = Yhidden + Yobserved
        Ycausal*= SP.sqrt(f_causal)#/Ycausal.std(0)

        # simulate iid noise
        Ynoise = SP.random.randn(N,D)
        Ynoise *= SP.sqrt(1-f_causal)/SP.sqrt(Ynoise.var(0).mean())
        
        Y = Ycausal + Ynoise
        
        RV = {'Yhidden':Yhidden,'Ynoise':Ynoise,'X_r':X_r,'f_causal':f_causal,'f_common':f_common,'f_hidden':f_hidden,'N':N,'D':D,'n_r':n_r,'n_omega':n_omega,'X_omega':X_omega,'Y':Y,'Ycausal':Ycausal,'Yobserved':Yobserved}
        out = f.create_group('rep%d'%rep_id)
        utils.storeHashHDF5(out,RV)

    f.close()
    return RV

 

def generate_datasetLIN(var_signal,N,D,fn,n_r=None,n_c=1,n_sigma=1,n_repeats=30):
    """ 
    generate dataset for simulation studies (linear kernel)

    var_signal: percentage of variance explained by the kron(C,R)
    
    N: number of observations
    D: number of phenotypes
    K: number of features
    n_c/n_r/n_sigma: number of dimensions (default: 1)
    n_repeats: number of repetitions
    """
    if n_r==None:
        n_r = N

    thetaLin = SP.array([0])

    f = h5py.File(fn,'w')  
    for i in range(n_repeats):
        R,X_r = sim_linear_kernel(N=N,n_dim=n_r,theta=thetaLin)
        scaleR = utils.getVariance(R)
        R /= scaleR
        X_r /= SP.sqrt(scaleR)

        Omega = SP.eye(N)
        C,X_c = sim_linear_kernel(N=D,n_dim=n_c,theta=thetaLin) 
        scaleC = getVariance(C)
        X_c  /= SP.sqrt(scaleC)
        C   /= scaleC

        Sigma,X_sigma = sim_linear_kernel(N=D,n_dim=n_sigma,theta=thetaLin) 
        scaleSigma = getVariance(Sigma)
        Sigma /= scaleSigma
        X_sigma /= SP.sqrt(scaleSigma)
        Sigma /= scaleSigma
        
        Ysignal = generate_multiKron(C,R)
        Ynoise = generate_multiKron(Sigma,Omega)
        Ysignal *= SP.sqrt(var_signal)/Ysignal.std()
        Ynoise *= SP.sqrt(1-var_signal)/Ynoise.std()
        Yvec = Ysignal + Ynoise
        
        Y = unravel(Yvec,N,D)
        Ysignal = unravel(Ysignal,N,D)
        Ynoise = unravel(Ynoise,N,D)
        
        RV = {'Y':Y,'C':C,'R':R,'Sigma':Sigma,'Omega':Omega,'X_c':X_c,'X_r':X_r,'X_sigma':X_sigma,'n_c':n_c,'n_r':n_r,'n_sigma':n_sigma,'D':D,'N':N,'var_signal':var_signal,'Ysignal':Ysignal,'Ynoise':Ynoise}

        out = f.create_group('rep%d'%i)
        utils.storeHashHDF5(out,RV)
    f.close()
    return RV
    
if __name__ == '__main__':
    # Notation: C: low-rank genetic correlation kernel, R: population kernel, Sigma: low-rank correlation noise, Omega: noise kerne

    sim_dir = '../sim_data/'
    if not(os.path.exists(sim_dir)):
        print 'creating directory sim_data'
        os.makedirs(sim_dir)
        

    # hidden variables scenario
    if 1:
        SP.random.seed(1)
        N = 200 # number of samples
        D = 10  # number of traits
        n_r = 200 # number of features (R)
        n_omega = 200 # number of features (Omega)

        for f_causal in [0.9]:
            for f_common in [0.0,0.1,0.3,0.5,0.7,0.9,1.0]:
                for f_hidden in [0.5]:
                    fn = os.path.join(sim_dir,'sim_N%d_D%d_causal%02d_common%02d_hidden%02d.hdf5'%(N,D,10*f_causal,10*f_common,10*f_hidden))
                    generate_datasetHIDDEN(fn,f_causal,f_common,f_hidden,N,D,n_r=n_r,n_omega=n_omega,n_repeats=30)

        for f_causal in [0.9]:
            for f_common in [0.9]:
                for f_hidden in [0.0,0.1,0.3,0.5,0.7,0.9,1]:
                    fn = os.path.join(sim_dir,'sim_N%d_D%d_causal%02d_common%02d_hidden%02d.hdf5'%(N,D,10*f_causal,10*f_common,10*f_hidden))
                    generate_datasetHIDDEN(fn,f_causal,f_common,f_hidden,N,D,n_r=n_r,n_omega=n_omega,n_repeats=30)

        for f_causal in [0.1,0.3,0.5,0.7,0.9,1.0]:
            for f_common in [0.9]:
                for f_hidden in [0.5]:
                    fn = os.path.join(sim_dir,'sim_N%d_D%d_causal%02d_common%02d_hidden%02d.hdf5'%(N,D,10*f_causal,10*f_common,10*f_hidden))
                    generate_datasetHIDDEN(fn,f_causal,f_common,f_hidden,N,D,n_r=n_r,n_omega=n_omega,n_repeats=30)


    # runtime comparison
    if 1:
        SP.random.seed(1)
        N = 100 # number of samples
        D = 2  # number of traits
        var_signal = 0.5
     
        for N in [16,32,64,128,256]:
            for D in [16,32,64,128,256]:
                fn = os.path.join(sim_dir,'sim_runtime_signal%03d_N%d_D%d.hdf5'%(var_signal*1E3,N,D))
                generate_datasetLIN(var_signal,N,D,fn,n_r=1000,n_repeats=10)

  
