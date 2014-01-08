"""
Classes for linear covariance function
======================================
Squared Exponential Kernel

SqExp
"""

import scipy as SP
import pdb
import sys
from covar_base import CovarianceFunction
import scipy.spatial.distance as DIST

sys.path.append('../linalg')
import dist

class SqExpCF(CovarianceFunction):
    """
    Standard Squared Exponential Covariance Function (same length-scale for all input dimensions)
    """
    def __init__(self,n_dimensions):
        self.n_dimensions = n_dimensions
        self.n_hyperparameters = 2
  
    def K(self,theta,X,X2=None):
        """
        evaluates the kernel
        """
        assert X.shape[1]==self.n_dimensions, 'dimensions do not match'
        if X2!=None:
            assert X2.shape[1]==self.n_dimensions,'dimensions do not match'
        else:
            X2 = X
        A = SP.exp(2*theta[0])
        L = SP.exp(theta[1])
        sqd = dist.sq_dist(X/L,X2/L)
        RV = A * SP.exp(-0.5*sqd)
        return RV

    def Kgrad_theta(self,theta,X,i):
        """
        evaluates the gradient with respect to the hyperparameter theta
        """
        assert i<2, 'unknown hyperparameter'
        assert X.shape[1]==self.n_dimensions, 'dimensions do not match'

        A = SP.exp(2*theta[0])
        L = SP.exp(theta[1])
        sqd = dist.sq_dist(X/L)
        K = A * SP.exp(-0.5*sqd)
        
        if i==0:
            return 2*K
        if i==1:
            return K*sqd


    
