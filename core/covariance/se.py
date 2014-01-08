import scipy as SP
import pdb
import sys
from covar_base import CovarianceFunction
import scipy.spatial.distance as DIST
import core.linalg.dist as dist

class SqExpCF(CovarianceFunction):
    """
    Standard Squared Exponential Covariance Function (same length-scale for all input dimensions)
    """
    def __init__(self,n_dimensions):
        self.n_dimensions = n_dimensions
        self.n_hyperparameters = 2

    @property
    def X(self):
        return self._X
    
    @property
    def Xcross(self):
        return self._Xcross
    
    @X.setter
    def X(self, X):
        assert self.n_dimensions==X.shape[1], 'dimensions do not match'
        self._X = X
        self._sqDist = dist.sq_dist(X,X)

    @Xcross.setter
    def Xcross(self, Xcross):
        assert self.n_dimensions==Xcross.shape[1], 'dimensions do not match'
        self._Xcross = Xcross
        self._sqDistCross = dist.sq_dist(self.X,Xcross)
        
    def K(self,theta):
        """
        evaluates the kernel
        """
        A = SP.exp(2*theta[0])
        L = SP.exp(2*theta[1])
        RV = A * SP.exp(-0.5*self._sqDist/L)
        return RV

    def Kcross(self,theta):
        """
        evaluates the cross covariance
        """
        A = SP.exp(2*theta[0])
        L = SP.exp(2*theta[1])
        RV = A * SP.exp(-0.5*self._sqDistCross/L)
        return RV
    
    def Kgrad_theta(self,theta,i):
        """
        evaluates the gradient with respect to the hyperparameter theta
        """
        assert i<2, 'unknown hyperparameter'
        A = SP.exp(2*theta[0])
        L = SP.exp(2*theta[1])
        K = A * SP.exp(-0.5*self._sqDist/L)
        
        if i==0:
            return 2*K
        if i==1:
            return K*self._sqDist/L


    
