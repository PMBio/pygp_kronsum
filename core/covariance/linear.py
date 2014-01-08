import scipy as SP
import pdb
from covar_base import CovarianceFunction
import copy

class LinearCF(CovarianceFunction):
    """
    linear covariance function with a single hyperparameter
    """
    __slots__ = ['n_hyperparameters','n_dimensions','_X','_Xcross','_XX','_XXcross']
    
    def __init__(self,n_dimensions):
        self.n_dimensions = n_dimensions
        self.n_hyperparameters = 1

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
        self._XX = SP.dot(X,X.T)

    @Xcross.setter
    def Xcross(self, Xcross):
        assert self.n_dimensions==Xcross.shape[1], 'dimensions do not match'
        self._Xcross = Xcross
        self._XXcross = SP.dot(self._X,Xcross.T)
        
    def K(self,theta):
        """
        evaluates the kernel
        """
        A = SP.exp(2*theta[0])
        RV = A * self._XX
        return RV

    def Kcross(self,theta):
        """
        evaluates the cross covariance
        """
        A = SP.exp(2*theta[0])
        RV = A * self._XXcross
        return RV


    def Kgrad_theta(self,theta,i):
        """
        partial derivative with repspect to the i-th hyperparamter theta[i]
        """
        assert i==0, 'unknown hyperparameter'

        K = self.K(theta)
        return 2*K

    def Kgrad_x(self,theta,d,n=None):
        """
        partial derivative with respect to X[n,d], if n is set to None with respect to
        the hidden factor X[:,d]
        """
        A = SP.exp(2*theta[0])
        if n!=None:
            # gradient with respect to X[n,d]
            XX = SP.zeros((self.n,self.n))
            XX[:,n] = self.X[:,d]
            XX[n,:] = self.X[:,d]
            XX[n,n] *= 2
            return A*(XX)
        else:
            # needed for faster computation of latent factors
            Xd = A*self.X[:,d,SP.newaxis]
            return Xd


