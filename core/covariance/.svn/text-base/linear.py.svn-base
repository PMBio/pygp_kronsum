"""
Classes for linear covariance function
======================================
Linear covariance functions

LinearCF
"""
import scipy as SP
import pdb
from covar_base import CovarianceFunction

class LinearCF(CovarianceFunction):
    """
    isotropic linear covariance function with a single hyperparameter
    """
    def __init__(self,n_dimensions):
        self.n_dimensions = n_dimensions
        self.n_hyperparameters = 1
        self._covar_cache = None
        self._use_cache = False
        

    def _is_cached(self):
        if self._covar_cache is None:
            return False
        if 'XX' in self._covar_cache:
            return True

    def reset_cache(self):
        self._covar_cache = None

    def use_cache(self,use_cache):
        self.reset_cache()
        self._use_cache = use_cache
        
    def _get_XX(self,X,X2):
        """
        check if we have to recompute XX (expensive if X has many dimensions)
        """
        if self._is_cached():
            return self._covar_cache['XX']

        self._covar_cache = {}
        self._covar_cache['XX'] = SP.dot(X,X2.T)
        return self._covar_cache['XX']
    
    def K(self,theta,X,X2=None,hack=True):
        """
        evaluates the kernel
        """
        assert X.shape[1]==self.n_dimensions, 'dimensions do not match'
        if X2!=None:
            assert X2.shape[1]==self.n_dimensions,'dimensions do not match'
        else:
            X2 = X
        
        #pdb.set_trace()
        if self._use_cache:
            XX = self._get_XX(X,X2)
        else:
            XX = SP.dot(X,X2.T)
            
        A = SP.exp(2*theta[0])
        RV = A * XX

        if hack and RV.shape[0]==RV.shape[1]:
            # hack to get full rank matrix
            RV += A*1E-4*SP.eye(RV.shape[0])
        return RV

    def Kgrad_theta(self,theta,X,i,hack=True):
        """
        evaluates the gradient with respect to the hyperparameter theta
        """
        assert i==0, 'unknown hyperparameter'
        assert X.shape[1]==self.n_dimensions, 'dimensions do not match'
        K = self.K(theta,X,hack=hack)
        return 2*K

    def Kgrad_x(self,theta,X,n=None,d=None):
        """
        evaluates the gradient 
        """
        assert X.shape[1]==self.n_dimensions, 'dimensions do not match'
        
        A = SP.exp(2*theta[0])
        if n!=None:
            # gradient with respect to X[n,d]
            XX = SP.zeros((X.shape[0],X.shape[0]))
            XX[:,n] = X[:,d]
            XX[n,:] = X[:,d]
            XX[n,n] *= 2
        else:
            # gradient with respect to X[:,d], but assuming K = XY^T (not symmetric!)
            Xd = SP.reshape(X[:,d],(X.shape[0],1))
            XX = SP.tile(Xd,X.shape[0])
        return A*(XX)


