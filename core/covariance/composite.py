import scipy as SP
import pdb
from covar_base import CovarianceFunction
from linear import LinearCF
from diag import DiagIsoCF,DiagArdCF

class SumCF(CovarianceFunction):
    """
    Sum of Covariance FUnctions (all covariance functions have the same features)
    """

    def __init__(self,n_dimensions):
        self._covar_list = []
        self.n_hyperparameters = 0
        self.n_dimensions = n_dimensions
        
    def append_covar(self,covar):
        self._covar_list.append(covar)
        self.n_hyperparameters += covar.n_hyperparameters

    @property
    def n_kernel(self):
        return len(self._Klist)

    @property
    def covar_list(self):
        return self._covar_list
    
    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        assert self.n_dimensions==X.shape[1], 'dimensions do not match'
        self._X = X
        for covar in self.covar_list:
            covar.X = X
        
    @property
    def Xcross(self):
        return self._Xcross

    @Xcross.setter
    def Xcross(self, Xcross):
        assert self.n_dimensions==Xcross.shape[1], 'dimensions do not match'
        self._Xcross = Xcross
        for covar in self.covar_list:
            covar.Xcross = Xcross
            
    def K(self,theta):
        """
        evaluates the kernel for given hyperparameters theta and inputs X
        """
        assert len(theta)==self.n_hyperparameters, 'ouch, number of hyperparameters does not match'
        
        K = SP.zeros((self.n,self.n))
        i_start=0
        for covar in self.covar_list:
            i_stop = i_start + covar.n_hyperparameters
            _theta = theta[i_start:i_stop]
            K += covar.K(_theta)
            i_start = i_stop
        return K

    def Kcross(self,theta):
        """
        evaluates the kernel for given hyperparameters theta and inputs X
        """
        assert len(theta)==self.n_hyperparameters, 'ouch, number of hyperparameters does not match'
        
        Kcross = SP.zeros((self.n,self.n_cross))
        i_start=0
        for covar in self.covar_list:
            i_stop = i_start + covar.n_hyperparameters
            _theta = theta[i_start:i_stop]
            Kcross += covar.Kcross(_theta)
            i_start = i_stop
        return Kcross


    def Kgrad_theta(self,theta,i):
        """
        partial derivative with repspect to the i-th hyperparamter theta[i]
        """
        assert len(theta)==self.n_hyperparameters, 'ouch, number of hyperparameters does not match'
        assert i<self.n_hyperparameters, 'unknown hyperparameter'

        i_start = 0
        for covar in self.covar_list:
            i_stop = i_start + covar.n_hyperparameters

            if i_start<=i and i<i_stop:
                _theta = SP.array(theta[i_start:i_stop])
                return covar.Kgrad_theta(_theta,i-i_start)

            i_start = i_stop
        

    def Kgrad_x(self,theta,d,n=None):
        """
        partial derivative with respect to X[n,d], if n is set to None with respect to
        the hidden factor X[:,d]
        """
        Kgrad_x = SP.zeros((self.n,self.n))
        i_start = 0
        for covar in self.covar_list:
            i_stop = i_start + covar.n_hyperparameters
            _theta = theta[i_start:i_stop]
            Kgrad_x += covar.Kgrad_x(_theta,d,n)
            i_start = i_stop
            
        return Kgrad_x
