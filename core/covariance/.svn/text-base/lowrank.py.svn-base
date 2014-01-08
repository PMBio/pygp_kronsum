import scipy as SP
import pdb
from covar_base import CovarianceFunction
from linear import LinearCF
from diag import DiagIsoCF,DiagArdCF

class LowRankCF(CovarianceFunction):
    def __init__(self,n_dimensions):
        self.n_dimensions = n_dimensions
        self.n_hyperparameters = 2
        self.covar_lin = LinearCF(n_dimensions)
        self.covar_iso = DiagIsoCF(n_dimensions)
        
    def K(self,theta,X,X2=None):
        _theta = SP.array([theta[0]])
        Klin = self.covar_lin.K(_theta,X,X2,hack=False)
        _theta = SP.array([theta[1]])
        Kiso = self.covar_iso.K(_theta,X,X2)
        return Klin + Kiso

    def Kgrad_theta(self,theta,X,i):
        assert i<2, 'unknown hyperparameter'
        if i==0:
            _theta = SP.array([theta[0]])
            return self.covar_lin.Kgrad_theta(_theta,X,0,hack=False)
        if i==1:
            _theta = SP.array([theta[1]])
            return self.covar_iso.Kgrad_theta(_theta,X,0)

    def Kgrad_x(self,theta,X,n=None,d=None):
        _theta = SP.array([theta[0]])
        return self.covar_lin.Kgrad_x(_theta,X,n,d)




class LowRankArdCF(CovarianceFunction):
    def __init__(self,n_dimensions,n_hyperparameters):
        self.n_dimensions = n_dimensions
        self.n_hyperparameters = 1+n_hyperparameters
        self.covar_lin = LinearCF(n_dimensions)
        self.covar_diag = DiagArdCF(n_dimensions,n_hyperparameters)
        
    def K(self,theta,X,X2=None):
        _theta = SP.array([theta[0]])
        Klin = self.covar_lin.K(_theta,X,X2,hack=False)
        _theta = theta[1:]
        Kiso = self.covar_diag.K(_theta,X,X2)
        return Klin + Kiso

    def Kgrad_theta(self,theta,X,i):
        assert i<self.n_hyperparameters, 'unknown hyperparameter'
        if i==0:
            _theta = SP.array([theta[0]])
            return self.covar_lin.Kgrad_theta(_theta,X,0,hack=False)
        if i>0:
            _theta =theta[1:]
            return self.covar_diag.Kgrad_theta(_theta,X,0)

    def Kgrad_x(self,theta,X,n=None,d=None):
        _theta = SP.array([theta[0]])
        return self.covar_lin.Kgrad_x(_theta,X,n,d)
