import scipy as SP
import pdb
from covar_base import CovarianceFunction
from linear import LinearCF
from diag import DiagIsoCF,DiagArdCF
from composite import SumCF

class LowRankCF(SumCF):
    """
    Low Rank Covariance Function (linear covariance plus isotropic covariance)
    """
    __slots__ = ['n_hyperparameters','n_dimensions','_X','_Xcross','covar_iso','covar_lin']
    
    def __init__(self,n_dimensions):
        self.n_hyperparameters = 2
        super(LowRankCF,self).__init__(n_dimensions)
        covar_lin = LinearCF(n_dimensions)
        covar_iso = DiagIsoCF(n_dimensions)
        self.append_covar(covar_lin)
        self.append_covar(covar_iso)

    @property
    def covar_lin(self):
        return self.covar_list[0]

    @property
    def covar_iso(self):
        return self.covar_list[1]

    def Kgrad_x(self,theta,d,n=None,**kwargs):
        """
        partial derivative with respect to X[n,d], if n is set to None with respect to
        the hidden factor X[:,d]
        """
        return self.covar_lin.Kgrad_x(theta[:self.covar_lin.n_hyperparameters],d,n=n,**kwargs)


class LowRankArdCF(SumCF):
    """
    Low Rank Covariance Function (linear covariance plus diagonal)
    """
    def __init__(self,n_dimensions,n_hyperparameters):
        # number of hyperparams is number of samples + 1 (1 for linear kernel)
        self.n_hyperparameters = n_hyperparameters
        super(LowRankArdCF,self).__init__(n_dimensions)
        covar_lin = LinearCF(n_dimensions)
        covar_diag = DiagArdCF(n_dimensions=n_dimensions,n_hyperparameters=n_hyperparameters-1)
        self.append_covar(covar_lin)
        self.append_covar(covar_diag)

    @property
    def covar_lin(self):
        return self.covar_list[0]

    @property
    def covar_diag(self):
        return self.covar_list[1]

