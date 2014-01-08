import scipy as SP
from covar_base import CovarianceFunction

class FixedCF(CovarianceFunction):
    """
    fixed kernel (without scaling)
    """

    def __init__(self,K,n_dimensions):
        self.n_dimensions = n_dimensions
        self._K = K
        self.n_hyperparameters = 1

    def K(self,theta,X,X2=None):
        """
        evaluates the kernel for given hyperparameters theta and inputs X
        """
        assert X.shape[1]==self.n_dimensions, 'dimensions do not match'
        if X2!=None:
            assert X2.shape[1]==self.n_dimensions,'dimensions do not match'
        else:
            X2 = X

        if X.shape==X2.shape:
            A = SP.exp(2*theta[0])
            return A*self._K
        else:
            # kernel unknown...
            return SP.zeros((X.shape[0],X2.shape[0]))

    def Kgrad_theta(self,theta,X,i):
        assert i==0, 'unknown hyperparameter'
        RV = self.K(theta,X)
        return 2*RV

    def Kgrad_x(self,theta,X,n=None,d=None):
        return SP.zeros((X.shape[0],X.shape[0]))
