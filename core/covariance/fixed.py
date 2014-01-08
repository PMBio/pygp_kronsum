import scipy as SP
from covar_base import CovarianceFunction

class FixedCF(CovarianceFunction):
    """
    fixed covariance
    """

    def __init__(self,n_dimensions=None):
        self.n_dimensions = n_dimensions
        self.n_hyperparameters = 1
        self._K = None
        self._Kcross = None
        
    @property
    def _K(self):
        return self.__K
    
    @property
    def _Kcross(self):
        return self.__Kcross
    
    @_K.setter
    def _K(self, K):
        self.__K = K
    
    @_Kcross.setter
    def _Kcross(self, Kcross):
        self.__Kcross = Kcross

    @property
    def n(self):
        # number of training points
        return self._K.shape[0]

    @property
    def n_cross(self):
        # number of test points
        return self._Kcross.shape[1]

    def K(self,theta):
        """
        evaluates the kernel for given hyperparameters theta and inputs X
        """
        A = SP.exp(2*theta[0])
        return A*self._K
   
    def Kcross(self,theta):
        """
        evaluates the kernel for given hyperparameters theta between the training samples X1 and the test samples X2
        """
        A = SP.exp(2*theta[0])
        return A*self._Kcross


    def Kgrad_theta(self,theta,i):
        """
        partial derivative with repspect to the i-th hyperparamter theta[i]
        """
        assert i==0, 'unknown hyperparameter'
        RV = self.K(theta)
        return 2*RV

    def Kgrad_x(self,theta,d,n=None):
        """
        partial derivative with respect to X[n,d], if n is set to None with respect to
        the hidden factor X[:,d]
        """
        return SP.zeros((self.n, self.n))
