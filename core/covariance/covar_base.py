import scipy as SP

class CovarianceFunction(object):
    """
    abstract super class for all implementations of covariance functions
    """
    __slots__ = ['n_hyperparameters','n_dimensions','_X','_Xcross']

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        assert self.n_dimensions==X.shape[1], 'dimensions do not match'
        self._X = X

    @property
    def Xcross(self):
        return self._Xcross

    @Xcross.setter
    def Xcross(self, Xcross):
        assert self.n_dimensions==Xcross.shape[1], 'dimensions do not match'
        self._Xcross = Xcross
        
    @property
    def n(self):
        # number of training points
        return self.X.shape[0]

    @property
    def n_cross(self):
        # number of test points
        return self.Xcross.shape[0]
    
    def K(self,theta):
        """
        evaluates the kernel for given hyperparameters theta and inputs X
        """
        raise Exception("%s: Function K not yet implemented"%(self.__class__))

    def Kcross(self,theta):
        """
        evaluates the kernel for given hyperparameters theta between the training samples X1 and the test samples X2
        """
        raise Exception("%s: Function K not yet implemented"%(self.__class__))
        
    def Kgrad_theta(self,theta,i):
        """
        partial derivative with repspect to the i-th hyperparamter theta[i]
        """
        raise Exception("%s: Function K not yet implemented"%(self.__class__))

    def Kgrad_x(self,theta,d,n=None):
        """
        partial derivative with respect to X[n,d], if n is set to None with respect to
        the hidden factor X[:,d]
        """
        raise Exception("%s: Function K not yet implemented"%(self.__class__))

    
