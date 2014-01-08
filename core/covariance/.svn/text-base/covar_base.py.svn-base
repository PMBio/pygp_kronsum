import scipy as SP

class CovarianceFunction:
    """
    abstract super class for all implementations of covariance functions
    """
    __slots__ = ['n_hyperparameters','n_dimensions','params_mask']

    def K(self,theta,X1,X2=None):
        """
        evaluates the kernel for given hyperparameters theta and inputs X
        """
        LG.critical("implement K")
        print("%s: Function K not yet implemented"%(self.__class__))
        return None
     
    def Kgrad_theta(self,theta,X,i):
        """
        partial derivative with repspect to the i-th hyperparamter theta[i]
        """
        LG.critical("implement K")
        print("%s: Function K not yet implemented"%(self.__class__))
        return None

    def Kgrad_x(self,theta,X,n=None,d=None):
        """
        partial derivative with respect to X[n,d], if n is set to None with respect to
        the hidden factor X[:,d]
        """
        LG.critical("implement K")
        print("%s: Function K not yet implemented"%(self.__class__))
        return None

    
