import scipy as SP
import pdb
from covar_base import CovarianceFunction

class DiagIsoCF(CovarianceFunction):
    """
    isotropic linear covariance function with a single hyperparameter
    """
    def __init__(self,n_dimensions):
        self.n_dimensions = n_dimensions
        self.n_hyperparameters = 1
    
    def K(self,theta,X,X2=None):
        """
        evaluates the kernel
        """
        assert X.shape[1]==self.n_dimensions, 'dimensions do not match'
        if X2!=None:
            assert X2.shape[1]==self.n_dimensions,'dimensions do not match'
        else:
            X2 = X

        if X.shape!=X2.shape:
            return SP.zeros((X.shape[0],X2.shape[0]))
        
        A = SP.exp(2*theta[0])
        RV = A * SP.eye(X.shape[0])
     
        return RV

    def Kgrad_theta(self,theta,X,i):
        """
        evaluates the gradient with respect to the hyperparameter theta
        """
        assert i==0, 'unknown hyperparameter'
        assert X.shape[1]==self.n_dimensions, 'dimensions do not match'
        K = self.K(theta,X)
        return 2*K

    def Kgrad_x(self,theta,X,n=None,d=None):
        """
        evaluates the gradient 
        """
        assert X.shape[1]==self.n_dimensions, 'dimensions do not match'
        return SP.zeros((X.shape[0],X.shape[0]))


class DiagArdCF(CovarianceFunction):
    """
    isotropic linear covariance function with a single hyperparameter
    """
    def __init__(self,n_dimensions,n_hyperparameters):
        self.n_dimensions = n_dimensions
        self.n_hyperparameters = n_hyperparameters
        self.params_mask = SP.zeros(n_hyperparameters,dtype=bool)
        
    def K(self,theta,X,X2=None):
        """
        evaluates the kernel
        """
        assert X.shape[1]==self.n_dimensions, 'dimensions do not match'
        if X2!=None:
            assert X2.shape[1]==self.n_dimensions,'dimensions do not match'
        else:
            X2 = X

        if X.shape!=X2.shape:
            return SP.zeros((X.shape[0],X2.shape[0]))
        
        A = SP.exp(2*theta)
        RV = SP.diag(A)
     
        return RV

    def Kgrad_theta(self,theta,X,i):
        """
        evaluates the gradient with respect to the hyperparameter theta
        """
        assert i<len(theta), 'unknown hyperparameter'
        assert X.shape[1]==self.n_dimensions, 'dimensions do not match'

        K = SP.zeros((X.shape[0],X.shape[0]))
        K[i,i] = 1
        sigma = SP.exp(2*theta[i])
        
        return 2*sigma*K

    def Kgrad_x(self,theta,X,n=None,d=None):
        """
        evaluates the gradient 
        """
        assert X.shape[1]==self.n_dimensions, 'dimensions do not match'
        return SP.zeros((X.shape[0],X.shape[0]))

        

