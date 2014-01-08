import scipy as SP
import pdb
from covar_base import CovarianceFunction

class DiagIsoCF(CovarianceFunction):
    """
    isotropic covariance function with a single hyperparameter (independent of X)
    """
    def __init__(self,n_dimensions):
        self.n_dimensions = n_dimensions
        self.n_hyperparameters = 1

    def K(self,theta):
        """
        evaluates the kernel
        """
        A = SP.exp(2*theta[0])
        RV = A * SP.eye(self.n)
        return RV

    def Kcross(self,theta):
        """
        evaluates the kernel for given hyperparameters theta between the training samples X1 and the test samples X2
        """
        return SP.zeros((self.n,self.n_cross))
    
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
        return SP.zeros((self.n,self.n))


class DiagArdCF(CovarianceFunction):
    """
    diagonal covariance function with one parameter for each dimension (independent of X)
    """
    def __init__(self,n_dimensions,n_hyperparameters):
        self.n_dimensions = n_dimensions
        self.n_hyperparameters = n_hyperparameters

    def K(self,theta):
        """
        evaluates the kernel
        """
        assert len(theta)==self.n_hyperparameters, 'dimensions do not match'
        A = SP.exp(2*theta)
        RV = SP.diag(A)
     
        return RV

    def Kcross(self,theta):
        """
        evaluates the kernel for given hyperparameters theta between the training samples X1 and the test samples X2
        """
        assert len(theta)==self.n_hyperparameters, 'dimensions do not match'
        return SP.zeros((self.n,self.n_cross))

    def Kgrad_theta(self,theta,i):
        """
        evaluates the gradient with respect to the hyperparameter theta
        """
        assert i<len(theta), 'unknown hyperparameter'
        assert len(theta)==self.n_hyperparameters, 'dimensions do not match'
        
        K = SP.zeros((self.n,self.n))
        K[i,i] = 1
        sigma = SP.exp(2*theta[i])
        return 2*sigma*K

    def Kgrad_x(self,theta,d,n=None):
        """
        partial derivative with respect to X[n,d], if n is set to None with respect to
        the hidden factor X[:,d]
        """
        return SP.zeros((self.n,self.n))

        

