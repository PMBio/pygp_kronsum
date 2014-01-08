import scipy as SP
import pdb

class PriorList():
    """
    class for fixed priors of the hyperparameters

    theta
    name
    functions
    """
    def __init__(self,prior_list=None):
        if prior_list==None:
            self._prior_list = []
        else:
            self._prior_list = prior_list

    def append_prior(self,prior):
        self._prior_list.append(prior)

    def LML(self,hyperparams):
        LML = 0
        for prior in self._prior_list:
            LML += prior.LML(hyperparams)
        return LML
    
    def LMLgrad(self,hyperparams):
        LMLgrad = {}
        for prior in self._prior_list:
            LMLgrad.update(prior.LMLgrad(hyperparams))
        return LMLgrad    
    
class GaussianPrior():
    """
    standard gaussian prior with zero mean
    """
    def __init__(self,key=None,theta=None):
        self.key = key
        self.theta = theta
        
    def LML(self,hyperparams):
        """
        computes the log likelihood of the hyperparams
        """
        x = hyperparams[self.key]
        ss = self.theta[0]
        nt = SP.prod(x.shape)
        
        lml_quad = 1./(2*ss)*(x*x).sum()
        lml_det = 0.5*nt*SP.log(ss)
        lml_const = 0.5*nt*SP.log(2*SP.pi)
        LML = lml_quad + lml_det + lml_const
        return LML
        

    def LMLgrad(self,hyperparams):
        x = hyperparams[self.key]
        ss = self.theta[0]

        LMLgrad = 1./ss*x
        return {self.key : LMLgrad}
        
