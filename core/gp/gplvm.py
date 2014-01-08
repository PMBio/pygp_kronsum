import pdb
import numpy as NP
import scipy as SP
import scipy.linalg as LA

from gp_base import GP
import logging as LG
import collections

def PCA(Y,n_factors):
    """
    run standard PCA: Y = WX + noise

    input:
    Y: [N,D] data matrix

    output:
    X: latent variables [D,K]
    W: weights [N,K]
    """
    # run singular value decomposition
    U,S,Vt = LA.svd(Y,full_matrices=True)
    U = U[:,:n_factors]
    S = SP.diag(S[:n_factors])
    US = SP.dot(U,S)
    Vt = Vt[:n_factors,:].T
    # normalize weights
    s = U.std(axis=0)
    US = US/s
    Vt = (Vt*s)
    return US,Vt

class GPLVM(GP):

    def LML(self,hyperparams):
        """
        calculate the log marginal likelihood for the given logtheta

        Input:
        hyperparams: dictionary
        """
        self._update_inputs(hyperparams)
        LML = self._LML_covar(hyperparams)

        if self.prior!=None:
            LML += self.prior.LML(hyperparams)
        return LML

    def LMLgrad(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood
        
        Input:
        hyperparams: dictionary
        """
        self._update_inputs(hyperparams)
        RV = {}
        # gradient with respect to hyperparameters
        RV.update(self._LMLgrad_covar(hyperparams))
        # gradient with respect to noise parameters
        if self.likelihood is not None:
            RV.update(self._LMLgrad_lik(hyperparams))
        # gradient with respect to X
        RV.update(self._LMLgrad_x(hyperparams))

        if self.prior!=None:
            priorRV = self.prior.LMLgrad(hyperparams)
            for key in priorRV.keys():
                if key in RV:
                    RV[key] += priorRV[key]
                else:
                    RV[key] = priorRV[key]

        return RV

    def _LMLgrad_x(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with
        respect to the latent variables
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LML_grad_x')
            return {'X': SP.zeros(hyperparams['X'].shape)}

        W = KV['W']
        LMLgrad = SP.zeros((self.n,self.covar.n_dimensions))
        for d in xrange(self.covar.n_dimensions):
            Kd_grad = self.covar.Kgrad_x(hyperparams['covar'],d)
            LMLgrad[:,d] = SP.sum(W*Kd_grad,axis=0)
        return {'X':LMLgrad}

    def _update_inputs(self,hyperparams):
        """ update the inputs from gplvm model """
        if 'X' in hyperparams:
            self.covar.X = hyperparams['X']
        
