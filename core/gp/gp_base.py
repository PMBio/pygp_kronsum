import pdb
import scipy as SP
import scipy.linalg as LA
import copy
import logging as LG
import sys

from core.linalg.linalg_matrix import jitChol
import core.likelihood.likelihood_base as likelihood_base
import collections



class GP(object):
    """
    Gaussian Process regression class. Holds all information for the GP regression to take place.
    """

    __slots__ = ['Y','n','t','nt','covar','likelihood','_covar_cache','prior']
    
    def __init__(self,covar=None,likelihood=None,covar_r=None,prior=None):
        """
        covar:        Covariance function
        likelihood:   Likelihood function
        Y:            Outputs [n x t]
        """
        self.covar = covar
        if covar_r !=None:
            self.covar = covar_r
        self.likelihood = likelihood
        self._covar_cache = None
        self.prior = prior
        
    def setData(self,Y=None,X=None,X_r=None,**kwargs):
        """
        set data
        Y:    Outputs [n x t]
        """
        assert Y.ndim==2, 'Y must be a two dimensional vector'
        self.Y = Y
        self.n = Y.shape[0]
        self.t = Y.shape[1]
        self.nt = self.n * self.t

        if X_r!=None:
            X = X_r
        if X!=None:
            self.covar.X = X
        
        self._invalidate_cache()

        
    def LML(self,hyperparams):
        """
        evalutes the log marginal likelihood for the given hyperparameters

        hyperparams
        """
        LML = self._LML_covar(hyperparams)

        if self.prior != None:
            LML += self.prior.LML(hyperparams)
        
        return LML

    def LMLgrad(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood for the given hyperparameters
        """
        RV = {}
        # gradient with respect to hyperparameters
        RV.update(self._LMLgrad_covar(hyperparams))
        if self.likelihood != None:
            # gradient with respect to noise parameters
            RV.update(self._LMLgrad_lik(hyperparams))

        if self.prior!=None:
            RVprior = self.prior.LMLgrad(hyperparams)
            for key in RVprior.keys():
                if key in RV:
                    RV[key] += RVprior[key]
                else:
                    RV[key] = RVprior[key]
 

            
        return RV

    def _LML_covar(self,hyperparams):
        """
        log marginal likelihood
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LML_covar')
            return 1E6
        
        alpha = KV['alpha']
        L = KV['L']
        
        lml_quad = 0.5 * (alpha*self.Y).sum()
        lml_det = self.t *SP.log(SP.diag(L)).sum()
        lml_const = 0.5*self.n*self.t*SP.log(2*SP.pi)
        LML = lml_quad + lml_det + lml_const
        return LML

    def _LMLgrad_covar(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with respect to the
        hyperparameters of the covariance function
        """
        logtheta = hyperparams['covar']

        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_covar')
            return {'covar':SP.zeros(len(logtheta))}
        
        W = KV['W']
        n_theta = len(logtheta)
        LMLgrad = SP.zeros(len(logtheta))
        for i in xrange(n_theta):
            Kd = self.covar.Kgrad_theta(hyperparams['covar'],i)
            LMLgrad[i] = 0.5 * (W*Kd).sum()
        return {'covar':LMLgrad}

    def _LMLgrad_lik(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with respect to
        the hyperparameters of the likelihood function
        """
        logtheta = hyperparams['lik']
        KV = self._covar_cache
        W = KV['W']
        n_theta = len(logtheta)
        LMLgrad = SP.zeros(len(logtheta))
        for i in xrange(n_theta):
            Kd = self.likelihood.Kgrad_theta(hyperparams['lik'],self.n,i)
            LMLgrad[i] = 0.5 * (W*Kd).sum()
        return {'lik':LMLgrad}


    def predict(self,hyperparams,Xstar=None,Xstar_r=None,**kwargs):
        """
        predict on Xstar
        """
        if Xstar_r!=None:
            Xstar = Xstar_r
        if Xstar != None:
            self.covar.Xcross = Xstar
        
        KV = self.get_covariances(hyperparams)
        Kstar = self.covar.Kcross(hyperparams['covar'])
        Ystar = SP.dot(Kstar.T,KV['alpha'])
        return Ystar.flatten()
        
    def get_covariances(self,hyperparams):
        """
        INPUT:
        hyperparams:  dictionary
        OUTPUT: dictionary with the fields
        K:     kernel
        Kinv:  inverse of the kernel
        L:     chol(K)
        alpha: solve(K,y)
        W:     D*Kinv * alpha*alpha^T
        """
        if self._is_cached(hyperparams):
            return self._covar_cache

        K = self.covar.K(hyperparams['covar'])
        
        if self.likelihood is not None:
            Knoise = self.likelihood.K(hyperparams['lik'],self.n)
            K += Knoise
            
        L = LA.cholesky(K).T# lower triangular
        alpha = LA.cho_solve((L,True),self.Y)
        Kinv = LA.cho_solve((L,True),SP.eye(L.shape[0]))
        W = self.t*Kinv - SP.dot(alpha,alpha.T)
        self._covar_cache = {}
        self._covar_cache['K'] = K
        self._covar_cache['Kinv'] = Kinv
        self._covar_cache['L'] = L
        self._covar_cache['alpha'] = alpha
        self._covar_cache['W'] = W
        self._covar_cache['hyperparams'] = copy.deepcopy(hyperparams) 
        return self._covar_cache

    def _is_cached(self,hyperparams,keys=None):
        """ check wheter model parameters are cached"""
        if self._covar_cache is None:
            return False
        if not ('hyperparams' in self._covar_cache):
            return False
        if keys==None:
            keys = hyperparams.keys()
        for key in keys:
            if (self._covar_cache['hyperparams'][key]!=hyperparams[key]).any():
                return False
        return True

    def _invalidate_cache(self):
        """ reset cache """
        self._covar_cache = None
