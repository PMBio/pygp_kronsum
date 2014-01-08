import pdb
import logging as LG
import scipy as SP
import scipy.linalg as LA
import copy
import sys

from gp_base import GP
from gplvm import GPLVM
from gp_kronprod import ravel,unravel

import core.likelihood.likelihood_base as likelihood_base
from core.linalg.linalg_matrix import jitChol
import core.covariance.linear as linear
import core.covariance.fixed as fixed

    
class KronSumGP(GPLVM):
    """GPLVM for kronecker type covariance structures
    This class assumes the following model structure
    vec(Y) ~ GP(0, C \otimes R + Sigma \otimes Omega)
    """

    __slots__ = ['covar_c','covar_r','covar_o','covar_s']
    
    def __init__(self,covar_r=None,covar_c=None,covar_o=None,covar_s=None,prior=None):
        self.covar_r = covar_r
        self.covar_c = covar_c
        self.covar_s = covar_s
        self.covar_o = covar_o
        self.likelihood = None
        self._covar_cache = None
        self.prior = None

    def setData(self,Y=None,X=None,X_r=None,X_o=None,**kwargs):
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
            self.covar_r.X = X
        if X_o!=None:
            self.covar_o.X = X_o
        
        self._invalidate_cache()
        
    def _update_inputs(self,hyperparams):
        """ update the inputs from gplvm model """
        if 'X_c' in hyperparams.keys():
            self.covar_c.X = hyperparams['X_c']
        if 'X_r' in hyperparams.keys():
            self.covar_r.X = hyperparams['X_r']
        if 'X_s' in hyperparams.keys():
            self.covar_s.X = hyperparams['X_s']
        if 'X_o' in hyperparams.keys():
            self.covar_o.X = hyperparams['X_o']
            

    def _update_kernel(self,hyperparams,covar_id):
        keys = []
        if 'covar_%s'%covar_id in hyperparams:
            keys.append('covar_%s'%covar_id)
        if 'X_%s'%covar_id in hyperparams:
            keys.append('X_%s'%covar_id)
 
        if not(self._is_cached(hyperparams,keys=keys)):
            K = getattr(self,'covar_%s'%covar_id).K(hyperparams['covar_%s'%covar_id])
            self._covar_cache['K_%s'%covar_id] = K

    def predict(self,hyperparams,Xstar_r=None,**kwargs):
        """
        predict on Xstar
        """
        if Xstar_r!=None:
            self.covar_r.Xcross = Xstar_r
        self._update_inputs(hyperparams)
        KV = self.get_covariances(hyperparams)
        Kstar_r = self.covar_r.Kcross(hyperparams['covar_r'])
        Kstar_c = self.covar_c.K(hyperparams['covar_c']) # kernel over tasks is fixed!
        Kstar = SP.kron(Kstar_c,Kstar_r)
        Ystar = SP.dot(Kstar.T,KV['alpha'])
        Ystar = unravel(Ystar,self.covar_r.n_cross,self.t)
        return Ystar
    
    def get_covariances(self,hyperparams):
        """
        INPUT:
        hyperparams:  dictionary
        OUTPUT: dictionary with the fields
        Kr:     kernel on rows
        Kc:     kernel on columns
        Knoise: noise kernel
        """
        if self._is_cached(hyperparams):
            return self._covar_cache
        if self._covar_cache==None:
            self._covar_cache = {}

        # get EVD of C
        self._update_kernel(hyperparams,'c')
        self._update_kernel(hyperparams,'r')
        self._update_kernel(hyperparams,'s')
        self._update_kernel(hyperparams,'o')

        # create short-cut
        KV = self._covar_cache
    
        Yvec = ravel(self.Y)
        K = SP.kron(KV['K_c'],KV['K_r']) + SP.kron(KV['K_s'],KV['K_o'])
        L = jitChol(K)[0].T # lower triangular
        alpha = LA.cho_solve((L,True),Yvec)
        alpha2D = SP.reshape(alpha,(self.nt,1))
        Kinv = LA.cho_solve((L,True),SP.eye(self.nt))
        W = Kinv - SP.dot(alpha2D,alpha2D.T)
        KV['Yvec'] = Yvec
        KV['K'] = K
        KV['Kinv'] = Kinv
        KV['L'] = L
        KV['alpha'] = alpha
        KV['W'] = W
            
        KV['hyperparams'] = copy.deepcopy(hyperparams)
        return KV
        

    def _LML_covar(self,hyperparams):
        """
        log marginal likelihood
        """
        self._update_inputs(hyperparams)
        
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LML_covar')
            return 1E6
      
        lml_const = 0.5*self.n*self.t*(SP.log(2*SP.pi))
        lml_quad = 0.5 * (KV['alpha']*KV['Yvec']).sum()
        lml_det =  SP.log(SP.diag(KV['L'])).sum()
        lml = lml_quad + lml_det + lml_const
        return lml
        
    def _LMLgrad_covar(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with respect to the
        hyperparameters of the covariance function
        """
        self._update_inputs(hyperparams)
        RV = {}
        if 'covar_r' in hyperparams:
            RV.update(self._LMLgrad_covar_r(hyperparams))

        if 'covar_c' in hyperparams:
            RV.update(self._LMLgrad_covar_c(hyperparams))
            
        if 'covar_s' in hyperparams:
            RV.update(self._LMLgrad_covar_sigma(hyperparams))
                
        if 'covar_o' in hyperparams:
            RV.update(self._LMLgrad_covar_omega(hyperparams))
            
        return RV

    def _LMLgrad_x(self,hyperparams,debugging=False):
        """
        evaluates the gradient of the log marginal likelihood with respect to
        the latent factors
        """
        RV = {}
        if 'X_r' in hyperparams:
            RV.update(self._LMLgrad_x_r(hyperparams,debugging=debugging))

        if 'X_c' in hyperparams:
            RV.update(self._LMLgrad_x_c(hyperparams,debugging=debugging))

        if 'X_s' in hyperparams:
            RV.update(self._LMLgrad_x_sigma(hyperparams,debugging=debugging))

        if 'X_o' in hyperparams:
            RV.update(self._LMLgrad_x_omega(hyperparams,debugging=debugging))
            
        return RV


    def _LMLgrad_x_r(self,hyperparams,debugging=False):
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_x_r')
            return {'X_r':SP.zeros(hyperparams['X_r'].shape)}

        LMLgrad = SP.zeros((self.n,self.covar_r.n_dimensions))
        for d in xrange(self.covar_r.n_dimensions):
            Kgrad_x = self.covar_r.Kgrad_x(hyperparams['covar_r' ],d)
            Kgrad_x = SP.tile(Kgrad_x,self.n)
            Kgrad_x = SP.kron(KV['K_c'],Kgrad_x)
            #Kgrad_x = SP.kron(KV['K_c'], Kgrad_x)
            LMLgrad[:,d] = unravel(SP.sum(KV['W']*Kgrad_x,axis=0),self.n,self.t).sum(1)

        if debugging:
            _LMLgrad = SP.zeros((self.n,self.covar_r.n_dimensions))
            for n in xrange(self.n):
                for d in xrange(self.covar_r.n_dimensions):
                    Kgrad_x = self.covar_r.Kgrad_x(hyperparams['covar_r'],d,n)
                    Kgrad_x = SP.kron(KV['K_c'],Kgrad_x)
                    _LMLgrad[n,d] = 0.5*(KV['W']*Kgrad_x).sum()
            assert SP.allclose(LMLgrad,_LMLgrad), 'ouch,something is wrong'
            
        return {'X_r': LMLgrad}

    def _LMLgrad_x_c(self,hyperparams,debugging=False):
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_x_c')
            return {'X_c':SP.zeros(hyperparams['X_c'].shape)}

        LMLgrad = SP.zeros((self.t,self.covar_c.n_dimensions))
        for d in xrange(self.covar_c.n_dimensions):
            Kgrad_x = self.covar_c.Kgrad_x(hyperparams['covar_c'],d)
            Kgrad_x = SP.tile(Kgrad_x,self.t)
            Kgrad_x = SP.kron(Kgrad_x,KV['K_r'])
            LMLgrad[:,d] = unravel(SP.sum(KV['W']*Kgrad_x,axis=0),self.n,self.t).sum(0)

        if debugging:
            _LMLgrad = SP.zeros((self.t,self.covar_c.n_dimensions))
            for t in xrange(self.t):
                for d in xrange(self.covar_c.n_dimensions):
                    Kgrad_x = self.covar_c.Kgrad_x(hyperparams['covar_c'],d,t)
                    Kgrad_x = SP.kron(Kgrad_x,KV['K_r'])
                    _LMLgrad[t,d] = 0.5*(KV['W']*Kgrad_x).sum()
            assert SP.allclose(LMLgrad,_LMLgrad), 'ouch,something is wrong'
            
        return {'X_c': LMLgrad}

    def _LMLgrad_x_sigma(self,hyperparams,debugging=False):
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_x_sigma')
            return {'X_s':SP.zeros(hyperparams['X_s'].shape)}

        LMLgrad = SP.zeros((self.t,self.covar_s.n_dimensions))
        for d in xrange(self.covar_s.n_dimensions):
            Kgrad_x = self.covar_s.Kgrad_x(hyperparams['covar_s'],d)
            Kgrad_x = SP.tile(Kgrad_x,self.t)
            Kgrad_x = SP.kron(Kgrad_x,KV['K_o'])
            LMLgrad[:,d] = unravel(SP.sum(KV['W']*Kgrad_x,axis=0),self.n,self.t).sum(0)

            
        if debugging:
            _LMLgrad = SP.zeros((self.t,self.covar_s.n_dimensions))
            for t in xrange(self.t):
                for d in xrange(self.covar_s.n_dimensions):
                    Kgrad_x = self.covar_s.Kgrad_x(hyperparams['covar_s'],d,t)
                    Kgrad_x = SP.kron(Kgrad_x,KV['K_o'])
                    _LMLgrad[t,d] = 0.5*(KV['W']*Kgrad_x).sum()
            assert SP.allclose(LMLgrad,_LMLgrad), 'ouch,something is wrong'
            
        return {'X_s': LMLgrad}

    def _LMLgrad_x_omega(self,hyperparams,debugging=False):
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_x_omega')
            return {'X_o':SP.zeros(hyperparams['X_o'].shape)}

        LMLgrad = SP.zeros((self.n,self.covar_o.n_dimensions))
        for d in xrange(self.covar_o.n_dimensions):
            Kgrad_x = self.covar_o.Kgrad_x(hyperparams['covar_o'],d)
            Kgrad_x = SP.tile(Kgrad_x,self.n)
            Kgrad_x = SP.kron(KV['K_s'],Kgrad_x)
            LMLgrad[:,d] = unravel(SP.sum(KV['W']*Kgrad_x,axis=0),self.n,self.t).sum(1)
   
        if debugging:
            _LMLgrad = SP.zeros((self.n,self.covar_o.n_dimensions))
            for n in xrange(self.n):
                for d in xrange(self.covar_o.n_dimensions):
                    Kgrad_x = self.covar_o.Kgrad_x(hyperparams['covar_o'],d,n)
                    Kgrad_x = SP.kron(KV['K_s'],Kgrad_x)
                    _LMLgrad[n,d] = 0.5*(KV['W']*Kgrad_x).sum()
            assert SP.allclose(LMLgrad,_LMLgrad), 'ouch,something is wrong'
                
        return {'X_o': LMLgrad}


    def _LMLgrad_covar_r(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with respect to the hyperparameters of covar_r
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_covar_r')
            return {'covar_r':SP.zeros(hyperparams['covar_r'].shape)}

        theta = SP.zeros(len(hyperparams['covar_r']))
        for i in range(len(theta)):
            Kgrad_r = self.covar_r.Kgrad_theta(hyperparams['covar_r'],i)
            Kd = SP.kron(KV['K_c'],Kgrad_r)
            LMLgrad = 0.5 * (KV['W']*Kd).sum()
            theta[i] = LMLgrad
        return {'covar_r':theta}

    def _LMLgrad_covar_c(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with respect to the hyperparameters of covar_c
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_covar_c')
            return {'covar_c':SP.zeros(hyperparams['covar_c'].shape)}

        theta = SP.zeros(len(hyperparams['covar_c']))
        for i in range(len(theta)):
            Kgrad_c = self.covar_c.Kgrad_theta(hyperparams['covar_c'],i)
            Kd = SP.kron(Kgrad_c,KV['K_r'])
            LMLgrad = 0.5 * (KV['W']*Kd).sum()
            theta[i] = LMLgrad

        return {'covar_c':theta}

    def _LMLgrad_covar_omega(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with respect to the hyperparameters of covar_omega
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMgradL_covar_omega')
            return {'covar_o':SP.zeros(hyperparams['covar_o'].shape)}

        theta = SP.zeros(len(hyperparams['covar_o']))
        for i in range(len(theta)):
            Kgrad_o = self.covar_o.Kgrad_theta(hyperparams['covar_o'],i)
            Kd = SP.kron(KV['K_s'], Kgrad_o)
            LMLgrad = 0.5 * (KV['W']*Kd).sum()
            theta[i] = LMLgrad

        return {'covar_o':theta}

    def _LMLgrad_covar_sigma(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood with respect to the hyperparameters of covar_sigma
        """
        try:
            KV = self.get_covariances(hyperparams)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_covar_sigma')
            return {'covar_s':SP.zeros(hyperparams['covar_s'].shape)}

        theta = SP.zeros(len(hyperparams['covar_s']))
        for i in range(len(theta)):
            Kgrad_s = self.covar_s.Kgrad_theta(hyperparams['covar_s'],i)
            Kd = SP.kron(Kgrad_s, KV['K_o'])
            LMLgrad = 0.5 * (KV['W']*Kd).sum()
            theta[i] = LMLgrad
        return {'covar_s':theta}
