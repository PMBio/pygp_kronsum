import pdb
import logging as LG
import scipy as SP
import scipy.linalg as LA
import copy
import sys

from gp_base import GP
from gplvm import GPLVM
from gp_kronprod import ravel,unravel
from core.linalg.linalg_matrix import jitChol
import core.likelihood.likelihood_base as likelihood_base
import collections
from core.linalg.linalg_matrix import jitEigh
from core.data_term import DataTerm

class KronSumGP(GPLVM):
    """GPLVM for kronecker type covariance structures
    This class assumes the following model structure
    vec(Y) ~ GP(0, C \otimes R + S \otimes O), where S \otimes O describes the noise.
    """

    __slots__ = ['covar_c','covar_r','covar_o','covar_s']
    
    def __init__(self,covar_r=None,covar_c=None,covar_o=None,covar_s=None,prior=None):
        self.covar_r = covar_r
        self.covar_c = covar_c
        self.covar_s = covar_s
        self.covar_o = covar_o
        self.likelihood  = None
        self._covar_cache = None
        self.prior = None

    def setPrior(self,prior):
        self.prior = None

    # def setData(self,Y=None,X=None,X_r=None,X_o=None,**kwargs):
    #     """
    #     set data
    #     Y:    Outputs [n x t]
    #     """
    #     assert Y.ndim==2, 'Y must be a two dimensional vector'
    #     self.Y = Y
    #     self.n = Y.shape[0]
    #     self.t = Y.shape[1]
    #     self.nt = self.n * self.t

    #     if X_r!=None:
    #         X = X_r
    #     if X!=None:
    #         self.covar_r.X = X
    #     if X_o!=None:
    #         self.covar_o.X = X_o
        
    #     self._invalidate_cache()

    def setData(self,Y=None,X=None,X_r=None,X_o=None,**kwargs):
        """
        set data
        Y:   DataTerm representing outputs [n x t]
        """
        assert isinstance(Y, DataTerm)
        
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
            

    def _update_evd(self,hyperparams,covar_id):
        keys = []
        if 'covar_%s'%covar_id in hyperparams:
            keys.append('covar_%s'%covar_id)
        if 'X_%s'%covar_id in hyperparams:
            keys.append('X_%s'%covar_id)
            
        if not(self._is_cached(hyperparams,keys=keys)):
            K = getattr(self,'covar_%s'%covar_id).K(hyperparams['covar_%s'%covar_id])
            S,U = LA.eigh(K)
        
            self._covar_cache['K_%s'%covar_id] = K
            self._covar_cache['U_%s'%covar_id] = U
            self._covar_cache['S_%s'%covar_id] = S

    def predict(self,hyperparams,Xstar_r=None,debugging=False):
        """
        predict over new training points
        """
        self._update_inputs(hyperparams)
        KV = self.get_covariances(hyperparams,debugging=debugging)
        if Xstar_r!=None:
            self.covar_r.Xcross = Xstar_r
        
        Kstar_r = self.covar_r.Kcross(hyperparams['covar_r'])
        Kstar_c = self.covar_c.K(hyperparams['covar_c']) # kernel over tasks is fixed!
        S = SP.kron(KV['Stilde_c'],KV['Stilde_r'])+1
        USUc = SP.dot(SP.sqrt(1./KV['S_s']) * KV['U_s'],KV['Utilde_c'])
        USUr = SP.dot(SP.sqrt(1./KV['S_o']) * KV['U_o'],KV['Utilde_r'])
        KinvY = SP.dot(USUr,SP.dot(unravel(ravel(KV['UYtildeU_rc']) * 1./S,self.n,self.t),USUc.T))
        Ystar = SP.dot(Kstar_r.T,SP.dot(KinvY,Kstar_c))
        Ystar = unravel(Ystar,self.covar_r.n_cross,self.t)

        if debugging:
            Kstar = SP.kron(Kstar_c,Kstar_r)
            Ynaive = SP.dot(Kstar.T,KV['alpha'])
            Ynaive = unravel(Ynaive,self.covar_r.n_cross,self.t)
            assert SP.allclose(Ystar,Ynaive), 'ouch, prediction does not work out'
        return Ystar
    
    def get_covariances(self,hyperparams,debugging=False):
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
        self._update_evd(hyperparams,'c')
        self._update_evd(hyperparams,'r')
        self._update_evd(hyperparams,'s')
        self._update_evd(hyperparams,'o')
        
        # create short-cut
        KV = self._covar_cache

        # do only recompute if hyperparameters of K_c,K_s change
        keys = list(set(hyperparams.keys()) & set(['covar_c','covar_s','X_c','X_s']))
        if not(self._is_cached(hyperparams,keys=keys)):
            KV['USi_c'] = SP.sqrt(1./KV['S_c']) * KV['U_c']
            KV['USi_s'] = SP.sqrt(1./KV['S_s']) * KV['U_s']
        
            Ktilde_c = SP.dot(KV['USi_s'].T,SP.dot(KV['K_c'],KV['USi_s']))
            Stilde_c,Utilde_c = LA.eigh(Ktilde_c)
            Ktilde_s = SP.dot(KV['USi_c'].T,SP.dot(KV['K_s'],KV['USi_c']))
            Stilde_s,Utilde_s = LA.eigh(Ktilde_s)
            KV['Ktilde_c'] = Ktilde_c; KV['Utilde_c'] = Utilde_c; KV['Stilde_c'] = Stilde_c
            KV['Ktilde_s'] = Ktilde_s; KV['Utilde_s'] = Utilde_s; KV['Stilde_s'] = Stilde_s

        # do only recompute if hyperparameters of K_r,K_o change
        keys = list(set(hyperparams.keys()) & set(['covar_r','covar_o','X_r','X_o']))
        if not(self._is_cached(hyperparams,keys=keys)):
            KV['USi_r'] = SP.sqrt(1./KV['S_r']) * KV['U_r']
            KV['USi_o'] = SP.sqrt(1./KV['S_o']) * KV['U_o']
            Ktilde_r = SP.dot(KV['USi_o'].T,SP.dot(KV['K_r'],KV['USi_o']))
            Stilde_r,Utilde_r = LA.eigh(Ktilde_r)
            Ktilde_o = SP.dot(KV['USi_r'].T,SP.dot(KV['K_o'],KV['USi_r']))
            Stilde_o,Utilde_o = LA.eigh(Ktilde_o)
            KV['Ktilde_r'] = Ktilde_r; KV['Utilde_r'] = Utilde_r; KV['Stilde_r'] = Stilde_r
            KV['Ktilde_o'] = Ktilde_o; KV['Utilde_o'] = Utilde_o; KV['Stilde_o'] = Stilde_o

    
        KV['UYtildeU_rc'] = SP.dot(KV['Utilde_r'].T,SP.dot(KV['USi_o'].T,SP.dot(self.Y.value(),SP.dot(KV['USi_s'],KV['Utilde_c']))))
        KV['UYtildeU_os'] = SP.dot(KV['Utilde_o'].T,SP.dot(KV['USi_r'].T,SP.dot(self.Y.value(),SP.dot(KV['USi_c'],KV['Utilde_s']))))

        KV['Stilde_rc'] = SP.kron(KV['Stilde_c'],KV['Stilde_r'])+1
        KV['Stilde_os'] = SP.kron(KV['Stilde_s'],KV['Stilde_o'])+1
        
        if debugging:
            # needed later
            Yvec = ravel(self.Y.value())
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
        

    def _LML_covar(self,hyperparams,debugging=False):
        """
        log marginal likelihood
        """
        self._update_inputs(hyperparams)

        try:
            KV = self.get_covariances(hyperparams,debugging=debugging)
        except LA.LinAlgError:
            pdb.set_trace()
            LG.error('linalg exception in _LML_covar')
            return 1E6
        
        Si = 1./KV['Stilde_rc']
        lml_quad = 0.5*(ravel(KV['UYtildeU_rc'])**2 * Si).sum()
        lml_det = 0.5*(SP.log(KV['S_s']).sum()*self.n + SP.log(KV['S_o']).sum()*self.t)
        lml_det+= 0.5*SP.log(KV['Stilde_rc']).sum()
        lml_const = 0.5*self.nt*(SP.log(2*SP.pi))
        
        if debugging:
            # do calculation without kronecker tricks and compare
            _lml_quad = 0.5 * (KV['alpha']*KV['Yvec']).sum()
            _lml_det =  SP.log(SP.diag(KV['L'])).sum()
            assert SP.allclose(_lml_quad,lml_quad,atol=1E-2,rtol=1E-2),  'ouch, quadratic form is wrong: %.2f'%LA.norm(lml_quad,_lml_quad)
            assert SP.allclose(_lml_det, lml_det,atol=1E-2,rtol=1E-2), 'ouch, ldet is wrong in _LML_covar'%LA.norm(lml_det,_lml_det)
        
        lml = lml_quad + lml_det + lml_const

        return lml

    def LMLgrad(self,hyperparams):
        """
        evaluates the gradient of the log marginal likelihood
        
        Input:
        hyperparams: dictionary
        """
        self._update_inputs(hyperparams)
        
        RV = {}
   
        # gradient with respect to covariance C
        if 'covar_c' in hyperparams or 'X_c' in hyperparams:
            RV.update(self._LMLgrad_c(hyperparams))
        
        # gradient with respect to covariance R
        if 'covar_r' in hyperparams or 'X_r' in hyperparams:
            RV.update(self._LMLgrad_r(hyperparams))

        # gradient with respect oc covariace Sigma
        if 'covar_s' in hyperparams or 'X_s' in hyperparams:
            RV.update(self._LMLgrad_s(hyperparams))

        # gradient with respect to covariance Omega
        if 'covar_o' in hyperparams or 'X_o' in hyperparams:
            RV.update(self._LMLgrad_o(hyperparams))
            
        if self.prior!=None:
            priorRV = self.prior.LMLgrad(hyperparams)
            for key in priorRV.keys():
                if key in RV:
                    RV[key] += priorRV[key]
                else:
                    RV[key] = priorRV[key]
      
        return RV


    def _LMLgrad_c(self,hyperparams,debugging=False):
        """
        evaluates the gradient with respect to the covariance matrix C
        """
        try:
            KV = self.get_covariances(hyperparams, debugging=debugging)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_x_c')
            return {'X_c':SP.zeros(hyperparams['X_c'].shape)}

        Si = 1./KV['Stilde_rc']
        SS = SP.dot(unravel(Si,self.n,self.t).T,KV['Stilde_r'])
        USU = SP.dot(KV['USi_s'],KV['Utilde_c'])        
        Yhat = unravel(Si * ravel(KV['UYtildeU_rc']),self.n,self.t)
        RV = {}
        
        if 'X_c' in hyperparams:
            USUY = SP.dot(USU,Yhat.T)
            USUYSYUSU = SP.dot(USUY,(KV['Stilde_r']*USUY).T)
        
            LMLgrad = SP.zeros((self.t,self.covar_c.n_dimensions))
            LMLgrad_det = SP.zeros((self.t,self.covar_c.n_dimensions))
            LMLgrad_quad = SP.zeros((self.t,self.covar_c.n_dimensions))
        
            for d in xrange(self.covar_c.n_dimensions):
                Kd_grad = self.covar_c.Kgrad_x(hyperparams['covar_c'],d)
                # calculate gradient of logdet
                UcU = SP.dot(Kd_grad.T,USU)*USU
                LMLgrad_det[:,d] = SP.dot(UcU,SS.T)
                # calculate gradient of squared form
                LMLgrad_quad[:,d] = - (USUYSYUSU*Kd_grad).sum(0)

            LMLgrad = LMLgrad_det + LMLgrad_quad
            RV['X_c'] = LMLgrad
            
            if debugging:
                _LMLgrad = SP.zeros((self.t,self.covar_c.n_dimensions))
                for t in xrange(self.t):
                    for d in xrange(self.covar_c.n_dimensions):
                        Kgrad_x = self.covar_c.Kgrad_x(hyperparams['covar_c'],d,t)
                        Kgrad_x = SP.kron(Kgrad_x,KV['K_r'])
                        _LMLgrad[t,d] = 0.5*(KV['W']*Kgrad_x).sum()

                assert SP.allclose(LMLgrad,_LMLgrad,rtol=1E-3,atol=1E-2), 'ouch, something is wrong: %.2f'%LA.norm(LMLgrad-_LMLgrad)

        if 'covar_c' in hyperparams:
            theta = SP.zeros(len(hyperparams['covar_c']))
            for i in range(len(theta)):
                Kgrad_c = self.covar_c.Kgrad_theta(hyperparams['covar_c'],i)
                UdKU = SP.dot(USU.T, SP.dot(Kgrad_c, USU))
                SYUdKU = SP.dot(UdKU,KV['Stilde_r'] * Yhat.T)
                LMLgrad_det = SP.sum(Si*SP.kron(SP.diag(UdKU),KV['Stilde_r']))
                LMLgrad_quad = -(Yhat.T*SYUdKU).sum()
                LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
                theta[i] = LMLgrad

                if debugging:
                    Kd = SP.kron(Kgrad_c,KV['K_r'])
                    _LMLgrad = 0.5 * (KV['W']*Kd).sum()
                    assert SP.allclose(LMLgrad,_LMLgrad,rtol=1E-3,atol=1E-2), 'ouch, something is wrong: %.2f'%LA.norm(LMLgrad-_LMLgrad)
            RV['covar_c']  = theta
        return RV


    def _LMLgrad_r(self,hyperparams,debugging=False):
        """
        evaluate gradients with respect to covariance matrix R
        """
        try:
            KV = self.get_covariances(hyperparams, debugging=debugging)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_x_r')
            return {'X_r':SP.zeros(hyperparams['X_r'].shape)}

        Si = 1./KV['Stilde_rc']
        SS = SP.dot(unravel(Si,self.n,self.t),KV['Stilde_c'])
        USU = SP.dot(KV['USi_o'],KV['Utilde_r'])        
        Yhat = unravel(Si * ravel(KV['UYtildeU_rc']),self.n,self.t)
        RV = {}

        if 'X_r' in hyperparams:
            USUY = SP.dot(USU,Yhat)
            USUYSYUSU = SP.dot(USUY,(KV['Stilde_c']*USUY).T)
        
            LMLgrad = SP.zeros((self.n,self.covar_r.n_dimensions))
            LMLgrad_det = SP.zeros((self.n,self.covar_r.n_dimensions))
            LMLgrad_quad = SP.zeros((self.n,self.covar_r.n_dimensions))

            for d in xrange(self.covar_r.n_dimensions):
                Kd_grad = self.covar_r.Kgrad_x(hyperparams['covar_r'],d)
                # calculate gradient of logdet
                UrU = SP.dot(Kd_grad.T,USU)*USU
                LMLgrad_det[:,d] = SP.dot(UrU,SS.T)
                # calculate gradient of squared form
                LMLgrad_quad[:,d] = -(USUYSYUSU*Kd_grad).sum(0)            

            LMLgrad = LMLgrad_det + LMLgrad_quad
            RV['X_r'] = LMLgrad
            
            if debugging:
                _LMLgrad = SP.zeros((self.n,self.covar_r.n_dimensions))
                for n in xrange(self.n):
                    for d in xrange(self.covar_r.n_dimensions):
                        Kgrad_x = self.covar_r.Kgrad_x(hyperparams['covar_r'],d,n)
                        Kgrad_x = SP.kron(KV['K_c'],Kgrad_x)
                        _LMLgrad[n,d] = 0.5*(KV['W']*Kgrad_x).sum()
                assert SP.allclose(LMLgrad,_LMLgrad,rtol=1E-3,atol=1E-2), 'ouch, something is wrong: %.2f'%LA.norm(LMLgrad-_LMLgrad)
                
        if 'covar_r' in hyperparams:
            theta = SP.zeros(len(hyperparams['covar_r']))
            for i in range(len(theta)):
                Kgrad_r = self.covar_r.Kgrad_theta(hyperparams['covar_r'],i)
                UdKU = SP.dot(USU.T, SP.dot(Kgrad_r, USU))
                SYUdKU = SP.dot(UdKU,Yhat*KV['Stilde_c'])
                LMLgrad_det = SP.sum(Si*SP.kron(KV['Stilde_c'],SP.diag(UdKU)))
                LMLgrad_quad = -(Yhat*SYUdKU).sum()
                LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
                theta[i] = LMLgrad

                if debugging:
                    Kd = SP.kron(KV['K_c'],Kgrad_r)
                    _LMLgrad = 0.5 * (KV['W']*Kd).sum()
                    assert SP.allclose(LMLgrad,_LMLgrad,rtol=1E-2,atol=1E-2), 'ouch, something is wrong: %.2f'%LA.norm(LMLgrad-_LMLgrad)
            RV['covar_r'] = theta
        return RV


    def _LMLgrad_s(self,hyperparams,debugging=False):
        """
        evaluate gradients with respect to covariance matrix Sigma
        """
        try:
            KV = self.get_covariances(hyperparams, debugging=debugging)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_x_sigma')
            return {'X_s':SP.zeros(hyperparams['X_s'].shape)}

        Si = 1./KV['Stilde_os']
        SS = SP.dot(unravel(Si,self.n,self.t).T,KV['Stilde_o'])
        USU = SP.dot(KV['USi_c'],KV['Utilde_s'])        
        Yhat = unravel(Si * ravel(KV['UYtildeU_os']),self.n,self.t)
        RV = {}
        
        if 'X_s' in hyperparams:
            USUY = SP.dot(USU,Yhat.T)
            USUYSYUSU = SP.dot(USUY,(KV['Stilde_o']*USUY).T)
        
            LMLgrad = SP.zeros((self.t,self.covar_s.n_dimensions))
            LMLgrad_det = SP.zeros((self.t,self.covar_s.n_dimensions))
            LMLgrad_quad = SP.zeros((self.t,self.covar_s.n_dimensions))
        
            for d in xrange(self.covar_s.n_dimensions):
                Kd_grad = self.covar_s.Kgrad_x(hyperparams['covar_s'],d)
                UsU = SP.dot(Kd_grad.T,USU)*USU
                LMLgrad_det[:,d] = SP.dot(UsU,SS.T)
                # calculate gradient of squared form
                LMLgrad_quad[:,d] = -(USUYSYUSU*Kd_grad).sum(0)
            LMLgrad = LMLgrad_det + LMLgrad_quad
            RV['X_s'] = LMLgrad
            
            if debugging:
                _LMLgrad = SP.zeros((self.t,self.covar_s.n_dimensions))
                for t in xrange(self.t):
                    for d in xrange(self.covar_s.n_dimensions):
                        Kgrad_x = self.covar_s.Kgrad_x(hyperparams['covar_s'],d,t)
                        Kgrad_x = SP.kron(Kgrad_x,KV['K_o'])
                        _LMLgrad[t,d] = 0.5*(KV['W']*Kgrad_x).sum()

                assert SP.allclose(LMLgrad,_LMLgrad,rtol=1E-3,atol=1E-2), 'ouch, something is wrong: %.2f'%LA.norm(LMLgrad-_LMLgrad)

        if 'covar_s' in hyperparams:
            theta = SP.zeros(len(hyperparams['covar_s']))
            for i in range(len(theta)):
                Kgrad_s = self.covar_s.Kgrad_theta(hyperparams['covar_s'],i)
                UdKU = SP.dot(USU.T, SP.dot(Kgrad_s, USU))
                SYUdKU = SP.dot(UdKU,KV['Stilde_o'] * Yhat.T)
                LMLgrad_det = SP.sum(Si*SP.kron(SP.diag(UdKU),KV['Stilde_o']))
                LMLgrad_quad = -(Yhat.T*SYUdKU).sum()
                LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
                theta[i] = LMLgrad

                if debugging:
                    Kd = SP.kron(Kgrad_s, KV['K_o'])
                    _LMLgrad = 0.5 * (KV['W']*Kd).sum()
                    assert SP.allclose(LMLgrad,_LMLgrad,rtol=1E-3,atol=1E-2), 'ouch, something is wrong: %.2f'%LA.norm(LMLgrad-_LMLgrad)
            RV['covar_s'] = theta
        return RV


    def _LMLgrad_o(self,hyperparams,debugging=False):
        """
        evaluates the gradient with respect to the covariance matrix Omega
        """
        try:
            KV = self.get_covariances(hyperparams,debugging=debugging)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_x_omega')
            return {'X_o':SP.zeros(hyperparams['X_o'].shape)}

        Si = 1./KV['Stilde_os']
        SS = SP.dot(unravel(Si,self.n,self.t),KV['Stilde_s'])
        USU = SP.dot(KV['USi_r'],KV['Utilde_o'])        
        Yhat = unravel(Si * ravel(KV['UYtildeU_os']),self.n,self.t)
        RV = {}
        
        if 'X_o' in hyperparams:
            USUY = SP.dot(USU,Yhat)
            USUYSYUSU = SP.dot(USUY,(KV['Stilde_s']*USUY).T)
        
            LMLgrad = SP.zeros((self.n,self.covar_o.n_dimensions))
            LMLgrad_det = SP.zeros((self.n,self.covar_o.n_dimensions))
            LMLgrad_quad = SP.zeros((self.n,self.covar_o.n_dimensions))
            
            for d in xrange(self.covar_o.n_dimensions):
                Kd_grad = self.covar_o.Kgrad_x(hyperparams['covar_o'],d)
                # calculate gradient of logdet
                UoU = SP.dot(Kd_grad.T,USU)*USU
                LMLgrad_det[:,d] = SP.dot(UoU,SS.T)
                # calculate gradient of squared form
                LMLgrad_quad[:,d] = -(USUYSYUSU*Kd_grad).sum(0)

            LMLgrad = LMLgrad_det + LMLgrad_quad
            RV['X_o'] = LMLgrad
            
            if debugging:
                _LMLgrad = SP.zeros((self.n,self.covar_o.n_dimensions))
                for n in xrange(self.n):
                    for d in xrange(self.covar_o.n_dimensions):
                        Kgrad_x = self.covar_o.Kgrad_x(hyperparams['covar_o'],d,n)
                        Kgrad_x = SP.kron(KV['K_s'],Kgrad_x)
                        _LMLgrad[n,d] = 0.5*(KV['W']*Kgrad_x).sum()
            assert SP.allclose(LMLgrad,_LMLgrad,rtol=1E-3,atol=1E-2), 'ouch, something is wrong: %.2f'%LA.norm(LMLgrad-_LMLgrad)

        if 'covar_o' in hyperparams:
            theta = SP.zeros(len(hyperparams['covar_o']))
            
            for i in range(len(theta)):
                Kgrad_o = self.covar_o.Kgrad_theta(hyperparams['covar_o'],i)
                UdKU = SP.dot(USU.T, SP.dot(Kgrad_o, USU))
                SYUdKU = SP.dot(UdKU,Yhat*KV['Stilde_s'])
                LMLgrad_det = SP.sum(Si*SP.kron(KV['Stilde_s'],SP.diag(UdKU)))
                LMLgrad_quad = -(Yhat*SYUdKU).sum()
                LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
                theta[i] = LMLgrad

                if debugging:
                    Kd = SP.kron(KV['K_s'], Kgrad_o)
                    _LMLgrad = 0.5 * (KV['W']*Kd).sum()
                    assert SP.allclose(LMLgrad,_LMLgrad,rtol=1E-2,atol=1E-2), 'ouch, something is wrong: %.2f'%LA.norm(LMLgrad-_LMLgrad)
            RV['covar_o'] = theta
                
        return RV
    
 
