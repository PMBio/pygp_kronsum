import pdb
import logging as LG
import scipy as SP
import scipy.linalg as LA
import copy
import sys

from gp_base import GP
from gplvm import GPLVM

from core.linalg.linalg_matrix import jitChol
import core.likelihood.likelihood_base as likelihood_base


def ravel(Y):
    """
    returns a flattened array: columns are concatenated
    """
    return SP.ravel(Y,order='F')

def unravel(Y,n,t):
    """
    returns a nxt matrix from a raveled matrix
    Y = uravel(ravel(Y))
    """
    return SP.reshape(Y,(n,t),order='F')
    
class KronProdGP(GPLVM):
    """GPLVM for kronecker type covariance structures
    This class assumes the following model structure
    vec(Y) ~ GP(0, Kx \otimes Kd + \sigma^2 \unit)
    """

    __slots__ = ['covar_c','covar_r']
    
    def __init__(self,covar_r=None,covar_c=None,likelihood=None,prior=None):
        assert isinstance(likelihood,likelihood_base.GaussIsoLik), 'likelihood is not implemented yet'
        self.covar_r = covar_r
        self.covar_c = covar_c
        self.likelihood = likelihood
        self.prior = prior
        self._covar_cache = None
        self.debugging = False
        self.Y = None
        

    def setData(self,Y=None,X=None,X_r=None,X_c=None,**kwargs):
        """
        set data
        Y:    Outputs [n x t]
        """
        if Y!=None:
            assert Y.ndim==2, 'Y must be a two dimensional vector'
            self.Y = Y
            self.n = Y.shape[0]
            self.t = Y.shape[1]
            self.nt = self.n * self.t

        if X_r!=None:
            X = X_r
        if X!=None:
            self.covar_r.X = X

        if X_c!=None:
            self.covar_c.X = X_c
        
        self._invalidate_cache()
        
    def _update_inputs(self,hyperparams):
        """ update the inputs from gplvm model """
        if 'X_c' in hyperparams.keys():
            self.covar_c.X = hyperparams['X_c']
        if 'X_r' in hyperparams.keys():
            self.covar_r.X = hyperparams['X_r']

    def predict(self,hyperparams,Xstar_r=None,debugging=False):
        """
        predict on Xstar
        """
        self._update_inputs(hyperparams)
        KV = self.get_covariances(hyperparams,debugging=debugging)
        
        if Xstar_r !=None:
            self.covar_r.Xcross = Xstar_r
        
        Kstar_r = self.covar_r.Kcross(hyperparams['covar_r'])
        Kstar_c = self.covar_c.K(hyperparams['covar_c'])

        KinvY = SP.dot(KV['U_r'],SP.dot(KV['Ytilde'],KV['U_c'].T))
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
            
        if not(self._is_cached(hyperparams,keys=['covar_c'])):
            K_c = self.covar_c.K(hyperparams['covar_c'])
            S_c,U_c = LA.eigh(K_c)
            self._covar_cache['K_c'] = K_c
            self._covar_cache['U_c'] = U_c
            self._covar_cache['S_c'] = S_c
        else:
            K_c = self._covar_cache['K_c']
            U_c = self._covar_cache['U_c']
            S_c = self._covar_cache['S_c']
            
        if not(self._is_cached(hyperparams,keys=['covar_r'])):
            K_r = self.covar_r.K(hyperparams['covar_r'])
            S_r,U_r = LA.eigh(K_r)
            self._covar_cache['K_r'] = K_r
            self._covar_cache['U_r'] = U_r
            self._covar_cache['S_r'] = S_r
        else:
            K_r = self._covar_cache['K_r']
            U_r = self._covar_cache['U_r']
            S_r = self._covar_cache['S_r']

        S = SP.kron(S_c,S_r) + self.likelihood.Kdiag(hyperparams['lik'],self.nt)
        UYU = SP.dot(U_r.T,SP.dot(self.Y,U_c))
        YtildeVec = (1./S)*ravel(UYU)
        self._covar_cache['S'] = S
        self._covar_cache['UYU'] = UYU
        self._covar_cache['Ytilde'] = unravel(YtildeVec,self.n,self.t)

        
        if debugging:
            # test ravel operations
            UYUvec = ravel(UYU)
            UYU2 = unravel(UYUvec,self.n,self.t)
            SP.allclose(UYU2,UYU)

            # needed later
            Yvec = ravel(self.Y)
            K_noise = self.likelihood.K(hyperparams['lik'],self.nt) # only works for iid noise
            K = SP.kron(K_c,K_r) + K_noise
            #L = LA.cholesky(K).T
            L = jitChol(K)[0].T # lower triangular
            alpha = LA.cho_solve((L,True),Yvec)
            alpha2D = SP.reshape(alpha,(self.nt,1))
            Kinv = LA.cho_solve((L,True),SP.eye(self.nt))
            W = Kinv - SP.dot(alpha2D,alpha2D.T)
            self._covar_cache['Yvec'] = Yvec
            self._covar_cache['K'] = K
            self._covar_cache['Kinv'] = Kinv
            self._covar_cache['L'] = L
            self._covar_cache['alpha'] = alpha
            self._covar_cache['W'] = W

        self._covar_cache['hyperparams'] = copy.deepcopy(hyperparams)
        return self._covar_cache
        

    def _LML_covar(self,hyperparams,debugging=False):
        """
        log marginal likelihood
        """
        try:
            KV = self.get_covariances(hyperparams,debugging=debugging)
        except LA.LinAlgError:
            LG.error('linalg exception in _LML_covar')
            return 1E6
        except ValueError:
            LG.error('value error in _LML_covar')
            return 1E6
 
        lml_quad = 0.5*(KV['Ytilde']*KV['UYU']).sum()
        lml_det =  0.5 * SP.log(KV['S']).sum()
        lml_const = 0.5*self.n*self.t*(SP.log(2*SP.pi))
        
        if debugging:
            # do calculation without kronecker tricks and compare
            _lml_quad = 0.5 * (KV['alpha']*KV['Yvec']).sum()
            _lml_det =  SP.log(SP.diag(KV['L'])).sum()
            assert SP.allclose(_lml_quad,lml_quad),  'ouch, quadratic form is wrong in _LMLcovar'
            assert SP.allclose(_lml_det, lml_det), 'ouch, ldet is wrong in _LML_covar'
        
        lml = lml_quad + lml_det + lml_const

        return lml

    def _LMLgrad_covar(self,hyperparams,debugging=False):
        """
        evaluates the gradient of the log marginal likelihood with respect to the
        hyperparameters of the covariance function
        """
        try:
            KV = self.get_covariances(hyperparams,debugging=debugging)
        except LA.LinAlgError:
            LG.error('linalg exception in _LMLgrad_covar')
            return {'covar_r':SP.zeros(len(hyperparams['covar_r'])),'covar_c':SP.zeros(len(hyperparams['covar_c'])),'covar_r':SP.zeros(len(hyperparams['covar_r']))}
        except ValueError:
            LG.error('value error in _LMLgrad_covar')
            return {'covar_r':SP.zeros(len(hyperparams['covar_r'])),'covar_c':SP.zeros(len(hyperparams['covar_c'])),'covar_r':SP.zeros(len(hyperparams['covar_r']))}
 
        RV = {}
        Si = unravel(1./KV['S'],self.n,self.t)

        if 'covar_r' in hyperparams:
            theta = SP.zeros(len(hyperparams['covar_r']))
            for i in range(len(theta)):
                Kgrad_r = self.covar_r.Kgrad_theta(hyperparams['covar_r'],i)
                d=(KV['U_r']*SP.dot(Kgrad_r,KV['U_r'])).sum(0)
                LMLgrad_det = SP.dot(d,SP.dot(Si,KV['S_c']))
                UdKU = SP.dot(KV['U_r'].T,SP.dot(Kgrad_r,KV['U_r']))
                SYUdKU = SP.dot(UdKU,(KV['Ytilde']*SP.tile(KV['S_c'][SP.newaxis,:],(self.n,1))))
                LMLgrad_quad = - (KV['Ytilde']*SYUdKU).sum()
                LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
                theta[i] = LMLgrad

                if debugging:
                    Kd = SP.kron(KV['K_c'], Kgrad_r)
                    _LMLgrad = 0.5 * (KV['W']*Kd).sum()
                    assert SP.allclose(LMLgrad,_LMLgrad), 'ouch, gradient is wrong for covar_r'
                    
            RV['covar_r'] = theta

        if 'covar_c' in hyperparams:
            theta = SP.zeros(len(hyperparams['covar_c']))
            for i in range(len(theta)):
                Kgrad_c = self.covar_c.Kgrad_theta(hyperparams['covar_c'],i)

                d=(KV['U_c']*SP.dot(Kgrad_c,KV['U_c'])).sum(0)
                LMLgrad_det = SP.dot(KV['S_r'],SP.dot(Si,d))

                UdKU = SP.dot(KV['U_c'].T,SP.dot(Kgrad_c,KV['U_c']))
                SYUdKU = SP.dot((KV['Ytilde']*SP.tile(KV['S_r'][:,SP.newaxis],(1,self.t))),UdKU.T)
                LMLgrad_quad = -SP.sum(KV['Ytilde']*SYUdKU)
                LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
                theta[i] = LMLgrad
            
                if debugging:
                    Kd = SP.kron(Kgrad_c, KV['K_r'])
                    _LMLgrad = 0.5 * (KV['W']*Kd).sum()
                    assert SP.allclose(LMLgrad,_LMLgrad), 'ouch, gradient is wrong for covar_c'
                    
                RV['covar_c'] = theta

        return RV

    def _LMLgrad_lik(self,hyperparams,debugging=False):
        """
        evaluates the gradient of the log marginal likelihood with respect to
        the hyperparameters of the likelihood function
        """
        try:
            KV = self.get_covariances(hyperparams,debugging=debugging)
        except LA.LinAlgError:
            LG.error('linalg exception in _LML_covar')
            return {'lik':SP.zeros(len(hyperparams['lik']))}
        except ValueError:
            LG.error('value error in _LML_covar')
            return {'lik':SP.zeros(len(hyperparams['lik']))}
        
        YtildeVec = ravel(KV['Ytilde'])
        Kd_diag = self.likelihood.Kdiag_grad_theta(hyperparams['lik'],self.nt,0)

        LMLgrad_det = ((1./KV['S'])*Kd_diag).sum()
        LMLgrad_quad = -(YtildeVec**2 * Kd_diag).sum()
        LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
        
        if debugging:
            W = KV['W']
            Kd = self.likelihood.Kgrad_theta(hyperparams['lik'],self.nt,0)
            _LMLgrad = 0.5 * (W*Kd).sum()
            assert SP.allclose(LMLgrad,_LMLgrad), 'ouch, gradient is wrong for likelihood param'

        return {'lik':SP.array([LMLgrad])}

    def _LMLgrad_x(self,hyperparams,debugging=False):
        """
        evaluates the gradient of the log marginal likelihood with respect to
        the latent factors
        """
        try:
            KV = self.get_covariances(hyperparams,debugging=debugging)
        except LA.LinAlgError:
            LG.error('linalg exception in _LML_covar')
            RV = {}
            if 'X_r' in hyperparams:
                RV['X_r'] = SP.zeros(hyperparams['X_r'].shape)
            if 'X_c' in hyperparams:
                RV['X_c'] = SP.zeros(hyperparams['X_c'].shape)
            return RV
        except ValueError:
            LG.error('value error in _LML_covar')
            RV = {}
            if 'X_r' in hyperparams:
                RV['X_r'] = SP.zeros(hyperparams['X_r'].shape)
            if 'X_c' in hyperparams:
                RV['X_c'] = SP.zeros(hyperparams['X_c'].shape)
            return RV
       
        RV = {}
        if 'X_r' in hyperparams:
            LMLgrad = SP.zeros((self.n,self.covar_r.n_dimensions))
            LMLgrad_det = SP.zeros((self.n,self.covar_r.n_dimensions))
            LMLgrad_quad = SP.zeros((self.n,self.covar_r.n_dimensions))

            SS = SP.dot(unravel(1./KV['S'],self.n,self.t),KV['S_c'])
            UY = SP.dot(KV['U_r'],KV['Ytilde'])
            UYSYU = SP.dot(UY,SP.dot(SP.diag(KV['S_c']),UY.T))
            for d in xrange(self.covar_r.n_dimensions):
                Kd_grad = self.covar_r.Kgrad_x(hyperparams['covar_r'],d)
                # calculate gradient of logdet
                URU = SP.dot(Kd_grad.T,KV['U_r'])*KV['U_r']
                LMLgrad_det[:,d] = 2*SP.dot(URU,SS.T)
                # calculate gradient of squared form
                LMLgrad_quad[:,d] = -2*(UYSYU*Kd_grad).sum(0)
            LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
            RV['X_r'] = LMLgrad
            
            if debugging:
                 _LMLgrad = SP.zeros((self.n,self.covar_r.n_dimensions))
                 for n in xrange(self.n):
                     for d in xrange(self.covar_r.n_dimensions):
                         Kgrad_x = self.covar_r.Kgrad_x(hyperparams['covar_r'],d,n)
                         Kgrad_x = SP.kron(KV['K_c'],Kgrad_x)
                         _LMLgrad[n,d] = 0.5*(KV['W']*Kgrad_x).sum()
                 assert SP.allclose(LMLgrad,_LMLgrad), 'ouch, gradient is wrong for X_r'

        if 'X_c' in hyperparams:
            LMLgrad = SP.zeros((self.t,self.covar_c.n_dimensions))
            LMLgrad_quad = SP.zeros((self.t,self.covar_c.n_dimensions))
            LMLgrad_det = SP.zeros((self.t,self.covar_c.n_dimensions))

            SS = SP.dot(KV['S_r'],unravel(1./KV['S'],self.n,self.t))
            UY = SP.dot(KV['U_c'],KV['Ytilde'].T)
            UYSYU = SP.dot(UY,SP.dot(SP.diag(KV['S_r']),UY.T))
            for d in xrange(self.covar_c.n_dimensions):
                Kd_grad = self.covar_c.Kgrad_x(hyperparams['covar_c'],d)
                # calculate gradient of logdet
                UCU = SP.dot(Kd_grad.T,KV['U_c'])*KV['U_c']
                LMLgrad_det[:,d] = 2*SP.dot(SS,UCU.T)
                # calculate gradient of squared form
                LMLgrad_quad[:,d] = -2*(UYSYU*Kd_grad).sum(0)
                
            LMLgrad = 0.5*(LMLgrad_det + LMLgrad_quad)
            RV['X_c'] = LMLgrad
            
            if debugging:
                 _LMLgrad = SP.zeros((self.t,self.covar_c.n_dimensions))
                 for n in xrange(self.t):
                     for d in xrange(self.covar_c.n_dimensions):
                         Kgrad_x = self.covar_c.Kgrad_x(hyperparams['covar_c'],d,n)
                         Kgrad_x = SP.kron(Kgrad_x,KV['K_r'])
                         _LMLgrad[n,d] = 0.5*(KV['W']*Kgrad_x).sum()
                 assert SP.allclose(LMLgrad,_LMLgrad), 'ouch, gradient is wrong for X_c'
    
        return RV
