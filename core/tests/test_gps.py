import sys
import pdb
import scipy as SP
import scipy.linalg as LA
import unittest

import core.likelihood.likelihood_base as likelihood_base
import core.gp.gp_base as gp_base
import core.gp.gplvm as gplvm
import core.gp.gp_kronprod as gp_kronprod
import core.gp.gp_kronsum as gp_kronsum
import core.gp.gp_kronsum_naive as gp_kronsum_naive
import core.covariance.linear as linear
import core.covariance.diag as diag
import core.covariance.lowrank as lowrank
import core.covariance.diag as diag
import core.optimize.optimize_base as optimize_base
import core.priors.priors as priors


import collections

class TestGPs(unittest.TestCase):

    def setUp(self):
        SP.random.seed(1)

        self.n_tasks = 5
        self.n_train = 40
        self.n_test = 10
        self.n = self.n_train + self.n_test
        self.n_dimensions = 20
        self.n_latent = 3

        X = SP.random.randn(self.n,self.n_dimensions)
        X/= SP.sqrt(self.n_dimensions)
        R = SP.dot(X,X.T)
        I = SP.eye(self.n_train + self.n_test)
        zero_mean = SP.zeros(self.n)

        Xtrain = X[:self.n_train]
        Xtest = X[self.n_train:]
        self.X = {'train':Xtrain, 'test':Xtest}

        # single GP
        Yuni = SP.random.multivariate_normal(zero_mean, R+I,1).T
        Ytrain = Yuni[:self.n_train]
        Ytest = Yuni[self.n_train:]
        self.Yuni = {'train':Ytrain, 'test':Ytest}

        # GPLVM
        self.Xlatent = SP.random.randn(self.n_tasks,self.n_latent)
        self.Xlatent /= SP.sqrt(self.n_latent)
        C = SP.dot(self.Xlatent,self.Xlatent.T)
        self.Ylatent = SP.random.multivariate_normal(SP.zeros(self.n_tasks), 0.9*C + 0.1*SP.eye(self.n_tasks),self.n).T

        # GP-Kronprod
        CkronR = SP.kron(C,R) + SP.eye(self.n_tasks * self.n)
        Ykronprod = SP.random.multivariate_normal(SP.zeros(self.n_tasks * self.n), CkronR + SP.eye(self.n_tasks * self.n),1).T
        Ykronprod = gp_kronprod.unravel(Ykronprod,self.n,self.n_tasks)
        self.Ykronprod = {'train':Ykronprod[:self.n_train], 'test':Ykronprod[self.n_train:]}

        # GP-Kronsum
        self.Xsigma = SP.random.randn(self.n_tasks,self.n_latent)
        self.Xsigma/= SP.sqrt(self.n_latent)
        Sigma = SP.dot(self.Xsigma, self.Xsigma.T)
        Ykronsum = SP.random.multivariate_normal(SP.zeros(self.n_tasks * self.n), CkronR + SP.kron(Sigma,I),1).T
        Ykronsum = gp_kronprod.unravel(Ykronsum,self.n,self.n_tasks)
        self.Ykronsum = {'train':Ykronsum[:self.n_train], 'test':Ykronsum[self.n_train:]}

        
    def test_gpbase(self):

        covar = linear.LinearCF(n_dimensions=self.n_dimensions)
        n_train = self.X['train'].shape[0]

        theta = 1E-1
        prior_cov = priors.GaussianPrior(key='covar',theta=SP.array([1.]))
        prior_lik = priors.GaussianPrior(key='lik',theta=SP.array([1.]))
        prior = priors.PriorList([prior_cov,prior_lik])
        
        lik = likelihood_base.GaussIsoLik()
        gp = gp_base.GP(covar_r=covar,likelihood=lik,prior=prior)
        gp.setData(Y=self.Yuni['train'],X=self.X['train'])

        # log likelihood and gradient derivation
        hyperparams = {'covar':SP.array([0.5]), 'lik':SP.array([0.5])}
        LML = gp.LML(hyperparams)
        LMLgrad = gp.LMLgrad(hyperparams)
        
        K = covar.K(hyperparams['covar']) + lik.K(hyperparams['lik'],n_train)
        Kgrad_covar = covar.Kgrad_theta(hyperparams['covar'],0)
        Kgrad_lik = lik.Kgrad_theta(hyperparams['lik'],n_train,0)

        KinvY = LA.solve(K,self.Yuni['train'])
        _LML = self.n_train/2*SP.log(2*SP.pi) + 0.5*SP.log(LA.det(K)) + 0.5*(self.Yuni['train']*KinvY).sum() + prior.LML(hyperparams)
        LMLgrad_covar = 0.5 * SP.trace(LA.solve(K, Kgrad_covar)) - 0.5*SP.dot(KinvY.T,SP.dot(Kgrad_covar,KinvY))
        LMLgrad_covar+= prior_cov.LMLgrad(hyperparams)['covar']
        LMLgrad_lik = 0.5 * SP.trace(LA.solve(K, Kgrad_lik)) - 0.5*SP.dot(KinvY.T,SP.dot(Kgrad_lik,KinvY))
        LMLgrad_lik+= prior_lik.LMLgrad(hyperparams)['lik']

        
        assert SP.allclose(LML,_LML), 'ouch, marginal log likelihood does not match'
        assert SP.allclose(LMLgrad['covar'], LMLgrad_covar), 'ouch, gradient with respect to theta does not match'
        assert SP.allclose(LMLgrad['lik'], LMLgrad_lik), 'ouch, gradient with respect to theta does not match'

        # predict
        Ystar = gp.predict(hyperparams,self.X['test'])
        Kstar = covar.Kcross(hyperparams['covar'])
        _Ystar = SP.dot(Kstar.T,LA.solve(K,self.Yuni['train'])).flatten()
        assert SP.allclose(Ystar,_Ystar), 'ouch, predictions, do not match'
        
        # optimize
        opts = {'gradcheck':True,'messages':False}
        hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp,hyperparams,opts=opts)
        
        # log likelihood and gradient derivation
        LML = gp.LML(hyperparams_opt)
        LMLgrad = gp.LMLgrad(hyperparams_opt)

        K = covar.K(hyperparams_opt['covar']) + lik.K(hyperparams_opt['lik'],n_train)
        Kgrad_covar = covar.Kgrad_theta(hyperparams_opt['covar'],0)
        Kgrad_lik = lik.Kgrad_theta(hyperparams_opt['lik'],n_train,0)
        
        KinvY = LA.solve(K,self.Yuni['train'])
        _LML = self.n_train/2*SP.log(2*SP.pi) + 0.5*SP.log(LA.det(K)) + 0.5*(self.Yuni['train']*KinvY).sum() + prior.LML(hyperparams_opt)
        LMLgrad_covar = 0.5 * SP.trace(LA.solve(K, Kgrad_covar)) - 0.5*SP.dot(KinvY.T,SP.dot(Kgrad_covar,KinvY))
        LMLgrad_covar+= prior_cov.LMLgrad(hyperparams_opt)['covar']
        LMLgrad_lik = 0.5 * SP.trace(LA.solve(K, Kgrad_lik)) - 0.5*SP.dot(KinvY.T,SP.dot(Kgrad_lik,KinvY))
        LMLgrad_lik+= prior_lik.LMLgrad(hyperparams_opt)['lik']

        assert SP.allclose(LML,_LML), 'ouch, marginal log likelihood does not match'
        assert SP.allclose(LMLgrad['covar'], LMLgrad_covar), 'ouch, gradient with respect to theta does not match'
        assert SP.allclose(LMLgrad['lik'], LMLgrad_lik), 'ouch, gradient with respect to theta does not match'
   
        # predict
        Ystar = gp.predict(hyperparams_opt,self.X['test'])
        Kstar = covar.Kcross(hyperparams_opt['covar'])
        _Ystar = SP.dot(Kstar.T,LA.solve(K,self.Yuni['train'])).flatten()
        assert SP.allclose(Ystar,_Ystar), 'ouch, predictions, do not match'
        
       
    def test_gplvm(self):
        covar = linear.LinearCF(n_dimensions=self.n_latent)
        lik = likelihood_base.GaussIsoLik()
        prior = priors.GaussianPrior(key='X',theta=SP.array([1.]))
        gp = gplvm.GPLVM(covar=covar,likelihood=lik,prior=prior)
        
        
        X0 = SP.random.randn(self.n_tasks, self.n_latent)
        X0 = self.Xlatent
        covar.X = X0
        gp.setData(Y=self.Ylatent)

        # gradient with respect to X
        hyperparams = {'covar':SP.array([0.5]), 'lik':SP.array([0.5]),'X':X0}
        
        LML = gp.LML(hyperparams)
        LMLgrad = gp.LMLgrad(hyperparams)

        LMLgrad_x = SP.zeros((self.n_tasks, self.n_latent))
        W = gp.get_covariances(hyperparams)['W']
        for d in xrange(self.n_latent):
            for n in xrange(self.n_tasks):
                Knd_grad = covar.Kgrad_x(hyperparams['covar'],d,n)
                LMLgrad_x[n,d] = 0.5*(W*Knd_grad).sum()
                
        LMLgrad_x += prior.LMLgrad(hyperparams)['X']
        assert SP.allclose(LMLgrad['X'],LMLgrad_x), 'ouch, gradient with respect to X is wrong'

        # optimize
        opts = {'gradcheck':True}
        hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp,hyperparams,opts=opts)
        Ktrue = SP.dot(self.Xlatent,self.Xlatent.T)
        covar.X = hyperparams_opt['X']
        Kest = covar.K(hyperparams_opt['covar'])

        # gradient with respect to X
        LML = gp.LML(hyperparams_opt)
        LMLgrad = gp.LMLgrad(hyperparams_opt)
        LMLgrad_x = SP.zeros((self.n_tasks, self.n_latent))
        W = gp.get_covariances(hyperparams_opt)['W']
        for d in xrange(self.n_latent):
            for n in xrange(self.n_tasks):
                Knd_grad = covar.Kgrad_x(hyperparams_opt['covar'],d,n)
                LMLgrad_x[n,d] = 0.5*(W*Knd_grad).sum()
        LMLgrad_x += prior.LMLgrad(hyperparams_opt)['X']
        assert SP.allclose(LMLgrad['X'],LMLgrad_x), 'ouch, gradient with respect to X is wrong'


    def test_gpkronsum(self):
        covar_c = lowrank.LowRankCF(n_dimensions=self.n_latent)
        covar_r = lowrank.LowRankCF(n_dimensions=self.n_dimensions)
        covar_s = lowrank.LowRankCF(n_dimensions=self.n_latent)
        covar_o = lowrank.LowRankCF(n_dimensions = self.n_dimensions)

        X0_c = SP.random.randn(self.n_tasks,self.n_latent)
        X0_s = SP.random.randn(self.n_tasks,self.n_latent)
        X0_r = SP.random.randn(self.n_train,self.n_dimensions)
        X0_o = SP.random.randn(self.n_train,self.n_dimensions)

        gp = gp_kronsum.KronSumGP(covar_c=covar_c, covar_r=covar_r, covar_s=covar_s, covar_o=covar_o)
        gp.setData(Y=self.Ykronsum['train'])

        gp2 = gp_kronsum_naive.KronSumGP(covar_c=covar_c,covar_r=covar_r,covar_s=covar_s,covar_o=covar_o)
        gp2.setData(Y=self.Ykronsum['train'])
        
        hyperparams = {'covar_c':SP.array([0.5,0.5]), 'X_c':X0_c, 'covar_r':SP.array([0.5,0.5]), 'X_r':X0_r,
                       'covar_s':SP.array([0.5,0.5]), 'X_s':X0_s, 'covar_o':SP.array([0.5,0.5]), 'X_o':X0_o}

        yhat = gp.predict(hyperparams,Xstar_r=self.X['test'],debugging=True)
        lml = gp._LML_covar(hyperparams,debugging=True)
        grad = {}
        grad.update(gp._LMLgrad_c(hyperparams, debugging=True))
        grad.update(gp._LMLgrad_r(hyperparams, debugging=True))
        grad.update(gp._LMLgrad_o(hyperparams, debugging=True))
        grad.update(gp._LMLgrad_s(hyperparams, debugging=True))
        

        yhat2 = gp2.predict(hyperparams,Xstar_r=self.X['test'])
        lml2 = gp2._LML_covar(hyperparams)
        grad2 = {}
        grad2.update(gp2._LMLgrad_covar(hyperparams))
        grad2.update(gp2._LMLgrad_x(hyperparams))

        assert SP.allclose(yhat,yhat2), 'predictions does not match'
        assert SP.allclose(lml,lml2), 'log likelihood does not match'
        for key in grad.keys():
            assert SP.allclose(grad[key],grad2[key]), 'gradient with respect to x does not match'
            
        covar_o = diag.DiagIsoCF(n_dimensions = self.n_dimensions)
        gp = gp_kronsum.KronSumGP(covar_c=covar_c, covar_r=covar_r, covar_s=covar_s, covar_o=covar_o)
        gp.setData(Y=self.Ykronsum['train'],X_r=self.X['train'],X_o=self.X['train'])

        gp2 = gp_kronsum_naive.KronSumGP(covar_c=covar_c, covar_r=covar_r, covar_s=covar_s, covar_o=covar_o)
        gp2.setData(Y=self.Ykronsum['train'],X_r=self.X['train'],X_o=self.X['train'])
        
        hyperparams = {'covar_c':SP.array([0.5,0.5]), 'X_c':X0_c, 'covar_r':SP.array([0.5,0.5]),
                       'covar_s':SP.array([0.5,0.5]), 'X_s':X0_s, 'covar_o':SP.array([0.5])}
        
        bounds = {'covar_c':SP.array([[-5,+5]]*2), 'covar_r':SP.array([[-5,+5]]*2), 'covar_s':SP.array([[-5,+5]]*2), 'covar_o':SP.array([[-5,+5]])}
        opts = {'gradcheck':True}
        import time
        t_start = time.time()
        hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp,hyperparams,opts=opts, bounds=bounds)
        t_stop = time.time()
        print 'time(training): %.4f'%(t_stop-t_start)

        t_start = time.time()
        hyperparams_opt2, lml_opt2 = optimize_base.opt_hyper(gp2,hyperparams,opts=opts, bounds=bounds)
        t_stop = time.time()
        
        print 'time(training): %.4f'%(t_stop-t_start)
        assert SP.allclose(lml_opt,lml_opt2), 'ouch, optimization did fail'
        
        gp._invalidate_cache() # otherwise debugging parameters are not up to date!
        yhat = gp.predict(hyperparams_opt,Xstar_r=self.X['test'],debugging=True)
        lml = gp._LML_covar(hyperparams_opt,debugging=True)
        grad = {}
        grad.update(gp._LMLgrad_c(hyperparams_opt, debugging=True))
        grad.update(gp._LMLgrad_r(hyperparams_opt, debugging=True))
        grad.update(gp._LMLgrad_o(hyperparams_opt, debugging=True))
        grad.update(gp._LMLgrad_s(hyperparams_opt, debugging=True))
        

        yhat2 = gp2.predict(hyperparams_opt,Xstar_r=self.X['test'])
        lml2 = gp2._LML_covar(hyperparams_opt)
        grad2 = {}
        grad2.update(gp2._LMLgrad_covar(hyperparams_opt))
        grad2.update(gp2._LMLgrad_x(hyperparams_opt))

        assert SP.allclose(yhat,yhat2), 'predictions does not match'
        assert SP.allclose(lml,lml2), 'log likelihood does not match'
        for key in grad.keys():
            assert SP.allclose(grad[key],grad2[key]), 'gradient with respect to x does not match'

 
    def test_gpkronprod(self):
       # initialize
       covar_c = linear.LinearCF(n_dimensions=self.n_latent)
       covar_r = linear.LinearCF(n_dimensions=self.n_dimensions)
       X0_c = SP.random.randn(self.n_tasks,self.n_latent)
       
       lik = likelihood_base.GaussIsoLik()
       gp = gp_kronprod.KronProdGP(covar_c=covar_c, covar_r=covar_r, likelihood=lik)
       gp.setData(Y=self.Ykronprod['train'],X_r=self.X['train'])
       hyperparams = {'lik':SP.array([0.5]), 'X_c':X0_c, 'covar_r':SP.array([0.5]), 'covar_c':SP.array([0.5]), 'X_r':self.X['train']}
       # check predictions, likelihood and gradients
       gp.predict(hyperparams,Xstar_r=self.X['test'],debugging=True)

       gp._LML_covar(hyperparams,debugging=True)
       gp._LMLgrad_covar(hyperparams,debugging=True)
       gp._LMLgrad_lik(hyperparams,debugging=True)
       gp._LMLgrad_x(hyperparams,debugging=True)
       
       # optimize
       hyperparams = {'lik':SP.array([0.5]), 'X_c':X0_c, 'covar_r':SP.array([0.5]), 'covar_c':SP.array([0.5])}
       opts = {'gradcheck':True}
       hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp,hyperparams,opts=opts)
       Kest = covar_c.K(hyperparams_opt['covar_c'])

       # check predictions, likelihood and gradients
       gp._invalidate_cache() # otherwise debugging parameters are not up to date!
       gp.predict(hyperparams_opt,debugging=True,Xstar_r=self.X['test'])
       gp._LML_covar(hyperparams_opt,debugging=True)
       gp._LMLgrad_covar(hyperparams_opt,debugging=True)
       gp._LMLgrad_lik(hyperparams_opt,debugging=True)
       gp._LMLgrad_x(hyperparams_opt,debugging=True)
       
       
       
       
     
if __name__ == "__main__":
    unittest.main()
    

