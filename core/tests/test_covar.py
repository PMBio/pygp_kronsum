import sys
import pdb
import scipy as SP
import unittest
import scipy.optimize as OPT

import core.covariance.diag as diag
import core.covariance.fixed as fixed
import core.covariance.linear as linear
import core.covariance.lowrank as lowrank
import core.covariance.se as se

class TestCovar(unittest.TestCase):

    def setUp(self):
        self.n_train = 90
        self.n_test = 10
        self.n_dimensions = 10
        self.Xtrain = SP.random.randn(self.n_train,self.n_dimensions)
        self.Xtest = SP.random.randn(self.n_test,self.n_dimensions)


    def test_se(self):
        theta = SP.random.randn(2)**2
        theta_hat = SP.exp(2*theta)
        cov = se.SqExpCF(n_dimensions=self.n_dimensions)
        cov.X = self.Xtrain
        cov.Xcross = self.Xtest

        K = cov.K(theta)
        Kcross = cov.Kcross(theta)

        # compute by hand
        _K = SP.zeros((self.n_train,self.n_train))
        _Kcross = SP.zeros((self.n_train,self.n_test))
        for i in range(self.n_train):
            for j in range(self.n_train):
                _K[i,j] = SP.sum((cov.X[i] - cov.X[j])**2)
            for j in range(self.n_test):
                _Kcross[i,j] = SP.sum((cov.X[i] - cov.Xcross[j])**2)

        _K =theta_hat[0] * SP.exp(-0.5 * _K/theta_hat[1])
        _Kcross = theta_hat[0] * SP.exp(-0.5 * _Kcross/theta_hat[1])
        assert SP.allclose(K,_K), 'ouch, covariance matrix is wrong' 
        assert SP.allclose(Kcross,_Kcross), 'ouch, cross covariance matrix is wrong'


        def f(theta_i):
            _theta = SP.copy(theta)
            _theta[i] = theta_i[0]
            return cov.K(_theta)[j,k]
        
        def grad(theta_i):
            _theta = SP.copy(theta)
            _theta[i] = theta_i[0]
            return cov.Kgrad_theta(_theta,i)[j,k]

        err_max = 0
        for i in range(2):
            theta_i = SP.array([theta[i]])
            for j in range(self.n_train):
                for k in range(self.n_train):
                    err = OPT.check_grad(f,grad,theta_i)
                    assert err<1E-5, 'ouch, gradient does not match'
  
        
        
    def test_diagonal_iso(self):
        theta = SP.array([SP.random.randn()**2])
        theta_hat = SP.exp(2*theta)
        cov = diag.DiagIsoCF(n_dimensions=self.n_dimensions)
        cov.X = self.Xtrain
        cov.Xcross = self.Xtest
        K = cov.K(theta)
        Kcross = cov.Kcross(theta)
        Kgrad_theta = cov.Kgrad_theta(theta,0)
        Kgrad_x = cov.Kgrad_x(theta,0)
        assert SP.allclose(K,theta_hat*SP.eye(self.n_train)), 'ouch, covariance matrix is wrong'
        assert SP.allclose(Kcross,SP.zeros((self.n_train,self.n_test))), 'ouch, cross covariance matrix is wrong'
        assert SP.allclose(Kgrad_theta, 2*theta_hat*SP.eye(self.n_train)), 'ouch, gradient with respect to theta is wrong' 
        assert SP.allclose(Kgrad_x, SP.zeros((self.n_train,self.n_train))), 'ouch, gradient with respect to x is wrong'

    def test_diagonal_ard(self):
        theta_vec = SP.array(SP.random.randn(self.n_train)**2)
        theta_vec_hat = SP.exp(2*theta_vec)
        
        cov = diag.DiagArdCF(n_dimensions=self.n_dimensions,n_hyperparameters=self.n_train)
        cov.X = self.Xtrain
        cov.Xcross = self.Xtest
        K = cov.K(theta_vec)
        Kcross = cov.Kcross(theta_vec)
        Kgrad_x = cov.Kgrad_x(theta_vec,0)
        assert SP.allclose(K,SP.diag(theta_vec_hat)), 'ouch, covariance matrix is wrong'
        assert SP.allclose(Kcross,SP.zeros((self.n_train,self.n_test))), 'ouch, cross covariance matrix is wrong'
        for i in range(self.n_dimensions):
            Kgrad_theta = cov.Kgrad_theta(theta_vec,i)
            Ktmp = SP.zeros((self.n_train,self.n_train))
            Ktmp[i,i] = 1
            assert SP.allclose(Kgrad_theta, 2*theta_vec_hat[i]*Ktmp), 'ouch, gradient with respect to theta is wrong' 
        assert SP.allclose(Kgrad_x, SP.zeros((self.n_train,self.n_train))), 'ouch, gradient with respect to x is wrong'

    def test_fixed(self):
        theta = SP.array([SP.random.randn()**2])
        theta_hat = SP.exp(2*theta)
        
        _K = SP.dot(self.Xtrain,self.Xtrain.T)
        _Kcross = SP.dot(self.Xtrain,self.Xtest.T)
        
        cov = fixed.FixedCF(n_dimensions=self.n_dimensions)
        cov._K = _K
        cov._Kcross = _Kcross
        
        K = cov.K(theta)
        Kcross = cov.Kcross(theta)
        Kgrad_x = cov.Kgrad_x(theta,0)
        Kgrad_theta = cov.Kgrad_theta(theta,0)
        
        assert SP.allclose(K, theta_hat*_K), 'ouch covariance matrix is wrong'
        assert SP.allclose(Kgrad_x, SP.zeros((self.n_train,self.n_train))), 'ouch gradient with respect to x is wrong'
        assert SP.allclose(Kgrad_theta, 2*theta_hat*_K), 'ouch, gradient with respect to theta is wrong'
        assert SP.allclose(Kcross, theta_hat*_Kcross), 'ouch, cross covariance is wrong'


    def test_linear(self):
        theta = SP.array([SP.random.randn()**2])
        theta_hat = SP.exp(2*theta)

        _K = SP.dot(self.Xtrain,self.Xtrain.T)
        _Kcross = SP.dot(self.Xtrain,self.Xtest.T)
        
        cov = linear.LinearCF(n_dimensions=self.n_dimensions)
        cov.X = self.Xtrain
        cov.Xcross = self.Xtest
        
        K = cov.K(theta)
        Kcross = cov.Kcross(theta)
        Kgrad_x = cov.Kgrad_x(theta,0)
        Kgrad_theta = cov.Kgrad_theta(theta,0)
        
        assert SP.allclose(K, theta_hat*_K), 'ouch covariance matrix is wrong'
        assert SP.allclose(Kgrad_theta, 2*theta_hat*_K), 'ouch, gradient with respect to theta is wrong'
        assert SP.allclose(Kcross, theta_hat*_Kcross), 'ouch, cross covariance is wrong'

        # gradient with respect to latent factors
        # for each entry
        for i in range(self.n_dimensions):
            for j in range(self.n_train):
                Xgrad = SP.zeros(self.Xtrain.shape)
                Xgrad[j,i] = 1
                _Kgrad_x =  theta_hat*(SP.dot(Xgrad,self.Xtrain.T) + SP.dot(self.Xtrain,Xgrad.T))
                Kgrad_x = cov.Kgrad_x(theta,i,j)
                assert SP.allclose(Kgrad_x,_Kgrad_x), 'ouch, gradient with respect to x is wrong for entry [%d,%d]'%(i,j)


    def test_lowrank_iso(self):
        theta = SP.array(SP.random.randn(2)**2)
        theta_hat = SP.exp(2*theta)

        _K = theta_hat[0]*SP.dot(self.Xtrain,self.Xtrain.T) + theta_hat[1]*SP.eye(self.n_train)
        _Kcross = theta_hat[0]*SP.dot(self.Xtrain,self.Xtest.T)
        _Kgrad_theta = []
        _Kgrad_theta.append(2*theta_hat[0]*SP.dot(self.Xtrain,self.Xtrain.T) )
        _Kgrad_theta.append(2*theta_hat[1]*SP.eye(self.n_train))

        cov = lowrank.LowRankCF(self.n_dimensions)
        cov.X = self.Xtrain
        cov.Xcross = self.Xtest
        
        K = cov.K(theta)
        Kcross = cov.Kcross(theta)

        assert SP.allclose(K,_K), 'ouch, covariance matrix is wrong'
        assert SP.allclose(Kcross,_Kcross), 'ouch, cross covariance matrix is wrong'
        assert SP.allclose(_Kgrad_theta[0],cov.Kgrad_theta(theta,0))
        assert SP.allclose(_Kgrad_theta[1],cov.Kgrad_theta(theta,1))

        # gradient with respect to latent factors
        for i in range(self.n_dimensions):
            for j in range(self.n_train):
                Xgrad = SP.zeros(self.Xtrain.shape)
                Xgrad[j,i] = 1
                _Kgrad_x =  theta_hat[0]*(SP.dot(Xgrad,self.Xtrain.T) + SP.dot(self.Xtrain,Xgrad.T))
                Kgrad_x = cov.Kgrad_x(theta,i,j)
                assert SP.allclose(Kgrad_x,_Kgrad_x), 'ouch, gradient with respect to x is wrong for entry [%d,%d]'%(i,j)

        
    def test_lowrank_ard(self):
        
        theta = SP.array(SP.random.randn(1+self.n_train)**2)
        theta_hat = SP.exp(2*theta)

        _K = theta_hat[0]*SP.dot(self.Xtrain,self.Xtrain.T) + SP.diag(theta_hat[1:])
        _Kcross = theta_hat[0]*SP.dot(self.Xtrain,self.Xtest.T)
        _Kgrad_theta = 2*theta_hat[0]*SP.dot(self.Xtrain,self.Xtrain.T)
        
        cov = lowrank.LowRankArdCF(n_dimensions=self.n_dimensions,n_hyperparameters=self.n_train+1)
        cov.X = self.Xtrain
        cov.Xcross = self.Xtest
        K = cov.K(theta)
        Kcross = cov.Kcross(theta)

        assert SP.allclose(K,_K), 'ouch, covariance matrix is wrong'
        assert SP.allclose(Kcross,_Kcross), 'ouch, cross covariance matrix is wrong'
        assert SP.allclose(_Kgrad_theta,cov.Kgrad_theta(theta,0)), 'ouch gradient with respect to theta[0] is wrong'

        # gradient with respect to parameters of the diagonal matrix
        for i in range(self.n_train):
            Kgrad_theta = cov.Kgrad_theta(theta,i+1)
            _Kgrad_theta = SP.zeros(Kgrad_theta.shape)
            _Kgrad_theta[i,i] = 2*theta_hat[i+1]
            assert SP.allclose(Kgrad_theta, _Kgrad_theta), 'ouch gradient with respect to theta[%d] is wrong'%(i+1)
            
        # gradient with respect to latent factors
        for i in range(self.n_dimensions):
            for j in range(self.n_train):
                Xgrad = SP.zeros(self.Xtrain.shape)
                Xgrad[j,i] = 1
                _Kgrad_x =  theta_hat[0]*(SP.dot(Xgrad,self.Xtrain.T) + SP.dot(self.Xtrain,Xgrad.T))
                Kgrad_x = cov.Kgrad_x(theta,i,j)
                assert SP.allclose(Kgrad_x,_Kgrad_x), 'ouch, gradient with respect to x is wrong for entry [%d,%d]'%(i,j)
                

     
if __name__ == "__main__":
    unittest.main()
    

