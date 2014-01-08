import sys
import core.likelihood.likelihood_base as likelihood_base
import pdb
import scipy as SP
import unittest

class TestLikelihood(unittest.TestCase):

    def setUp(self):
        self.n_dimensions = 10
       
        
    def test_likelihood(self):
        theta = SP.array([SP.random.randn()**2])
        theta_hat = SP.exp(2*theta[0])

        _K = theta_hat*SP.eye(self.n_dimensions)
        _Kgrad = 2*theta_hat*SP.eye(self.n_dimensions)

        lik = likelihood_base.GaussIsoLik()
        assert SP.allclose(_K,lik.K(theta,self.n_dimensions)), 'ouch, covariance is wrong'
        assert SP.allclose(_Kgrad,lik.Kgrad_theta(theta,self.n_dimensions,0)), 'ouch, gradient of covariance is wrong'
    
     
if __name__ == "__main__":
    unittest.main()
    

