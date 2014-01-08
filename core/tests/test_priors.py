import unittest
import core.priors.priors as priors
import scipy as SP
import scipy.optimize as OPT
import pdb

class TestGPs(unittest.TestCase):

    def test_gaussian_prior(self):
        gp = priors.GaussianPrior(key='X',theta = SP.array([2]))
        def f(Xvec):
            d = {'X':Xvec}
            return gp.LML(d)

        def grad(Xvec):
            d = {'X':Xvec}
            return gp.LMLgrad(d)['X']

        err = OPT.check_grad(f,grad,SP.random.randn(10))
        assert err<1E-6, 'ouch, gradient does not match'

        
if __name__ == "__main__":
    unittest.main()
    
