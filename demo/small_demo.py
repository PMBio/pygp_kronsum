import pdb
import sys
import time
import scipy as SP

sys.path.append('../')

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
import core.priors.priors as prior
import experiments.initialize as initialize
from core.data_term import IdentDT

import matplotlib.pylab as PLT

if __name__ == "__main__":
    # settings
    n_latent = 1
    n_tasks = 10
    n_train = 100
    n_dimensions = 100

    # initialize covariance functions
    covar_c = lowrank.LowRankCF(n_dimensions=n_latent)
    covar_s = lowrank.LowRankCF(n_dimensions=n_latent)
    covar_r = linear.LinearCF(n_dimensions=n_dimensions)
    covar_o = diag.DiagIsoCF(n_dimensions = n_dimensions)

    # true parameters
    X_c = SP.random.randn(n_tasks,n_latent)
    X_s = SP.random.randn(n_tasks,n_latent)
    X_r = SP.random.randn(n_train,n_dimensions)/SP.sqrt(n_dimensions)
    R = SP.dot(X_r,X_r.T)
    C = SP.dot(X_c,X_c.T) 
    Sigma = SP.dot(X_s,X_s.T)
    K = SP.kron(C,R) + SP.kron(Sigma,SP.eye(n_train))
    y = SP.random.multivariate_normal(SP.zeros(n_tasks*n_train),K)
    Y = SP.reshape(y,(n_train,n_tasks),order='F')
    Y = IdentDT(SP.reshape(y,(n_train,n_tasks),order='F'))
    
    # initialization parameters
    hyperparams, Ifilter, bounds = initialize.init('GPkronsum_LIN',Y.value().T,X_r,{'n_c':n_latent, 'n_sigma':n_latent})
    
    # initialize gp and its covariance functions
    covar_r.X = X_r
    covar_o.X = X_r
    covar_o._K = SP.eye(n_train)
    covar_s.X = hyperparams['X_s']
    covar_c.X = hyperparams['X_c']
    gp = gp_kronsum.KronSumGP(covar_c=covar_c, covar_r=covar_r, covar_s=covar_s, covar_o=covar_o)
    gp.setData(Y=Y)  

    # optimize hyperparameters
    t_start = time.time()
    hyperparams_opt,lml_opt = optimize_base.opt_hyper(gp,hyperparams, bounds=bounds,Ifilter=Ifilter)
    t_stop = time.time()
    print 'time(training): %.4f'%(t_stop-t_start)

    # compare
    SigmaOpt = covar_s.K(hyperparams_opt['covar_s'])
    COpt = covar_c.K(hyperparams_opt['covar_c'])

    fig = PLT.figure(1)
    fig.add_subplot(221)
    fig.subplots_adjust(hspace=0.5)
    
    PLT.imshow(C,interpolation='nearest')
    PLT.title('True Signal Covariance')
    PLT.xlabel('Tasks'); PLT.ylabel('Tasks')
    
    fig.add_subplot(222)
    PLT.imshow(Sigma,interpolation='nearest')
    PLT.title('True Noise Covariance')
    PLT.xlabel('Tasks'); PLT.ylabel('Tasks')
    
    fig.add_subplot(223)
    PLT.imshow(COpt, interpolation='nearest')
    PLT.title('Learnt Signal Covariance')
    PLT.xlabel('Tasks'); PLT.ylabel('Tasks')
    
    fig.add_subplot(224)
    PLT.imshow(SigmaOpt, interpolation='nearest')
    PLT.title('Learnt Noise Covariance')
    PLT.xlabel('Tasks'); PLT.ylabel('Tasks')
    
    PLT.show()
