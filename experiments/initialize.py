import scipy as SP
import pdb
import sys
import utils
import scipy.linalg as LA

sys.path.append('../')
import core.covariance.linear as linear

def init(method,Y,X_r,opts):
    """
    initialized hyperparams, bounds and filters
    """
    # init
    if method=='GPbase_LIN':
        hyperparams = {'covar':SP.array([SP.random.randn()**2]),'lik':SP.array([SP.random.randn()**2])}
        Ifilter = None
        bounds = {'covar': SP.array([[-5,+5]]),'lik': SP.array([[-5,+5]])}

    if method=='GPpool_LIN':
        covar0_c,lik0,covar0_r = init_GPpool(Y,X_r)
        hyperparams = {'covar_c':covar0_c,'covar_r':covar0_r,'lik':lik0}
        covar0_r = SP.ones(covar0_r.size)
        covar0_r[0] = 0
        Ifilter = {'covar_c':SP.ones(covar0_c.size),'covar_r':covar0_r,'lik':SP.ones(lik0.size)}
        bounds = {'covar_c':SP.array([[-5,+5]]*covar0_c.size), 'lik':SP.array([[-5,+5]])}

    if method=='GPkronsum_LIN':
        X0_c,X0_s,covar0_c,covar0_s,covar0_r = init_GPkronsum(Y,X_r,opts['n_c'],opts['n_sigma'])
        hyperparams = {'covar_c':covar0_c,'covar_r':covar0_r,'covar_s':covar0_s,'covar_o':SP.log([1]),'X_c':X0_c,'X_s':X0_s}
        # do not optimize amplitude of R,Omega (already done over C, Sigma)
        Ifilter_r = SP.ones(covar0_r.size)
        Ifilter_r[0] = 0
        Ifilter_c = SP.zeros(1)
        Ifilter =  {'X_s':SP.ones(X0_s.shape),'X_c':SP.ones(X0_c.shape),'covar_s':SP.ones(covar0_s.shape),'covar_r':Ifilter_r,'covar_c':SP.ones(covar0_c.shape),'covar_o':Ifilter_c}
        # bound individual effects to prevent singularities
        bounds = {'covar_c': SP.array([[-5,+5]]*covar0_c.size),'covar_s': SP.array([[-5,+5]]*covar0_s.size)}

    if method=='GPkronprod_LIN':
        X0_c,covar0_c, lik0,covar0_r = init_GPkronprod(Y,X_r,opts['n_c'])
        hyperparams = {'covar_c':covar0_c,'covar_r':covar0_r,'lik':lik0,'X_c':X0_c}
        covar0_r = SP.ones(covar0_r.size)
        covar0_r[0] = 0
        Ifilter = {'X_c':SP.ones(X0_c.shape),'covar_c':SP.ones(covar0_c.size),'covar_r':covar0_r,'lik':SP.ones(lik0.size)}
        bounds = {'covar_c':SP.array([[-5,+5]]*covar0_c.size), 'lik':SP.array([[-5,+5]])}
                  
        
    return hyperparams,Ifilter,bounds


def init_GPpool(Y,X_r):
    """
    init parameter for kron(Ones,R) + sigma*I
    """
    covar0_r = SP.array([0])
    cov = SP.cov(Y)

    # split into signal and noise
    ratio = SP.random.rand(2)
    ratio/= ratio.sum()
    lik0 = ratio[0] * SP.diag(cov).min()
    covar0_c = ratio[1] * SP.diag(cov).min()

    # transform as neccessary
    covar0_c = 0.5*SP.log(SP.array([covar0_c]))
    lik0 = 0.5 * SP.log(SP.array([lik0]))

    return covar0_c,lik0,covar0_r

def init_GPkronprod(Y,X_r,n_c):
    """
    init parameters for kron(C + sigma I,R) + sigma*I
    """
    # build linear kernel with the features
    covar0_r = SP.array([0])
    covar_r = linear.LinearCF(n_dimensions=X_r.shape[1])
    covar_r.X = X_r
    R = covar_r.K(covar0_r)
    var_R = utils.getVariance(R)
    cov = SP.cov(Y)
    
    # split into likelihood and noise terms
    ratio = SP.random.rand(3)
    ratio/= ratio.sum()
    lik0 = ratio[0] * SP.diag(cov).min()
    covar0_c = ratio[1] * SP.diag(cov).min()

    # remaining variance is assigned to latent factors
    if n_c > 1:
        X0_c = SP.zeros((Y.shape[0],n_c))
        ratio = SP.random.rand(n_c)
        ratio/= ratio.sum()
        for i in range(n_c):
            # split further up
            X0_c[:,i] = SP.sign(SP.random.rand)*SP.sqrt(ratio[i]*(SP.diag(cov)-lik0 - covar0_c))
    else:
        X0_c = SP.sign(SP.random.rand)*SP.sqrt(SP.diag(cov) - lik0 - covar0_c)
    X0_c = SP.reshape(X0_c,(X0_c.shape[0],n_c))

    # check if variance of initial values match observed variance
    assert SP.allclose(SP.diag(cov),(X0_c**2).sum(1) + lik0 + covar0_c), 'ouch, something is wrong'

    # bring in correct format and transform as neccessary
    covar0_c = 0.5*SP.log(SP.array([1./var_R,covar0_c]))
    lik0 = 0.5 * SP.log(SP.array([lik0]))
    return X0_c,covar0_c,lik0,covar0_r



def init_GPkronsum(Y,X_r,n_c,n_sigma):
    """
    init parameters for kron(C + sigmaI,R) + kron(Sigma + sigmaI,Omega)

    input:
    Y       task matrix
    X_r     feature matrxi
    n_c     number of hidden factors in C
    n_sigma number of hidden factors in Sigma
    """
    n_t,n_s = Y.shape
    n_f = X_r.shape[1]

    # build linear kernel with the features
    covar_r = linear.LinearCF(n_dimensions=n_f)
    covar_r.X = X_r
    covar0_r = SP.array([0])
    R = covar_r.K(covar0_r)
    var_R = utils.getVariance(R)

    # initialize hidden factors
    X0_c = SP.zeros((n_t,n_c))
    X0_sigma = SP.zeros((n_t,n_sigma))

    # observed variance
    var = Y.var(1)
    var0 = SP.copy(var)
    
    # assign parts of the variance to individual effects
    ratio = SP.random.rand(3)
    ratio/= ratio.sum()
    covar0_c = ratio[0]*var.min()
    covar0_sigma = ratio[1]*var.min()

    # remaining variance is assigned to latent factors
    var-= covar0_c
    var-= covar0_sigma
    for i in range(n_t):
        signal = SP.random.rand()*var[i] 
        if n_c==1:
            X0_c[i] = SP.sign(SP.random.rand()) * SP.sqrt(signal)
        else:
            ratio = SP.random.rand(n_c)
            ratio/= ratio.sum()
            for j in range(n_c):
                X0_c[i,j] = SP.sign(SP.random.rand)*SP.sqrt(ratio[j]*signal)
        if n_sigma==1:
            X0_sigma[i] = SP.sign(SP.random.rand()) * SP.sqrt(var[i] - signal)
        else:
            ratio = SP.random.rand(n_sigma)
            ratio/= ratio.sum()
            for j in range(n_sigma):
                X0_sigma[i,j] = SP.sign(SP.random.rand)*SP.sqrt(ratio[j]*(var[i]-signal))

    # check if variance of initial values match observed variance
    assert SP.allclose(var0,(X0_c**2).sum(1).flatten() + (X0_sigma**2).sum(1).flatten() + covar0_c + covar0_sigma), 'ouch, something is wrong'

    # bring in correct format and transform as neccessary
    covar0_c = 0.5*SP.log([1./var_R,covar0_c])
    covar0_sigma = 0.5*SP.log([1,covar0_sigma])
    X0_c = SP.reshape(X0_c,(X0_c.shape[0],n_c))
    X0_sigma = SP.reshape(X0_sigma,(X0_sigma.shape[0],n_sigma))

    return X0_c,X0_sigma,covar0_c,covar0_sigma,covar0_r


