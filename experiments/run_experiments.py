import pdb
import sys
import scipy as SP
import h5py
import time
import utils
import scipy.linalg as LA
import logging as LG

import initialize

sys.path.append('../')
import core.gp.gp_base as gp_base
import core.gp.gp_kronsum as gp_kronsum
import core.gp.gp_kronprod as gp_kronprod
import core.covariance.linear as linear
import core.covariance.fixed as fixed
import core.covariance.diag as diag
import core.covariance.lowrank as lowrank
import core.likelihood.likelihood_base as lik
import core.optimize.optimize_base as opt




def get_r2(Y1,Y2):
    """
    return list of squared correlation coefficients (one per task)
    """
    if Y1.ndim==1:
        Y1 = SP.reshape(Y1,(Y1.shape[0],1))
    if Y2.ndim==1:
        Y2 = SP.reshape(Y2,(Y2.shape[0],1))

    t = Y1.shape[1]
    r2 = []
    for i in range(t):
        _r2 = SP.corrcoef(Y1[:,i],Y2[:,i])[0,1]**2
        r2.append(_r2)
    r2 = SP.array(r2)
    return r2

    
def run_optimizer(method,gp,opts,Y,X_r,Icv,cv_idx,X_o=None):
    if 'min_iter' in opts:
        min_iter  = opts['min_iter']
    else:
        min_iter = 10
    if 'max_iter' in opts:
        max_iter = opts['max_iter']
    else:
        max_iter = 100

    # initialize
    LG.info('Optimize %s'%method)
    converged = False
    lmltest_global = SP.inf
    hyperparams_global = None
    Ypred_global = None
    r2_global = -SP.inf
    # hold nfolds of the data out
    Itrain = Icv!=cv_idx
    Itest = Icv==cv_idx
    i=1
    while True:
        LG.info('Iteration: %d'%i)
        converged = False
        # stop, if maximum number of iterations is reached
        if i>max_iter:
            break

        # set data
        if X_o==None:
            gp.setData(Y=Y[Itrain],X_r=X_r[Itrain]) 
        else:
            gp.setData(Y=Y[Itrain],X_r=X_r[Itrain],X_o=X_o[Itrain])
        hyperparams,Ifilter,bounds = initialize.init(method,Y[Itrain].T,X_r[Itrain],opts)

        try:
            [hyperparams_opt,lmltrain] = opt.opt_hyper(gp,hyperparams,opts=opts,Ifilter=Ifilter,bounds=bounds)
            # gradient need not to be 0, because we have bounds on the hyperparameters...
            gradient = SP.array([LA.norm(x) for x in gp.LMLgrad(hyperparams_opt).values()]).mean()
            LG.info('LMLtrain: %.3f'%gp.LML(hyperparams_opt))
            LG.info('Gradient: %.3f'%(gradient))
            converged = True

        except AssertionError, error:
            print 'Assertion Error: %s'%error
            continue
        except:
            print "Error:", sys.exc_info()[0]
            continue

        # predict on test data
        Ypred = gp.predict(hyperparams_opt,Xstar_r=X_r[Itest])
        r2 = get_r2(Y[Itest],Ypred).mean()
        LG.info('R2: %.3f'%r2)

        # compute test likelihood
        if X_o==None:
            gp.setData(Y=Y[Itest],X_r=X_r[Itest]) 
        else:
            gp.setData(Y=Y[Itest],X_r=X_r[Itest],X_o=X_o[Itest])
        lmltest = gp.LML(hyperparams_opt)
        LG.info('LMLtest: %.3f'%lmltest)

        if converged and lmltest+1E-3<lmltest_global:
            LG.info('Update solution')
            lml_global = lmltrain
            lmltest_global = lmltest
            hyperparams_global = hyperparams_opt
            Ypred_global = Ypred
            r2_global = r2
            iter_converged = i
            
        if (hyperparams_global is not None) and i>=min_iter:
            break
        
        i+=1

    n_s = Itest.shape[0]
    # set complete data
    if X_o==None:
        gp.setData(Y=Y,X_r=X_r) 
    else:
        gp.setData(Y=Y,X_r=X_r,X_o=X_o)

    RV = {'hyperparams_opt':hyperparams_global,'lml_train':lml_global,'lml_test':lmltest_global,'lml_gradient':gradient,'iter':iter_converged,'Ypred':Ypred_global}
    return RV
        

def run(methods,data,opts,f):
    """
    run methods
    """
    # load data
    X_r = data.getX(standardized=opts['standardizedX'],maf=opts['maf'])
    Y = data.getY(standardized=opts['standardizedY']).T
    n_s,n_f = X_r.shape
    n_t = Y.shape[1]
    
    # indices for cross-validation
    r = SP.random.permutation(n_s)
    Icv=SP.floor(((SP.ones((n_s))*opts['nfolds'])*r)/n_s)

    if 'CV_GPbase_LIN' in methods:
        print 'do cross-validation with GPbase'
        t_start = time.time()
        covariance = linear.LinearCF(n_dimensions=n_f)
        likelihood = lik.GaussIsoLik()
        gp = gp_base.GP(covar=covariance,likelihood=likelihood)
        Ypred = SP.zeros(Y.shape)
        YpredCV = SP.zeros(Y.shape)
        
        lml_test = SP.zeros((n_t,opts['nfolds']))
        for i in range(n_t):
            for j in range(opts['nfolds']):
                LG.info('Train Pheno %d'%i)
                LG.info('Train Fold %d'%j)
                Itrain = Icv!=j
                Itest = Icv==j
                y = SP.reshape(Y[:,i],(n_s,1))
                cv_idx = (j+1)%opts['nfolds']
                RV = run_optimizer('GPbase_LIN',gp,opts=opts,Y=y[Itrain],X_r=X_r[Itrain],Icv=Icv[Itrain],cv_idx=cv_idx)
                lml_test[i,j] = RV['lml_test']
                Ypred[Itest,i] = gp.predict(RV['hyperparams_opt'],X_r[Itest])
                YpredCV[Icv==cv_idx,i] = RV['Ypred']
                
        lml_test = lml_test.sum(0)
        t_stop = time.time()
        r2 = (SP.corrcoef(Y.flatten(),Ypred.flatten())[0,1])**2
        print '... squared correlation coefficient: %.4f'%r2
        RV = {'Y':Y,'Ypred':Ypred,'r2':r2,'time':t_stop-t_start,'Icv':Icv,'lml_test':lml_test,'YpredCV':YpredCV}

        if f!=None:
            out = f.create_group('CV_GPbase_LIN')
            utils.storeHashHDF5(out,RV)

    if 'CV_GPpool_LIN' in methods:
        print 'do cross-validation with GPpool'
        t_start = time.time()
        covar_c = linear.LinearCF(n_dimensions=1) # vector of 1s
        covar_r = linear.LinearCF(n_dimensions=n_f)
        likelihood = lik.GaussIsoLik()
        gp = gp_kronprod.KronProdGP(covar_r=covar_r,covar_c=covar_c,likelihood=likelihood)
        gp.setData(X_c=SP.ones((Y.shape[1],1)))
        
        Ypred = SP.zeros(Y.shape)
        YpredCV = SP.zeros(Y.shape)
        lml_test = SP.zeros(opts['nfolds'])
        for j in range(opts['nfolds']):
            LG.info('Train Fold %d'%j)
            Itrain = Icv!=j
            Itest = Icv==j
            cv_idx = (j+1)%opts['nfolds']
            RV = run_optimizer('GPpool_LIN',gp,opts=opts,Y=Y[Itrain],X_r=X_r[Itrain],Icv=Icv[Itrain],cv_idx=cv_idx)
            Ypred[Itest] = gp.predict(RV['hyperparams_opt'],X_r[Itest])
            YpredCV[Icv==cv_idx] = RV['Ypred']
            lml_test[j]= RV['lml_test']
            
        t_stop = time.time()
        r2 = (SP.corrcoef(Y.flatten(),Ypred.flatten())[0,1])**2
        print '... squared correlation coefficient: %.4f'%r2
        RV = {'Y':Y,'Ypred':Ypred,'r2':r2,'time':t_stop-t_start,'Icv':Icv,'lml_test':lml_test,'YpredCV':YpredCV}

        if f!=None:
            out = f.create_group('CV_GPpool_LIN')
            utils.storeHashHDF5(out,RV)

    if 'CV_GPkronprod_LIN' in methods:
        print 'do cross-validation with GPkronprod (linear kernel)'
        t_start = time.time()
        covar_c = lowrank.LowRankCF(n_dimensions=opts['n_c'])
        covar_r = linear.LinearCF(n_dimensions=n_f)
        likelihood = lik.GaussIsoLik()
        Ypred = SP.zeros(Y.shape)
        YpredCV = SP.zeros(Y.shape)
        gp = gp_kronprod.KronProdGP(covar_r=covar_r,covar_c=covar_c,likelihood=likelihood)
    
        lml_test = SP.zeros(opts['nfolds'])
        for j in range(opts['nfolds']):
            LG.info('Train Fold %d'%j)
            Itrain = Icv!=j
            Itest = Icv==j
            cv_idx = (j+1)%opts['nfolds']
            RV = run_optimizer('GPkronprod_LIN',gp,opts=opts,Y=Y[Itrain],X_r=X_r[Itrain],Icv=Icv[Itrain],cv_idx=cv_idx)
            Ypred[Itest] = gp.predict(RV['hyperparams_opt'],X_r[Itest])
            YpredCV[Icv==cv_idx] = RV['Ypred']
            lml_test[j]= RV['lml_test']

        t_stop = time.time()
        r2 = (SP.corrcoef(Y.flatten(),Ypred.flatten())[0,1])**2
        print '... squared correlation coefficient: %.4f'%r2
        RV = {'Y':Y,'Ypred':Ypred,'r2':r2,'time':t_stop-t_start,'Icv':Icv,'lml_test':lml_test,'YpredCV':YpredCV}

        if f!=None:
            out = f.create_group('CV_GPkronprod_LIN')
            utils.storeHashHDF5(out,RV)

    if 'CV_GPkronsum_LIN' in methods:
        print 'do cross-validation with GPkronsum (linear kernel)'
        t_start = time.time()
        Ypred = SP.zeros(Y.shape)

        covar_c = lowrank.LowRankCF(n_dimensions=opts['n_c'])
        covar_r = linear.LinearCF(n_dimensions=n_f)
        covar_s = lowrank.LowRankCF(n_dimensions=opts['n_sigma'])

        X_o = SP.zeros((Y.shape[0],1))
        covar_o = diag.DiagIsoCF(n_dimensions=1)
        
        gp = gp_kronsum.KronSumGP(covar_r=covar_r,covar_c=covar_c,covar_s=covar_s,covar_o=covar_o)
        lml_test = SP.zeros(opts['nfolds'])
        Ypred = SP.zeros(Y.shape)
        YpredCV = SP.zeros(Y.shape)

        for j in range(opts['nfolds']):
            LG.info('Train Fold %d'%j)
            Itrain = Icv!=j
            Itest = Icv==j
            cv_idx = (j+1)%opts['nfolds']
            RV = run_optimizer('GPkronsum_LIN',gp,opts=opts,Y=Y[Itrain],X_r=X_r[Itrain],Icv=Icv[Itrain],cv_idx=cv_idx, X_o=X_o[Itrain])
            Ypred[Itest] = gp.predict(RV['hyperparams_opt'],X_r[Itest])
            YpredCV[Icv==cv_idx] = RV['Ypred']
            lml_test[j] = RV['lml_test']

        t_stop = time.time()
        r2 = (SP.corrcoef(Y.flatten(),Ypred.flatten())[0,1])**2
        print '... squared correlation coefficient: %.4f'%r2
        RV = {'Y':Y,'Ypred':Ypred,'r2':r2,'time':t_stop-t_start,'Icv':Icv,'lml_test':lml_test,'YpredCV':YpredCV}
        
        if f!=None:
            out = f.create_group('CV_GPkronsum_LIN')
            utils.storeHashHDF5(out,RV)

    return RV
