import sys
import pdb
from data import load_simulations
import scipy as SP
import os
import logging as LG
import h5py
import time
import sys
import subprocess
import signal

sys.path.append('../')
import core.gp.gp_kronsum as gp_kronsum
import core.gp.gp_kronsum_naive as gp_kronsum_naive
import core.covariance.linear as linear
import core.covariance.fixed as fixed
import core.covariance.lowrank as lowrank
import core.covariance.diag as diag
import core.optimize.optimize_base as opt

import initialize



def handler(signum,frame):
    raise Exception("time limited exceeded")
        
def measure_runtime(env,N,D,n_reps=10,time_out=10000):
    opts = {'messages':False}
    out_dir = os.path.join(env['out_dir'],'simulations_runtime')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    t_fast = SP.zeros(n_reps)
    t_slow = SP.zeros(n_reps)
    lml_fast = SP.zeros(n_reps)
    lml_slow = SP.zeros(n_reps)
     
    for i in range(n_reps):
        # load data
        var_signal = 0.5
        data,RV = load_simulations(env,var_signal,N,D,i)

        # initialize
        covar_c = lowrank.LowRankCF(n_dimensions=RV['n_c'])
        covar_r = linear.LinearCF(n_dimensions=RV['n_r'])
        covar_s = lowrank.LowRankCF(n_dimensions=RV['n_sigma'])
        covar_o = fixed.FixedCF(n_dimensions=RV['n_r'])
        X = data.getX(standardized=False)
        Y = data.getY(standardized=False).T
        hyperparams,Ifilter,bounds = initialize.init('GPkronsum_LIN',Y.T,X,RV)
        covar_r.X = X
        covar_o.X = X
        covar_o._K = SP.eye(RV['N'])
        covar_s.X = hyperparams['X_s']
        covar_c.X = hyperparams['X_c']
        kgp_fast = gp_kronsum.KronSumGP(covar_r=covar_r,covar_c=covar_c,covar_s=covar_s,covar_o=covar_o)
        kgp_fast.setData(Y=Y)
        
        # measure time
        signal.signal(signal.SIGALRM,handler)
        signal.alarm(time_out)
        try:
             t_start = time.clock()
             hyperparams_opt,lmltrain = opt.opt_hyper(kgp_fast,hyperparams,Ifilter=Ifilter,bounds=bounds,opts=opts)
             t_stop = time.clock()
             signal.alarm(0)
             t_fast[i] = t_stop - t_start
             lml_fast[i] = lmltrain
        except Exception, e:
            print e
            t_slow += time_out
            break

    # save
    fn_out =  os.path.join(out_dir,'results_runtime_signal%03d_N%d_D%d.hdf5'%(var_signal*1E3,N,D))
    f = h5py.File(fn_out,'w')
    f['t_fast'] = t_fast
    f['t_slow'] = t_slow
    f['lml_fast'] = lml_fast
    f['lml_slow'] = lml_slow
    f.close()

    for i in range(n_reps):
        # initialize
        data,RV = load_simulations(env,var_signal,N,D,i)
        covar_c = lowrank.LowRankCF(n_dimensions=RV['n_c'])
        covar_r = linear.LinearCF(n_dimensions=RV['n_r'])
        covar_s = lowrank.LowRankCF(n_dimensions=RV['n_sigma'])
        covar_o = fixed.FixedCF(n_dimensions=RV['n_r'])
        X = data.getX(standardized=False)
        Y = data.getY(standardized=False).T
        hyperparams,Ifilter,bounds = initialize.init('GPkronsum_LIN',Y.T,X,RV)
        covar_r.X = X
        covar_o.X = X
        covar_o._K = SP.eye(RV['N'])
        covar_s.X = hyperparams['X_s']
        covar_c.X = hyperparams['X_c']
        kgp_slow = gp_kronsum_naive.KronSumGP(covar_r=covar_r,covar_c=covar_c,covar_s=covar_s,covar_o=covar_o)
        kgp_slow.setData(Y=Y)

        # measure time
        signal.signal(signal.SIGALRM,handler)
        signal.alarm(time_out)
        try:
             t_start = time.clock()
             hyperparams_opt,lmltrain = opt.opt_hyper(kgp_slow,hyperparams,Ifilter=Ifilter,bounds=bounds,opts=opts)
             t_stop = time.clock()
             signal.alarm(0)
             t_slow[i] = t_stop - t_start
             lml_slow[i] = lmltrain
        except Exception, e:
            print e
            t_slow += time_out
            break 

    # save
    fn_out =  os.path.join(out_dir,'results_runtime_signal%03d_N%d_D%d.hdf5'%(var_signal*1E3,N,D))
    f = h5py.File(fn_out,'w')
    f['t_fast'] = t_fast
    f['t_slow'] = t_slow
    f['lml_fast'] = lml_fast
    f['lml_slow'] = lml_slow
    f.close()


if __name__ == "__main__":
    print sys.argv
    if 'debug' in sys.argv:
        N = 16
        D = 16
        n_reps = 10
        
    else:
        N = int(sys.argv[1])
        D = int(sys.argv[2])
        n_reps = 10

    env = {'sim_dir':'sim_data','out_dir':'out'}
    measure_runtime(env,N,D)

