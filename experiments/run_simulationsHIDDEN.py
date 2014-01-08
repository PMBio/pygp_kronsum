import sys
import pdb
from data import load_simulationsHIDDEN
import run_experiments
import scipy as SP
import os
import logging as LG
import h5py


def run_simulationsHIDDEN(env,f_causal,f_common,f_hidden,methods,n_repeats):
    N = 200
    D = 10

    opts = {}
    if methods=='ALL':
        methods = ['CV_GPbase_LIN','CV_GPkronprod_LIN','CV_GPkronsum_LIN','CV_GPpool_LIN','CV_GPpool_LIN']
    
    opts['n_c'] = 1
    opts['n_sigma'] = 1
    opts['nfolds'] = 10
    opts['standardizedX'] = False
    opts['standardizedY'] = False
    opts['min_iter'] = 5
    opts['max_iter'] = 200
    opts['maf'] = None

    out_dir = os.path.join(env['out_dir'],'simulations_hidden')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fn_out = os.path.join(out_dir,'results_N%d_D%d_causal%02d_common%02d_hidden%02d.hdf5'%(N,D,10*f_causal,10*f_common,10*f_hidden))
    f = h5py.File(fn_out,'w')


    for rep_id in range(n_repeats):
        # load data
        data,RV = load_simulationsHIDDEN(env,N,D,f_causal,f_common,f_hidden,rep_id)
    
        # run experiments
        SP.random.seed(0)
        out = f.create_group('rep%d'%rep_id)
        
        try:
            run_experiments.run(methods,data,opts,out)
        except:
            print 'ouch, something is wrong...'
        
    f.close()


if __name__ == "__main__":
    print sys.argv
    f_causal = [0.1,0.3,0.5,0.7,1.0]
    f_common = [0.0,0.5,1.0]
    f_hidden = [0.0,0.3,0.5,0.7,1.0]
    
    if 'debug' in sys.argv:   
        methods = ['CV_GPpool_LIN']
        f_causal = 1.0
        f_common = 0.9
        f_hidden = 0.5
        n_repeats = 1
    else:
        f_causal = float(sys.argv[1])
        f_common = float(sys.argv[2])
        f_hidden = float(sys.argv[3])
        methods = 'ALL'
        n_repeats = 30

    env = {'sim_dir':os.path.join('..','sim_data'),'out_dir':os.path.join('..','out')}  
    run_simulationsHIDDEN(env,f_causal,f_common,f_hidden,methods,n_repeats)


