import scipy.optimize as OPT
import pdb
import scipy as SP
import logging as LG
import numpy as np
import time
import sys

def param_dict_to_list(dict,skeys=None):
    """convert from param dictionary to list"""
    #sort keys
    RV = SP.concatenate([dict[key].flatten() for key in skeys])
    return RV
    pass

def param_list_to_dict(list,param_struct,skeys):
    """convert from param dictionary to list
    param_struct: structure of parameter array
    """
    RV = []
    i0= 0
    for key in skeys:
        val = param_struct[key]
        shape = SP.array(val) 
        np = shape.prod()
        i1 = i0+np
        params = list[i0:i1].reshape(shape)
        RV.append((key,params))
        i0 = i1
    return dict(RV)


            
def opt_hyper(gpr,hyperparams,Ifilter=None,bounds=None,opts={},*args,**kw_args):
    """
    optimize hyperparams
    
    Input:
    gpr: GP regression class
    hyperparams: dictionary filled with starting hyperparameters
    opts: options for optimizer
    """
    if 'max_iter_opt' in opts:
        max_iter = opts['max_iter_opt']
    else:
        max_iter = 5000
    if 'pgtol' in opts:
        pgtol = opts['pgtol']
    else:
        pgtol = 1e-10
    if 'messages'in opts:
        messages = opts['messages']
    else:
        messages = True
    if 'approx_grad' in opts:
        approx_grad = opts['approx_grad']
    else:
        approx_grad = 0
        
    def f(x):
        x_ = X0
        x_[Ifilter_x] = x
        rv = gpr.LML(param_list_to_dict(x_,param_struct,skeys),*args,**kw_args)
        if SP.isnan(rv):
            return 1E6
        return rv

    def df(x):
        x_ = X0
        x_[Ifilter_x] = x
        rv = gpr.LMLgrad(param_list_to_dict(x_,param_struct,skeys),*args,**kw_args)
        rv = param_dict_to_list(rv,skeys)
        if (~SP.isfinite(rv)).any():
            idx = (~SP.isfinite(rv))
            rv[idx] = 1E6
        return rv[Ifilter_x]

    skeys = SP.sort(hyperparams.keys())
    param_struct = dict([(name,hyperparams[name].shape) for name in skeys])

    # mask params that should not be optimized
    X0 = param_dict_to_list(hyperparams,skeys)
    if Ifilter is not None:
        Ifilter_x = SP.array(param_dict_to_list(Ifilter,skeys),dtype=bool)
    else:
        Ifilter_x = SP.ones(len(X0),dtype='bool')

    # add bounds if necessary
    if bounds is not None:
        _b = []
        for key in skeys:
            if key in bounds.keys():
                _b.extend(bounds[key])
            else:
                _b.extend([[-SP.inf,+SP.inf]]*hyperparams[key].size)
        bounds = SP.array(_b)
        bounds = bounds[Ifilter_x]

    LG.info('Starting optimization ...')
    t = time.time()

    x = X0.copy()[Ifilter_x]

    if approx_grad:
        RVopt = OPT.fmin_tnc(f,x,messages=messages,maxfun=int(max_iter),pgtol=pgtol,bounds=bounds,approx_grad=1)
    else:
        RVopt = OPT.fmin_tnc(f,x,fprime=df,messages=messages,maxfun=int(max_iter),pgtol=pgtol,bounds=bounds)

    LG.info('%s'%OPT.tnc.RCSTRINGS[RVopt[2]])
    LG.info('Optimization is converged at iteration %d'%RVopt[1])
    LG.info('Total time: %.2fs'%(time.time()-t))
    xopt = RVopt[0]
    Xopt = X0.copy()
    Xopt[Ifilter_x] = xopt
    hyperparams_opt = param_list_to_dict(Xopt,param_struct,skeys)
    lml_opt = gpr.LML(hyperparams_opt,**kw_args)

    return [hyperparams_opt,lml_opt]
