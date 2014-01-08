"""
plot_simulations.py

Author:		Barbara Rakitsch
Year:		2013
Group:		Machine Learning and Computational Biology Group
Institutes:	Max Planck Institute for Developmental Biology and Max Planck Institute for Intelligent Systems
"""

import pdb
import sys
import os
import h5py
import scipy as SP

import matplotlib.pylab as PLT
import utils
from run_experiments import get_r2

def estimate_heritability(C,R,Sigma,Omega):
    varK = utils.getVarianceKron(R,C)
    varNoise = utils.getVarianceKron(Omega,Sigma)
    return varK/(varK + varNoise)
  
def plot_sqCC(var_signal,corr,xlabel,fn_out=None,legend=False):
    """
    plot squared correlation coefficient
    """
    idx = SP.arange(len(var_signal))
    width = 0.2
    fig = PLT.figure(figsize=(3.2,3.2))
    fig.subplots_adjust(bottom=0.2,left=0.2)
    
    ax = fig.add_subplot(111)
    n_reps = corr['CV_GPbase_LIN'].shape[1]
    rects1 = ax.bar(idx,corr['CV_GPpool_LIN'].mean(1),width=width,color='y',yerr=corr['CV_GPpool_LIN'].std(1)/SP.sqrt(n_reps),ecolor='k')
    rects2=ax.bar(idx+width,corr['CV_GPbase_LIN'].mean(1),width=width,color='r',yerr=corr['CV_GPbase_LIN'].std(1)/SP.sqrt(n_reps),ecolor='k')
    rects3=ax.bar(idx+2*width,corr['CV_GPkronprod_LIN'].mean(1),width=width,color='b',yerr=corr['CV_GPkronprod_LIN'].std(1)/SP.sqrt(n_reps),ecolor='k')
    rects4=ax.bar(idx+3*width,corr['CV_GPkronsum_LIN'].mean(1),width=width,color='g',yerr=corr['CV_GPkronsum_LIN'].std(1)/SP.sqrt(n_reps),ecolor='k')
    ax.set_ylabel('Squared Correlation Coefficient')
    ax.set_xlabel(xlabel)
    ax.set_xticks(idx+width)
    ax.set_xticklabels(var_signal)
    if legend:
        ax.legend( (rects1[0],rects2[0],rects3[0],rects4[0]), ('GP-pool','GP-single','GP-kronprod','GP-kronsum'),loc=2)
    PLT.grid(True)
    PLT.ylim([0,0.8])
    if fn_out!=None:
        PLT.savefig(fn_out)
    PLT.close()

if __name__ == "__main__":
    N = 200
    D = 10
    path = os.path.join('..','out','simulations_hidden')
    methods = ['CV_GPbase_LIN','CV_GPpool_LIN','CV_GPkronprod_LIN','CV_GPkronsum_LIN']
    
    # set additional parameters
    if len(sys.argv)>1:
        n_repeats = int(sys.argv[1])
    else:
        n_repeats = 30
    f_causal_arr = SP.array([0.1,0.3,0.5,0.7,0.9,1.0])
    f_common_arr = SP.array([0.0,0.1,0.3,0.5,0.7,0.9,1.0])
    f_hidden_arr = SP.array([0.0,0.1,0.3,0.5,0.7,0.9,1])
 
    # vary common
    f_causal = 0.9
    f_hidden = 0.5
    n_signal = len(f_common_arr)
    corr = {}
    for method in methods:
        corr[method] = SP.zeros((n_signal,n_repeats))
    for i,f_common in enumerate(f_common_arr):
        for j in range(n_repeats):
            fn_out =  os.path.join(path,'results_N%d_D%d_causal%02d_common%02d_hidden%02d.hdf5'%(N,D,10*f_causal,10*f_common,10*f_hidden))
            f = h5py.File(fn_out,'r')
            for method in methods:
                RV = utils.readHDF5Hash(f['rep%d'%j][method])
                corr[method][i,j] = get_r2(RV['Y'],RV['Ypred']).mean()
            f.close()

    fn_out =  os.path.join(path,'sqCC_N%d_D%d_causal%02d_hidden%02d.pdf'%(N,D,10*f_causal,10*f_hidden))
    plot_sqCC(f_common_arr,corr,'Fraction of Common Signal $f_{common}$',fn_out=fn_out)

    # vary hidden
    f_common = 0.9
    f_causal = 0.9
    n_signal = len(f_hidden_arr)
    corr = {}
    for method in methods:
        corr[method] = SP.zeros((n_signal,n_repeats))
    for i,f_hidden in enumerate(f_hidden_arr):
        for j in range(n_repeats):
            fn_out =  os.path.join(path,'results_N%d_D%d_causal%02d_common%02d_hidden%02d.hdf5'%(N,D,10*f_causal,10*f_common,10*f_hidden))
            f = h5py.File(fn_out,'r')
            for method in methods:
                RV = utils.readHDF5Hash(f['rep%d'%j][method])
                corr[method][i,j] = get_r2(RV['Y'],RV['Ypred']).mean()
            f.close()

    fn_out =  os.path.join(path,'sqCC_N%d_D%d_causal%02d_common%02d.pdf'%(N,D,10*f_causal,10*f_common))
    plot_sqCC(f_hidden_arr,corr,'Fraction of Hidden Signal $f_{hidden}$',fn_out=fn_out)

    # vary causal
    f_common = 0.9
    f_hidden = 0.5
    n_signal = len(f_causal_arr)
    corr = {}
    for method in methods:
        corr[method] = SP.zeros((n_signal,n_repeats))
    for i,f_causal in enumerate(f_causal_arr):
        for j in range(n_repeats):
            fn_out =  os.path.join(path,'results_N%d_D%d_causal%02d_common%02d_hidden%02d.hdf5'%(N,D,10*f_causal,10*f_common,10*f_hidden))
            f = h5py.File(fn_out,'r')
            for method in methods:
                RV = utils.readHDF5Hash(f['rep%d'%j][method])
                corr[method][i,j] = get_r2(RV['Y'],RV['Ypred']).mean()
            f.close()

    fn_out =  os.path.join(path,'sqCC_N%d_D%d_common%02d_hidden%02d.pdf'%(N,D,10*f_common,10*f_hidden))
    plot_sqCC(f_causal_arr,corr,'Fraction of Signal $f_{signal}$',fn_out=fn_out,legend=True)


