"""
plot_runtime.py

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

import matplotlib
import matplotlib.pylab as PLT
from matplotlib.colors import LogNorm
import utils



def plot_heatmap(t,max_val,xticks,yticks,colorbar=False,fn=None):
    fig = PLT.figure(figsize=(3,3))
    fig.add_subplot(111)
    fig.subplots_adjust(left=0.2,bottom=0.2)
    PLT.imshow(t,norm=LogNorm(vmin=1,vmax=max_val),interpolation='nearest')

    PLT.xticks(range(t.shape[1]),xticks)
    PLT.yticks(range(t.shape[0]),yticks)
    PLT.xlabel('Number of Tasks')
    PLT.ylabel('Number of Samples')

    if colorbar:
        PLT.colorbar()

    if fn!=None:
        PLT.savefig(fn)
    PLT.close()

def get_mean(t,max_val=1E4):
    """
    returns mean:

    if an entry is 0, its calculations are not finished yet:
    - if all entries are 0, set to max_val
    - if not all entries are 0, ignore them
    """
    if t.sum()==0:
        return max_val
    
    t[t>max_val] = max_val

    return t[t!=0].mean()
    
if __name__ == "__main__":
    # output folder
    out_dir = os.path.join('..','out','simulations_runtime')
    
    # plot 1: runtime vs. number of samples
    N_arr = SP.array([16,32,64,128,256])
    D_arr = SP.array([16,32,64,128,256])
    
    t_fast = SP.zeros((len(N_arr),len(D_arr)))
    t_slow = SP.zeros((len(N_arr),len(D_arr)))
    max_val = 1E4
    
    for i,N in enumerate(N_arr):
        for j,D in enumerate(D_arr):
            fn = os.path.join(out_dir,'results_runtime_signal500_N%d_D%d.hdf5'%(N,D))
            f = h5py.File(fn,'r')
            t_fast[i,j] = get_mean(f['t_fast'][:],max_val=1E4)
            t_slow[i,j] = get_mean(f['t_slow'][:],max_val=1E4)
            f.close()

    fn = os.path.join('..','out','simulations_runtime','runtime_speedups.pdf')
    plot_heatmap(t_fast,max_val,D_arr,N_arr,fn=fn,colorbar=True)
    fn = os.path.join('..','out','simulations_runtime','runtime_naive.pdf')
    plot_heatmap(t_slow,max_val,D_arr,N_arr,colorbar=True,fn=fn)
