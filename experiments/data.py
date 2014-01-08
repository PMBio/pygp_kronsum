import pdb
import scipy
import scipy.io as IO
import sys
import os
import h5py
import scipy as SP
import logging as LG
import numpy as NP
import utils
import scipy.linalg as LA
import scipy.stats.morestats as MS

sys.path.append('..')
from core.gp.gp_kronprod import ravel,unravel

class MatrixData():
    def __init__(self,X=None,Y=None,ids=None):
        self.n_s,self.n_f = X.shape
        self.n_t = Y.shape[1]
        assert self.n_s==Y.shape[0], 'dimensions do not match'
        assert self.n_t==ids.shape[0], 'dimensions do not match'

        self.X = X
        self.Y = Y
        self.ids = ids
        self.idx_samples = SP.ones(self.n_s,dtype=bool)
        self.idx_traits = SP.ones(self.n_t,dtype=bool)
        
    def getIDs(self):
        return self.ids[self.idx_traits]

    def setX(self,X):
        self.X = X
        self.n_f = X.shape[1]
        assert self.n_s==X.shape[0], 'ouch, something is wrong'

    def doBoxCoxTransformation(self):
        for i in range(self.n_t):
            y = self.Y[:,i]
            idx = SP.isfinite(y)
            self.Y[idx,i] = MS.boxcox(y[idx])[0]

    def selectTraits(self,phenoMAF=None,corrMin=None,nUnique=False):
        """
        use only a subset of traits

        filter out all individuals that have missing values for the selected ones
        """
        self.idx_samples = SP.ones(self.n_s,dtype=bool)
        
        # filter out nan samples
        self.idx_samples[SP.isnan(self.Y[:,self.idx_traits]).any(1)] = False
        
        # filter out phenotypes that are not diverse enough
        if phenoMAF!=None:
            expr_mean = self.Y[self.idx_samples].mean(0)
            expr_std = self.Y[self.idx_samples].std(0)
            z_scores = SP.absolute(self.Y[self.idx_samples]-expr_mean)/SP.sqrt(expr_std)
            self.idx_traits[(z_scores>1.5).mean(0) < phenoMAF] = False

        # use only correlated phenotypes
        if corrMin!=None and self.Y.shape[1]>1:
            corr = SP.corrcoef(self.Y[self.idx_samples].T)
            corr-= SP.eye(corr.shape[0])
            self.idx_traits[SP.absolute(corr).max(0)<0.3] = False

        # filter out binary phenotypes
        if nUnique and self.Y.shape[1]>1:
            for i in range(self.Y.shape[1]):
                if len(SP.unique(self.Y[self.idx_samples][:,i]))<=nUnique:
                    self.idx_traits[i] = False

        LG.debug('number of traits(before filtering): %d'%self.n_t)
        LG.debug('number of traits(after filtering): %d'%self.idx_traits.sum())
        LG.debug('number of samples(before filtering): %d'%self.n_s)
        LG.debug('number of samples(after filtering): %d'%self.idx_samples.sum())

    def setTraits(self,idx_traits):
        self.idx_traits = idx_traits

    def getTraits(self):
        return self.idx_traits

    def setSamples(self,idx_samples):
        self.idx_samples = idx_samples

    def getSamples(self):
        return self.idx_samples


    def getX(self,standardized=True,maf=None):
        """
        return SNPs, if neccessary standardize them
        """
        X = SP.copy(self.X)

        # test for missing values
        isnan = SP.isnan(X)
        for i in isnan.sum(0).nonzero()[0]:
            # set to mean 
            X[isnan[:,i],i] = X[~isnan[:,i],i].mean()
                
        if maf!=None:
            LG.debug('filter SNPs')
            LG.debug('... number of SNPs(before filtering): %d'%X.shape[1])
            idx_snps = SP.logical_and(X[self.idx_samples].mean(0)>0.1,X[self.idx_samples].mean(0)<0.9)
            LG.debug('... number of SNPs(after filtering) : %d'%idx_snps.sum())
        else:
            idx_snps = SP.ones(self.n_f,dtype=bool)
        
        if standardized:
            LG.debug('standardize SNPs')
            X = X[self.idx_samples][:,idx_snps]
            X-= X.mean(0)
            X /= X.std(0,dtype=NP.float32)
            X /= SP.sqrt(X.shape[1])
            return X
      
        return X[self.idx_samples][:,idx_snps]

    def getY(self,standardized=True):
        """
        return traits, if neccessary standardize them
        """
        assert SP.isnan(self.Y[self.idx_samples][:,self.idx_traits]).any()==False, 'Phenotypes are not filtered'
        if standardized:
            LG.debug('standardize Phenotypes')
            Y = SP.copy(self.Y[self.idx_samples][:,self.idx_traits])
            Y-= Y.mean(0)
            Y /= Y.std(0,dtype=NP.float32)
            return Y.T
        return self.Y[self.idx_samples][:,self.idx_traits].T


def load_simulations(env,var_signal,N,D,rep_id):
    """
    load simulation data
    """
    fn = os.path.join(env['sim_dir'],'sim_runtime_signal%03d_N%d_D%d.hdf5'%(var_signal*1E3,N,D))
    f = h5py.File(fn,'r')
    RV =  utils.readHDF5Hash(f['rep%d'%rep_id])
    ids = SP.array([str(x) for x in range(RV['Y'].shape[1])])
    data = MatrixData(RV['X_r'],RV['Y'],ids)
    return data,RV

def load_simulationsHIDDEN(env,N,D,f_causal,f_common,f_hidden,rep_id):
     fn = os.path.join(env['sim_dir'],'sim_N%d_D%d_causal%02d_common%02d_hidden%02d.hdf5'%(N,D,10*f_causal,10*f_common,10*f_hidden))
     f = h5py.File(fn,'r')
     RV =  utils.readHDF5Hash(f['rep%d'%rep_id])
     Y = unravel(RV['Y'],RV['N'],RV['D'])
     ids = SP.array([str(x) for x in range(Y.shape[1])])
     data = MatrixData(RV['X_r'],Y,ids)
     return data,RV

