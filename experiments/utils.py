import scipy as SP

def storeHashHDF5(group,RV):
    for key,value in RV.iteritems():
        if SP.isscalar(value):
            value = SP.array([value])
        group.create_dataset(key,data=value,chunks=True,compression='gzip')

def readHDF5Hash(group):
    RV = {}
    for key in group.keys():
        RV[key] = group[key][:]
    return RV

def getVariance(K):
    """get variance scaling of K"""
    c = SP.sum((SP.eye(len(K)) - (1.0 / len(K)) * SP.ones(K.shape)) * SP.array(K))
    scalar = (len(K) - 1) / c
    return 1.0/scalar


def getVarianceKron(C,R,verbose=False):
    """ get variance scaling of kron(C,R)"""
    n_K = len(C)*len(R)
    c = SP.kron(SP.diag(C),SP.diag(R)).sum() - 1./n_K * SP.dot(R.T,SP.dot(SP.ones((R.shape[0],C.shape[0])),C)).sum()
    scalar = (n_K-1)/c
    return 1.0/scalar
