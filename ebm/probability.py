import numpy as np


def fit_distributions(X, y):
    '''Fit distribution p(x|E), p(x|~E).'''
    from scipy.stats import norm, uniform
    y = np.array(y)
    X = np.array(X).astype(np.float64)
    X = X / X.max(axis=1)[:, np.newaxis]
    
    avg = X[y==0, ...].mean(axis=0)
    std = X[y==0, ...].std(axis=0)
    p_not_E = [norm(loc, s) for loc, s in zip(avg, std)]
    
    eps=1e-3
    _min = X[y==1, ...].min(axis=0) - eps
    p_E = [uniform(m1, m2) for m1, m2 in zip(_min, avg)]
    
    return np.array(p_E), np.array(p_not_E)

def log_distributions(X, y):
    p_E, p_not_E = fit_distributions(X, y)
    
    n, m = X.shape
    p_E_precomputed = np.zeros_like(X)
    
    X = np.array(X).astype(np.float64)
    X = X / X.max(axis=0)[np.newaxis, :]

    for i in range(n):
        for j in range(m):
            p_E_precomputed[i,j] = np.log(p_E[j].cdf(X[i, j]))

    p_not_E_precomputed = np.zeros_like(X)

    for i in range(n):
        for j in range(m):
            p_not_E_precomputed[i,j] = np.log(p_not_E[j].cdf(X[i, j]))
    
    return p_E_precomputed, p_not_E_precomputed
