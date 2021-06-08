import numpy as np


def fit_distributions(X, y, normalize=False):
    """Fit distribution p(x|E), p(x|~E) as a mixture of Gaussian and Uniform, see Fonteijn 
    section `Mixture models for the data likelihood`. 
    - P(x|E) = P(x > X | E)
    - P(x|~E) = P(x < X| ~E)
    """
    # TODO: not sure about how to compute probabilities
    from scipy.stats import norm, uniform
    if normalize:
        X = X / X.max(axis=1)[:, np.newaxis]
    
    avg = X[y==0, ...].mean(axis=0)
    std = X[y==0, ...].std(axis=0)
    p_not_E = [norm(loc, s) for loc, s in zip(avg, std)]

    left_min = X.min(axis=0)
    p_E = [uniform(m1, m2) for m1, m2 in zip(left_min, avg)]
    return np.array(p_E), np.array(p_not_E)


def log_distributions(X, y, normalize=False, eps=1e-6):
    """Precomute probabilities for all features."""
    X = np.array(X).astype(np.float64)
    y = np.array(y)
    cdf_p_E, cdf_p_not_E = fit_distributions(X, y, normalize=normalize)
    
    n, m = X.shape
    log_p_E, log_p_not_E = np.zeros_like(X), np.zeros_like(X)

    for i in range(n):
        for j in range(m):
            log_p_E[i,j] = np.log(1 - cdf_p_E[j].cdf(X[i, j])+eps)
            log_p_not_E[i,j] = np.log(cdf_p_not_E[j].cdf(X[i, j])+eps)
    return log_p_E, log_p_not_E
