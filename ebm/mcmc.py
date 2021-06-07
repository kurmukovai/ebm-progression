import numpy as np
from tqdm import tqdm
from .likelihood import EventProbabilities

def greedy_ascent(log_p_E: np.ndarray, log_p_not_E: np.ndarray,
                  order: np.ndarray=None, n_iter: int=10000, random_state: int=None):
    """Performs greedy ascent optimization phase."""
    
    if order is None:
        order = np.arange(log_p_E.shape[1])
    if random_state is None or isinstance(random_state, int):
        random = np.random.RandomState(random_state)
    else:
        raise TypeError
        
    indices = np.arange(len(order))
    model = EventProbabilities(log_p_E, log_p_not_E)
    loglike, update_iters = [], []
    old_loglike = model.compute_total_likelihood(order)
        
    for i in tqdm(range(n_iter)):
        random.shuffle(indices)
        a, b = indices[0], indices[1]
        order[a], order[b] = order[b], order[a]
        new_loglike = model.compute_total_likelihood(order)
        if new_loglike > old_loglike:
            old_loglike = new_loglike
            loglike.append(old_loglike)
            update_iters.append(i)
        else:
            order[a], order[b] = order[b], order[a]
    return order, loglike, update_iters
