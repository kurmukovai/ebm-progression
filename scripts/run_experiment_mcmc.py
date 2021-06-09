import sys
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

from ebm.probability import log_distributions, fit_distributions
from ebm.mcmc import greedy_ascent, mcmc

if __name__=="__main__":
    
    try:
        prior_path = sys.argv[1] # Path to file with numpy array with connectivity prior, 
        # `/data01/bgutman/parkinson_ebm/log_transition_probabilities_adni.npy`
        # TODO: add prior computation
    except:
        prior_path, prior = None, None
        
    # 1. Load data
    folder = Path('/data01/bgutman/MRI_data/PPMI/EBM_data/')
    data = pd.read_csv(folder / 'corrected_ENIGMA-PD_Mixed_Effects_train_test_split.csv', index_col=0)
    train, test = train_test_split(data, stratify=data['cohort'], test_size=0.1, random_state=777)
    X = train.drop(['SubjID', 'Dx', 'Sex', 'Age', 'cohort'], axis=1).values
    y = train['Dx'].values
    if prior_path:
        prior = np.load(prior_path)
    
    # 2. Precomute distributions P(x|E), P(x| not E)
    log_p_e, log_p_not_e = log_distributions(X, y)

    # 3. Run greedy ascent optimization phase
    order, loglike, update_iters = greedy_ascent(log_p_e, log_p_not_e, n_iter=100_000, prior=prior, random_state=2020)

    prefix = '' if not prior_path else 'prior_'
    np.save(f'../logs/{prefix}order_greedy_ascent.npy', np.array(order))
    np.save(f'../logs/{prefix}loglike_greedy_ascent.npy', np.array(loglike))
    np.save(f'../logs/{prefix}update_iters_greedy_ascent.npy', np.array(update_iters))

    # 4. Run MCMC optimization phase
    orders, loglike, update_iters, probas = mcmc(log_p_e, log_p_not_e,
                                                 order=order, n_iter=1_000_000, prior=prior, random_state=2020)

    np.save(f'../logs/{prefix}order_mcmc.npy', np.array(orders))
    np.save(f'../logs/{prefix}loglike_mcmc.npy', np.array(loglike))
    np.save(f'../logs/{prefix}update_iters_mcmc.npy', np.array(update_iters))
    np.save(f'../logs/{prefix}probas_mcmc.npy', np.array(probas))
