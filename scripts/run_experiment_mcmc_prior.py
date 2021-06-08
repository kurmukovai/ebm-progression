import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from ebm.probability import log_distributions, fit_distributions
from ebm.mcmc import greedy_ascent, mcmc

if __name__=="__main__":
    folder = Path('/data01/bgutman/MRI_data/PPMI/EBM_data/')
    data = pd.read_csv(folder / 'corrected_ENIGMA-PD_Mixed_Effects_train_test_split.csv', index_col=0)
    train, test = train_test_split(data, stratify=data['cohort'], test_size=0.1, random_state=777)
    X = train.drop(['SubjID', 'Dx', 'Sex', 'Age', 'cohort'], axis=1).values
    y = train['Dx'].values

    log_p_e, log_p_not_e = log_distributions(X, y)

    order, loglike, update_iters = greedy_ascent(log_p_e, log_p_not_e, n_iter=100_000)

    np.save('../logs/order_greedy_ascent_prior.npy', np.array(order))
    np.save('../logs/loglike_greedy_ascent_prior.npy', np.array(loglike))
    np.save('../logs/update_iters_greedy_ascent_prior.npy', np.array(update_iters))

    orders, loglike, update_iters, probas = mcmc(log_p_e, log_p_not_e, order=order, n_iter=1_000_000)

    np.save('../logs/order_mcmc_prior.npy', np.array(orders))
    np.save('../logs/loglike_mcmc_prior.npy', np.array(loglike))
    np.save('../logs/update_iters_mcmc_prior.npy', np.array(update_iters))
    np.save('../logs/probas_mcmc_prior.npy', np.array(probas))