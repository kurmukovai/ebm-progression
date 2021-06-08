import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from ebm.probability import log_distributions, fit_distributions
from ebm.mcmc import greedy_ascent, mcmc

folder = Path('/data01/bgutman/MRI_data/PPMI/EBM_data/')
data = pd.read_csv(folder / 'corrected_ENIGMA-PD_Mixed_Effects_train_test_split.csv', index_col=0)
train, test = train_test_split(data, stratify=data['cohort'], test_size=0.1, random_state=777)

log_p_e, log_p_not_e = log_distributions(X, y)

order, loglike, update_iters = greedy_ascent(log_p_e, log_p_not_e, n_iter=100_000)
orders, loglike, update_iters, probas = mcmc(log_p_e, log_p_not_e, order=order, n_iter=100_000)