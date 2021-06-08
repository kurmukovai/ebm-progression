import numpy as np


class EventProbabilities:
    def __init__(self, log_p_E: np.ndarray, log_p_not_E: np.ndarray):
        self.log_p_E = log_p_E
        self.log_p_not_E = log_p_not_E
        self.n_stages = log_p_E.shape[1]
        
        self._likelihood = self._init_likelihood()
        self.subjects_likelihood = None
    
    def _init_likelihood(self,):
        """Computes likelihood for stage k=0, see Fonteijn, (1), for every subject."""
        return np.sum(self.log_p_not_E, axis=1)
    
    def _subject_likelihood(self, event_order: np.ndarray = None):
        """Computes formula (2) form Fonteijn"""
        likelihood = self._likelihood.copy()
        for k in range(self.n_stages-1):
            likelihood += self.log_p_E[:, event_order[k]] \
                        - self.log_p_not_E[:, event_order[k]] 
            self.subjects_likelihood += np.exp(likelihood)
        # assuming flat prior p(k)
        return self.subjects_likelihood
    
    def _compute_connectivity_prior(self, aver_connectome: np.ndarray = None):
        pass
    
    def compute_total_likelihood(self, event_order: np.ndarray = None):
        """Computes log(P(X|S)), see Fonteijn, (3)"""
        # TODO? could further improve for MCMC procedure, since we only
        # swap two positions in the order
        self.subjects_likelihood = np.exp(self._likelihood)
        if event_order is None:
            event_order = np.arange(self.n_stages)
            
        return np.sum(np.log(self._subject_likelihood(event_order)))
    


