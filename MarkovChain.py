import numpy as np

class MarkovChain:

    def __init__(self, log_likelihood, initial_state, g_sigma=0.5):
        self._chain= [initial_state]
        self._initial_state = initial_state
        self._current_state = initial_state
        self._current_step = 0
        self._log_likelihood = log_likelihood
        self._generating_sigma = g_sigma
    
    def get_candidate_state(self):
        return np.random.normal(loc=self._current_state, scale=self._generating_sigma)

    def acceptance_prob(self, new):
        diff = self._log_likelihood(new) - self._log_likelihood(self._current_state)
        prob = np.min([1.0, np.exp(diff)])
        return prob

    def advance_step(self):
        candidate = self.get_candidate_state()
        probability = self.acceptance_prob(candidate)
        test = np.random.uniform()
        if(test<=probability):
            self._current_state = candidate
            self._chain.append(self._current_state)
        else:
            self._chain.append(self._current_state)
        
        self._current_step += 1

    def get_trace(self):
        return np.arange(self._current_step+1), np.array(self._chain).copy()

    def get_chain(self):
        return np.array(self._chain).copy()

    def run_chain(self, n):
        while(n!=0):
            self.advance_step()
            n-=1