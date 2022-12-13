from abc import ABC, abstractmethod

class BayesianEstimator(ABC):
    """Abstract Base class for a Univariate Bayesian Estimator."""
  
    @abstractmethod
    def _compute_posterior(self):
        """Return the Bayesian posterior for the random variable being modeled."""
        pass

    @abstractmethod
    def compute_MAP_estimate(self):
        """Return the Bayesian 'Maximum a Priori' point estimate for the random variable being modeled."""
        pass

    @abstractmethod
    def compute_credible_interval(self):
        pass

    @abstractmethod
    def compute_expected_value(self):
        """Return the posterior distribution's expected value for the random variable being modeled."""
        pass

    @abstractmethod
    def plot_posterior(self):
        """Plot the posterior distribution for the random variable being modeled."""
        pass