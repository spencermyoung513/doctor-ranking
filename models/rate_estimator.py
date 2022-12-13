from typing import Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from models import BayesianEstimator

class RateEstimator(BayesianEstimator):
    """Models the distribution of an unknown rate in a binomial likelihood process.

    More specifically, given draws from some binomial distribution with unknown rate p, this class
    performs Bayesian estimation to predict the posterior distribution of p.
    
    Attributes:
        prior (np.ndarray): Discretized prior pdf of the unknown rate. An array of values is passed in to
                            finitely approximate the prior at a desired granularity.
        posterior (np.ndarray): Discretized posterior pdf of the unknown rate. Represented as an array of
                                values covering the same support as the prior pdf.
        support (np.ndarray): Discrete values of the unknown rate for which the posterior density is computed.
        num_successes (int): Number of observed "successes" (as defined by whatever binomial distribution is
                             being modeled by this estimator).
        num_observations (int): Number of total observations out of which `num_successes` occurred.
    """
    def __init__(self, prior: np.ndarray, support: np.ndarray, num_successes: int, num_trials: int):
        """Create a RateEstimator with given prior and observations.
        
        Upon creation, provided with data `num_successes` and `num_trials`, the estimator computes a posterior
        distribution for the unknown rate over the specified support domain.

        Args:
            prior (np.ndarray): Discretized prior pdf of the unknown rate. An array of values is passed in to
                                finitely approximate the prior at a desired granularity.
            support (np.ndarray): Discrete values of the unknown rate for which the posterior density is computed.
            num_successes (int): Number of observed "successes" (as defined by whatever binomial distribution is
                             being modeled by this estimator).
            num_observations (int): Number of total observations out of which `num_successes` occurred.
        """
        # super(prior, likelihood=binomial, support=support, data=num_referrals, num_benign).__init__()
        self.prior = prior
        self.num_successes = num_successes
        self.num_observations = num_trials
        self.support = support
        self.h = support[1] - support[0]
        self.posterior = self._compute_posterior()
    
    def _compute_posterior(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the estimated posterior distribution of the unknown rate.

        This function is called internally whenever a RateEstimator is created. At initialization,
        the prior is specified and data for num_successes and num_observations is passed in.
    
        Returns:
            posterior (np.ndarray): Posterior densities for all possible rates in `self.support`. This is
                                    a discrete approximation of the posterior pdf for the unknown rate.
        """
        likelihood = lambda p: stats.binom(self.num_observations, p).pmf(self.num_successes)

        likelihoods = np.array([likelihood(p) for p in self.support])

        numerator = self.prior * likelihoods    # Form of the posterior.
        posterior = numerator / (numerator*self.h).sum() # Makes it so the posterior is a pdf (integrates to 1).
        
        return posterior

    def compute_MAP_estimate(self) -> float:
        """Return the Bayesian 'Maximum a Priori' point estimate for the true value of the unknown rate."""
        return self.support[self.posterior.argmax()]

    def compute_credible_interval(self, alpha=0.05) -> Tuple[float, float]:
        """Return the centered Bayesian (1-alpha)-credible interval for the unknown rate.
        
        Args:
            alpha (float): Determines the width of the credible interval (the amount of posterior
                           probability in the interval is 1 - alpha)

        Returns:
            credible_interval (tuple): Tuple containing left and right endpoints of desired credible interval.
        """
        # Search left tail until (alpha / 2) probability is accounted for to the left.
        probability_sum, i = 0, 0
        while probability_sum < (alpha) / 2:
            probability_sum += self.h * self.posterior[i]
            i += 1

        # Search right tail until (alpha / 2) probability is accounted for to the right.
        probability_sum, j = 0, len(self.posterior) - 1
        while probability_sum < (alpha) / 2:
            probability_sum += self.h * self.posterior[j]
            j -= 1

        return (self.support[i], self.support[j])

    def compute_expected_value(self) -> float:
        """Return the posterior distribution's expected value for the unknown rate."""
        return (self.support * self.posterior).sum() / self.posterior.sum()

    def plot_posterior(self, x_lim: Tuple[Union[int, float], Union[int, float]] = (0, 1), show=False, **plt_kwargs):
        """Plot the posterior distribution for the unknown rate.
        
        Args:
            x_lim (tuple): Tuple specifying left/right endpoints for posterior plot.
            show (bool): Specifies whether to call plt.show() on generated plot. Defaults to False.
            **plt_kwargs: Allows for passing in matplotlib-specific arguments to the plt.plot() (such as label, color, etc.)
        """
        plt.plot(self.support, self.posterior, **plt_kwargs)
        plt.xlim(*x_lim)

        if show:
            plt.show()
