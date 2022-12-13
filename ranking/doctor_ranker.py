from typing import Dict
from models import BayesianEstimator
from util import metrics, orderings

class RankCriteria:

    def __init__(self, metric: str, ordering: str):
        """Create a RankCriteria object.
        
        Raises:
            ValueError if invalid metric or ordering is specified.
        """
        if metric not in {metrics.MAP, metrics.EXPECTED_VALUE, metrics.CREDIBLE_INTERVAL}:
            raise ValueError("Invalid metric specified.")
        if ordering not in {orderings.ASCENDING, orderings.DESCENDING}:
            raise ValueError("Invalid ordering specified.")

        self.metric = metric
        self.ordering = ordering

class DoctorRanker:
    """Produces rankings of doctors based on specified criteria."""

    def __init__(self, estimators: Dict[str, BayesianEstimator], criteria: RankCriteria):
        """Create a DoctorRanker object."""
        self.criteria = criteria
        self.estimators = estimators

    def get_rankings(self) -> Dict[int, str]:
        """Return prioritized rankings of doctors based on specified criteria.
        
        Rankings are returned as a dictionary mapping numeric rank to an unordered set of tuples (doctor_id, metric).

        For point estimates, rank is determined by simply looking in sorted order at the metric values. For intervals,
        however, a more nuanced approach is taken:

            As currently implemented, when asked to rank doctors based on Bayesian credible intervals, the DoctorRanker
            will assign rank 1 to the interval with the lowest/highest left/right endpoint (depending on if the criteria
            is ascending or descending). Next, any subsequent doctors whose intervals overlap the rank 1 interval are also
            assigned rank 1. For the first disjoint interval (whose boundaries do not intersect the rank 1 interval), a rank
            of 2 is assigned, and the process repeats itself.  
        """
        doctors, posteriors = self.estimators.keys(), self.estimators.values()
        reverse = (self.criteria.ordering == orderings.DESCENDING)

        if self.criteria.metric == metrics.MAP:

            maps = [posterior.compute_MAP_estimate() for posterior in posteriors]
            rankings = {
                i+1: {(doctor, map)}
                for i, (map, doctor) in enumerate(sorted(zip(maps, doctors), reverse=reverse))
            }

        elif self.criteria.metric == metrics.EXPECTED_VALUE:

            expected_vals = [posterior.compute_expected_value() for posterior in posteriors]
            rankings = {
                i+1: {(doctor, expected_val)}
                for i, (expected_val, doctor) in enumerate(sorted(zip(expected_vals, doctors), reverse=reverse))
            }

        elif self.criteria.metric == metrics.CREDIBLE_INTERVAL:

            credible_intervals = [posterior.compute_credible_interval() for posterior in posteriors]
            credible_intervals, doctors = zip(*sorted(zip(credible_intervals, doctors), key=lambda x: x[0], reverse=reverse))

            rankings = {1: set()}

            rankings[1].add((doctors[0], credible_intervals[0]))
       
            current_rank = 1
            benchmark_interval = credible_intervals[0]
            for i, doctor in zip(range(1, len(credible_intervals)), doctors[1:]):

                current_interval = credible_intervals[i]

                interval_disjoint_from_benchmark = (
                    current_interval[0] > benchmark_interval[1] if not reverse
                    else current_interval[1] < benchmark_interval[0]
                )

                # Rank only goes down if current credible interval is outside the range of the "benchmark".
                if interval_disjoint_from_benchmark:
                    current_rank += 1
                    benchmark_interval = current_interval
    
                if current_rank in rankings:
                    rankings[current_rank].add((doctor, current_interval))
                else:
                    rankings[current_rank] = set()
                    rankings[current_rank].add((doctor, current_interval))
        
        return rankings
