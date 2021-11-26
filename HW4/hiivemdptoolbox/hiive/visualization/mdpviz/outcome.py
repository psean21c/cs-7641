import typing
from collections import defaultdict


class Outcome:
    """An outcome can be either a reward or a next state.

    For a given (state, action) transition all potential outcomes
    are weighted according to their `weight` and normalized.
    """

    def __init__(self, outcome, weight):
        self.weight = weight
        self.outcome = outcome

    @staticmethod
    def get_choices(outcomes: typing.Iterable['Outcome']):
        """Normalize outcomes and deduplicate into a Dict[outcome, probability]."""

        # Deduplicate elements
        deduped_outcomes = defaultdict(float)
        total_weight = 0.
        for outcome in outcomes:
            deduped_outcomes[outcome.outcome] += outcome.weight
            total_weight += outcome.weight

        return {outcome: weight / total_weight for outcome, weight in deduped_outcomes.items()}
