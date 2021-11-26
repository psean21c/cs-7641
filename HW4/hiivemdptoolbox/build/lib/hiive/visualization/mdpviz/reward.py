import typing

from hiive.visualization.mdpviz.outcome import Outcome


class Reward(Outcome):
    def __init__(self, value, weight=1.0):
        super().__init__(value, weight)

    def __repr__(self):
        return 'Reward(%s, %s)' % (self.outcome, self.weight)

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def get_choices(rewards: typing.Iterable['Reward']):
        return Outcome.get_choices(rewards) or {0.: 1.}
