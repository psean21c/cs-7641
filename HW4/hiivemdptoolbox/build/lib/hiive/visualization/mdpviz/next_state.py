import typing

from hiive.visualization.mdpviz.outcome import Outcome


class NextState(Outcome):
    def __init__(self, state, weight=1.0):
        super().__init__(state, weight)

    def __repr__(self):
        return 'NextState(%s, %s)' % (self.outcome, self.weight)

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def get_choices(next_states: typing.Iterable['NextState']):
        return Outcome.get_choices(next_states)
