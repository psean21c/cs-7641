class Transition:
    def __init__(self, action, state, index):
        self.action = action
        self.state = state
        self.index = index

    def __repr__(self):
        return f'{self.state} : {self.action} {self.index}'

    def __str__(self):
        return f'{self.state}/{self.action}'.replace(' ', '_')

    def __hash__(self):
        return self.__str__().__hash__()
