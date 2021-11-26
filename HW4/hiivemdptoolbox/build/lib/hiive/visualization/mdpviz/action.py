class Action:
    def __init__(self, name, index, extra_data=None):
        self.name = name
        self.index = index
        if extra_data is not None:
            self.data = extra_data

    def __repr__(self):
        return 'Action(%s, %s)' % (self.name, self.index)

    def __str__(self):  # A{self.index}_
        return f'{self.name}'.replace(' ', '_')

    def __hash__(self):
        return self.__str__().__hash__()
