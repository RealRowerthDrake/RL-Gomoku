import random


class TDPlayer(object):
    def __init__(self, table):
        self._table = table

    def move(self, state):
        if self._table[state] is None:
            return random.choice(state.valid_actions)
        else:
            return max(self._table[state].items(), key=lambda x: x[-1])[0]
