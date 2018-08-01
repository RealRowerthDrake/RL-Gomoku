import random


class RandomPlayer(object):
    def __init__(self):
        pass

    def move(self, state):
        return random.choice(state.valid_actions)
