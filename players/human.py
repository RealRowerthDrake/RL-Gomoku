class HumanPlayer(object):
    def __init__(self):
        pass

    def move(self, state):
        print(state)
        pos = tuple(map(int, input().split(",")))
        return pos
