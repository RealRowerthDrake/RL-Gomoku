import random
from envs import GomokuEnv

class RandomPlayer(object):
    def __init__(self):
        pass

    def move(self, state):
        return random.choice(state.valid_actions)

class HumanPlayer(object):
    def __init__(self):
        pass

    def move(self, state):
        print(state)
        pos = tuple(map(int, input().split(",")))
        return pos

class TDPlayer(object):
    def __init__(self, table):
        self._table = table

    def move(self, state):
        if self._table[state] is None:
            return random.choice(state.valid_actions)
        else:
            return max(self._table[state].items(), key=lambda x:x[-1])[0]

def evaluate(env, players, num_games):
    def play_once():
        state = env.reset()
        while(True):
            action = players[env.cur_player].move(state)
            state, reward, done, _ = env.step(action)
            if done:
                return reward if env.cur_player == 1 else -reward
    stats = [0] * 3
    for _ in range(num_games):
        stats[play_once()] += 1
    return stats[1], stats[-1], stats[0]


if __name__ == '__main__':
    env = GomokuEnv(3, 3)

    from models.td import train
    import models.sarsa
    Q1 = train(env, models.sarsa.build_fn, 1000)

    import models.q_learning
    Q2 = train(env, models.q_learning.build_fn, 1000)

    print("Sarsa vs Random")
    result = evaluate(env, (TDPlayer(Q1), RandomPlayer()), 100)
    print("-- 1P: ", "#Win={}, #Lose={}, #Draw={}".format(result[0], result[1], result[2]))
    result = evaluate(env, (RandomPlayer(), TDPlayer(Q1)), 100)
    print("-- 2P: ", "#Win={}, #Lose={}, #Draw={}".format(result[1], result[0], result[2]))
    print()

    print("Q-learning vs Random")
    result = evaluate(env, (TDPlayer(Q2), RandomPlayer()), 100)
    print("-- 1P: ", "#Win={}, #Lose={}, #Draw={}".format(result[0], result[1], result[2]))
    result = evaluate(env, (RandomPlayer(), TDPlayer(Q2)), 100)
    print("-- 2P: ", "#Win={}, #Lose={}, #Draw={}".format(result[1], result[0], result[2]))
    print()

    print("Sarsa vs Q-learning")
    result = evaluate(env, (TDPlayer(Q1), TDPlayer(Q2)), 100)
    print("-- 1P: ", "#Win={}, #Lose={}, #Draw={}".format(result[0], result[1], result[2]))
    result = evaluate(env, (TDPlayer(Q2), TDPlayer(Q1)), 100)
    print("-- 2P: ", "#Win={}, #Lose={}, #Draw={}".format(result[1], result[0], result[2]))
    print()

