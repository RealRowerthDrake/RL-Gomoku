import random
from envs import GomokuEnv
from players import *


def evaluate(env, players, num_plays):
    def play(players, num_plays):
        def play_once():
            state = env.reset()
            while(True):
                action = players[state.cur_player].move(state)
                state, reward, done, _ = env.step(action)
                if done:
                    return reward if state.cur_player == 1 else -reward
        stats = [0] * 3
        for _ in range(num_plays):
            stats[play_once()] += 1
        return stats[1], stats[-1], stats[0]

    result = play(players, num_plays)
    print("-- 1P: ", "#Win={}, #Lose={}, #Draw={}".format(result[0], result[1], result[2]))
    result = play(players[::-1], num_plays)
    print("-- 2P: ", "#Win={}, #Lose={}, #Draw={}".format(result[1], result[0], result[2]))
    print()


if __name__ == '__main__':
    env = GomokuEnv(3, 3)

    print("Random vs Random")
    evaluate(env, (RandomPlayer(), RandomPlayer()), 100)

    from models.td import train
    from models import sarsa, q_learning
    Q1 = train(env, sarsa.build_fn, 10000)
    Q2 = train(env, q_learning.build_fn, 10000)

    print("Sarsa vs Random")
    evaluate(env, (TDPlayer(Q1), RandomPlayer()), 100)

    print("Q-Learning vs Random")
    evaluate(env, (TDPlayer(Q2), RandomPlayer()), 100)

    from models import reinforce
    """
        Note: Since the variance of REINFORCE is quite large, it cannot be trainning for too long, or the gradient might be exploded.
    """
    F1 = reinforce.train(env, 1000)


    print("REINFORCE vs Random")
    evaluate(env, (PolicyGradientPlayer(F1), RandomPlayer()), 100)

    print("MCTSPlayer vs Random")
    evaluate(env, (MCTSPlayer(1000), RandomPlayer()), 10)

    print("Sarsa vs Q-learning")
    evaluate(env, (TDPlayer(Q1), TDPlayer(Q2)), 100)

    print("REINFORCE vs Sarsa")
    evaluate(env, (PolicyGradientPlayer(F1), TDPlayer(Q1)), 100)

    print("REINFORCE vs Q-learning")
    evaluate(env, (PolicyGradientPlayer(F1), TDPlayer(Q2)), 100)

    print("MCTSPlayer vs Sarsa")
    evaluate(env, (MCTSPlayer(1000), TDPlayer(Q1)), 10)

    print("MCTSPlayer vs Q-Learning")
    evaluate(env, (MCTSPlayer(1000), TDPlayer(Q2)), 10)

    print("MCTSPlayer vs REINFORCE")
    evaluate(env, (MCTSPlayer(1000), PolicyGradientPlayer(F1)), 10)
