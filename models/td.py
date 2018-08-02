from collections import deque
from collections import defaultdict


class Agent(object):
    def __init__(self, maxlen=2):
        self.states = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)


def train(env, build_fn, episodes=100, epsilon=0.1, alpha=0.01, gamma=0.1):
    Q = defaultdict(lambda: None)
    eval_fn, ctrl_fn = build_fn(Q, eps=epsilon, alpha=alpha, gamma=gamma)

    for i in range(episodes):
        agents = [Agent() for _ in range(env.num_player)]

        state = env.reset()
        action = ctrl_fn(state)
        last_reward = 0
        last_player = None

        agents[0].states.append(state)
        agents[0].actions.append(action)

        while True:
            state, reward, done, _ = env.step(action)
            action = ctrl_fn(state)

            player = state.cur_player
            agents[player].states.append(state)
            agents[player].actions.append(action)
            if len(agents[player].states) > 1:
                eval_fn(agents[player].states[0], agents[player].actions[0],
                        last_reward - reward,
                        agents[player].states[1], agents[player].actions[1])

            if done:
                # Update last player
                eval_fn(agents[last_player].states[1], agents[last_player].actions[1],
                        reward,
                        None, None)
                break

            last_reward = reward
            last_player = player

    return Q
