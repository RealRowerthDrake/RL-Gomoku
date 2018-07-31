from collections import deque
class Agent(object):
    def __init__(self, maxlen=2):
        self.state = deque(maxlen=maxlen)
        self.action = deque(maxlen=maxlen)

from collections import defaultdict
def train(env, build_fn, episodes=100, epsilon=0.1, alpha=0.01, gamma=0.1):
    Q = defaultdict(lambda: None)
    eval_fn, ctrl_fn = build_fn(Q, eps=epsilon, alpha=alpha, gamma=gamma)

    for i in range(episodes):
        agents = [Agent() for _ in range(env.num_player)]

        state = env.reset()
        action = ctrl_fn(state)
        last_reward = 0
        last_player = None

        agents[0].state.append(state)
        agents[0].action.append(action)

        while(True):
            state, reward, done, _ = env.step(action)
            action = ctrl_fn(state)

            player = env.cur_player
            agents[player].state.append(state)
            agents[player].action.append(action)
            if len(agents[player].state) > 1:
                eval_fn(agents[player].state[0], agents[player].action[0],
                        last_reward - reward,
                        agents[player].state[1], agents[player].action[1])

            if done:
                # Update last player
                eval_fn(agents[last_player].state[1], agents[last_player].action[1],
                        reward,
                        None, None)
                break

            last_reward = reward
            last_player = player

    return Q
