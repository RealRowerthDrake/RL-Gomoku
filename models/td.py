from collections import defaultdict
from collections import deque

class Agent(object):
    def __init__(self, eval_fn, ctrl_fn, maxlen=2):
        self.states = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)

        self._eval = eval_fn
        self._ctrl = ctrl_fn

    def eval(self, reward, terminal=False):
        if terminal:
            self._eval(self.states[-1], self.actions[-1], reward, None, None)
        else:
            self._eval(self.states[0], self.actions[0],
                       reward,
                       self.states[1], self.actions[1])

    def ctrl(self, state):
        action = self._ctrl(state)

        self.states.append(state)
        self.actions.append(action)
        return action

    def reset(self):
        self.states.clear()
        self.actions.clear()


def train(env, build_fn, episodes=100, epsilon=0.1, alpha=0.01, gamma=0.9):
    Q = defaultdict(lambda: None)
    params = {
        'eps': epsilon,
        'alpha': alpha,
        'gamma': gamma,
    }
    agents = [Agent(*build_fn(Q, **params)) for _ in range(env.num_player)]

    for i in range(episodes):
        state = env.reset()
        for agent in agents: agent.reset()

        action = agents[state.cur_player].ctrl(state)

        last_reward = None
        done = False
        while not done:
            state, reward, done, _ = env.step(action)

            if not done:
                action = agents[state.cur_player].ctrl(state)
            if last_reward is not None:
                agents[state.cur_player].eval(last_reward - reward, done)

            last_reward = reward

        next_player = 1 - state.cur_player
        agents[next_player].eval(last_reward, done)

    return Q
