import numpy as np
import random

def build_fn(Q, **params):
    """Build evaluate & control functions.

    Args:
        params:
            eps (float): epsilon-greedy improvement

    Returns:
        eval_fn (func)
        ctrl_fn (func): policy(dict) -> action(int)
    """
    def eval_fn(state, action, reward, next_state, next_action):
        if next_state is not None and len(Q[next_state]) > 0:
            target = reward + params['gamma'] * max(Q[next_state].values())
        else:
            target = reward
        delta = target - Q[state][action]
        Q[state][action] += params['alpha'] * delta

    def ctrl_fn(state):
        if Q[state] is None:
            available_actions = list(zip(*np.where(state==-1)))
            Q[state] = dict(zip(available_actions, np.zeros(len(available_actions))))

        if not Q[state]:
            return None

        policy = Q[state]
        if random.uniform(0, 1) < params['eps']:
            return random.choice(list(policy.keys()))
        else:
            return max(policy.items(), key=lambda x:x[-1])[0]

    return eval_fn, ctrl_fn
