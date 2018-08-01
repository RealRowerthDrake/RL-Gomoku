import numpy as np
import numba as nb
import random


def UCB1(node):
    mean = node.rewards / (node.n + 1e-3)
    var = 2 * np.log(node.parent.n + 1) / (node.n + 1e-3)
    std = np.sqrt(var)
    return mean + std


class TreeNode(object):
    def __init__(self, parent, state):
        self.parent = parent
        self.state  = state

        self.rewards = 0
        self.n = 0
        self.childs = []

    def select(self, policy=UCB1):
        return max(self.childs, key=policy)

    def expand(self):
        for action in self.state.valid_actions:
            self.childs.append(TreeNode(self, self.state.act(action)))
        return self

    def update(self, result):
        self.rewards += result
        self.n += 1
        return self.parent

    def isLeaf(self):
        return self.state.done


def UCT(state, max_iter):
    root = TreeNode(None, state)

    for i in range(max_iter):
        node = root
        while(node.childs): node = node.select()              # Select
        if(not node.isLeaf()): node = node.expand().select()  # Expand

        # Rollout
        rollout = node.state
        while not rollout.done:
            rollout = rollout.act(random.choice(rollout.valid_actions))

        # Backpropagate
        reward = rollout.result
        while(node != None):
            if node.state.cur_player == rollout.cur_player:
                node = node.update(reward)
            else:
                node = node.update(-reward)

    return root


class MCTSPlayer(object):
    def __init__(self, n_simu):
        self._n_simu = n_simu

    def move(self, state):
        root = UCT(state, self._n_simu)
        return root.select().state._last_move
