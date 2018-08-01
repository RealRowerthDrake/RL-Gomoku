import numpy as np
import numba as nb
from random import choice


eps = 1e-3


@nb.jit('i4(f4[:],i4[:])')
def UCB1(rewards, n):
    mean = rewards / (n + eps)
    var = 2 * np.log(n.sum() + 1) / (n + eps)
    std = np.sqrt(var)
    ucb = mean + std
    return np.argmax(ucb)


class TreeNode(object):
    def __init__(self, parent, state):
        self.parent = parent
        self.state  = state

        self.rewards = None
        self.n = None
        self.childs = []

    def select(self):
        return self.childs[UCB1(self.rewards, self.n)]

    def expand(self):
        for action in self.state.valid_actions:
            self.childs.append(TreeNode(self, self.state.act(action)))

        self.rewards = np.zeros(len(self.childs), dtype='f4')
        self.n = np.zeros(len(self.childs), dtype='i4')
        return self

    def update(self, result):
        if self.parent is not None:
            idx = self.parent.childs.index(self)
            self.parent.rewards[idx] += result
            self.parent.n[idx] += 1
        return self.parent

    def isLeaf(self):
        return self.state.done


def UCT(state, max_iter):
    root = TreeNode(None, state)

    for i in range(max_iter):
        node = root

        # Select
        while(node.childs):
            node = node.select()

        # Expand
        if(not node.isLeaf()):
            node = node.expand().select()

        # Rollout
        rollout = node.state
        while not rollout.done:
            rollout = rollout.act(choice(rollout.valid_actions))

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
