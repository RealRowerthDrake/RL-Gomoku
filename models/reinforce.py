import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(num_actions, 100)
        self.fc2 = nn.Linear(100, num_actions)

        self.log_probs = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        scores = self.fc2(x)
        prob = F.softmax(scores, dim=1)
        return prob


def build_ctrl_fn(net):
    def ctrl_fn(state):
        state_feats = torch.from_numpy(state.board.flatten()).float().unsqueeze(0)
        probs = net(state_feats)
        mask = torch.zeros_like(probs).index_fill_(1, torch.from_numpy(state.valid_actions), 1)
        probs = probs * mask
        action = probs.multinomial(1).item()
        log_prob = probs[:, action].log()

        net.log_probs.append(log_prob)
        return action

    return ctrl_fn



def train(env, episodes, gamma=0.9):
    num_actions = env._board_size ** 2
    nets = [PolicyNet(num_actions), PolicyNet(num_actions)]
    optimizers = [optim.Adam(net.parameters()) for net in nets]
    ctrl_fns = [build_ctrl_fn(net) for net in nets]

    for episode in range(episodes):
        state = env.reset()
        rewards = []

        done = False
        while not done:
            action = ctrl_fns[state.cur_player](state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)

        rewards = np.array(rewards)
        rewards[:-1] -= rewards[1:]

        rewards = [rewards[0::2], rewards[1::2]]
        for i, (net, optimizer) in enumerate(zip(nets, optimizers)):
            Rs, R = [], 0
            for r in reversed(rewards[i]):
                R = gamma * R + r
                Rs.insert(0, R)
            Rs = torch.tensor(Rs)
            Rs = (Rs - Rs.mean()) / (Rs.std() + 1e-3)

            loss = []
            for t, R in enumerate(Rs):
                loss.append(-net.log_probs[t] * R)
            loss = torch.cat(loss).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del net.log_probs[:]
        del rewards

    return ctrl_fns

