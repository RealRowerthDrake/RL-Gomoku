import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(num_actions, 100)
        self.policy_head = nn.Linear(100, num_actions)
        self.value_head = nn.Linear(100, 1)

        self.log_prob = None
        self.value = None

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy = F.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        return policy, value


def build_ctrl_fn(net, train=False):
    def ctrl_fn(state):
        state_feats = torch.from_numpy(state.board.flatten()).float().unsqueeze(0)
        probs, value = net(state_feats)
        mask = torch.zeros_like(probs).index_fill_(1, torch.from_numpy(state.valid_actions), 1)
        probs = probs * mask

        if train:
            m = Categorical(probs)
            action = m.sample()

            net.log_prob = m.log_prob(action)
            net.value = value
            return action.item()
        else:
            action = probs.argmax(dim=-1)
            return action.item()

    return ctrl_fn


def build_optim_fn(optimizer, gamma):
    def optim_fn(reward, value, next_value, log_prob):
        target = reward + gamma * next_value
        delta = target - value
        policy_loss = -log_prob * delta.data
        value_loss = F.smooth_l1_loss(value, target.data)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return optim_fn


def train(env, episodes, gamma=0.9):
    num_actions = env._board_size ** 2
    nets = [PolicyNet(num_actions), PolicyNet(num_actions)]
    optimizers = [optim.Adam(net.parameters(), lr=1e-2) for net in nets]

    ctrl_fns = [build_ctrl_fn(net, train=True) for net in nets]
    optim_fns = [build_optim_fn(optimizer, gamma) for optimizer in optimizers]

    for episode in range(episodes):
        state = env.reset()
        action = ctrl_fns[state.cur_player](state)

        last_reward = None
        done = False
        while not done:
            state, reward, done, _ = env.step(action)

            if last_reward is not None:
                r = last_reward - reward
                value = nets[state.cur_player].value
                log_prob = nets[state.cur_player].log_prob
                if not done:
                    action = ctrl_fns[state.cur_player](state)
                    next_value = nets[state.cur_player].value
                else:
                    next_value = torch.tensor([[0.0]])

                optim_fns[state.cur_player](r, value, next_value, log_prob)

                del value
                del log_prob
            else:
                action = ctrl_fns[state.cur_player](state)

            last_reward = reward

    return [build_ctrl_fn(net, train=False) for net in nets]
