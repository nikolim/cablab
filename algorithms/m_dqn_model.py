import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import random
from collections import deque, namedtuple


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")


class DDQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size, seed, layer_type="ff"):
        super(DDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size

        self.head_1 = nn.Linear(self.input_shape, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)
        weight_init([self.head_1, self.ff_1])

    def forward(self, input):

        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        return out


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device, seed, gamma, n_step=1):
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

    def add(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return()
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

    def calc_multistep_return(self):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma ** idx * self.n_step_buffer[idx][2]

        return (
            self.n_step_buffer[0][0],
            self.n_step_buffer[0][1],
            Return,
            self.n_step_buffer[-1][3],
            self.n_step_buffer[-1][4],
        )

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.stack([e.state for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .long()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.stack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class M_DQN_Agent:
    def __init__(
        self,
        state_size,
        action_size,
        layer_size,
        BATCH_SIZE,
        BUFFER_SIZE,
        LR,
        TAU,
        GAMMA,
        UPDATE_EVERY,
        device,
        seed,
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0

        self.action_step = 4
        self.last_action = None

        # Q-Network
        self.qnetwork_local = DDQN(state_size, action_size, layer_size, seed).to(device)
        self.qnetwork_target = DDQN(state_size, action_size, layer_size, seed).to(
            device
        )
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(
            BUFFER_SIZE, BATCH_SIZE, self.device, seed, self.GAMMA, 1
        )
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                self.Q_updates += 1

    def act(self, state, eps=0.0):
        if self.action_step == 4:
            state = np.array(state)

            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            # Epsilon-greedy
            if random.random() > eps:
                action = np.argmax(action_values.cpu().data.numpy())
                self.last_action = action
                return action
            else:
                action = random.choice(np.arange(self.action_size))
                self.last_action = action
                return action
        else:
            self.action_step += 1
            return self.last_action

    def learn(self, experiences):

        entropy_tau = 0.03
        alpha = 0.9

        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach()
        # calculate entropy term with logsum
        logsum = torch.logsumexp(
            (Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1)) / entropy_tau, 1
        ).unsqueeze(-1)

        tau_log_pi_next = (
            Q_targets_next
            - Q_targets_next.max(1)[0].unsqueeze(-1)
            - entropy_tau * logsum
        )
        # target policy
        pi_target = F.softmax(Q_targets_next / entropy_tau, dim=1)
        Q_target = (
            self.GAMMA
            * (pi_target * (Q_targets_next - tau_log_pi_next) * (1 - dones)).sum(1)
        ).unsqueeze(-1)

        # calculate munchausen addon with logsum trick
        q_k_targets = self.qnetwork_target(states).detach()
        v_k_target = q_k_targets.max(1)[0].unsqueeze(-1)
        logsum = torch.logsumexp((q_k_targets - v_k_target) / entropy_tau, 1).unsqueeze(
            -1
        )
        log_pi = q_k_targets - v_k_target - entropy_tau * logsum
        munchausen_addon = log_pi.gather(1, actions)

        # calc munchausen reward:
        munchausen_reward = rewards + alpha * torch.clamp(
            munchausen_addon, min=-1, max=0
        )

        # Compute Q targets for current states
        Q_targets = munchausen_reward + Q_target

        # Get expected Q values from local model
        q_k = self.qnetwork_local(states)
        Q_expected = q_k.gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)  # mse_loss
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data
            )

    def save_model(self, path):
        full_path = os.path.join(path, "mdqn.pth")
        torch.save(self.qnetwork_local.state_dict(), full_path)
        print(f"Model saved {full_path}")

    def load_model(self, path):
        self.qnetwork_local.load_state_dict(torch.load(path, map_location=self.device))
        self.qnetwork_local.eval()
        print(f"Model loaded {path}")
