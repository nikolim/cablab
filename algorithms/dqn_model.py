import torch
from torch.autograd import Variable
from torch.nn.functional import softmax, mse_loss
import numpy as np
import random
import copy
import os

torch.manual_seed(0)

counter = 0


class DQN:
    def __init__(self, n_state, n_action, n_hidden=16, lr=0.01):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_action),
        )
        self.model_target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.episode_loss = 0

        # needed for munchhausen
        self.entropy_tau = 0.03
        self.alpha = 0.9

    def sample(self, memory, replay_size):

        if len(memory) < replay_size:
            return

        replay_data = random.sample(memory, replay_size)
        states = (
            torch.from_numpy(np.stack([tmp[0] for tmp in replay_data]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(np.vstack([tmp[1] for tmp in replay_data]))
            .long()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(np.stack([tmp[2] for tmp in replay_data]))
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(np.vstack([tmp[3] for tmp in replay_data]))
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([tmp[4] for tmp in replay_data]).astype(np.uint8)
            )
            .float()
            .to(self.device)
        )

        return states, actions, rewards, next_states, dones

    def replay(self, memory, replay_size, gamma):

        self.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = self.sample(memory, replay_size)

        q_values = self.model(states)
        q_values_next = self.model_target(next_states).detach()
        q_values_pred = q_values.gather(1, actions)

        q_targets_done = dones.view(-1, 1) * rewards
        q_targets_not_done = (1 - dones) * (
            rewards + gamma * q_values_next.max(1)[0].unsqueeze(-1)
        )

        q_targets = q_targets_done + q_targets_not_done

        loss = mse_loss(q_values_pred, q_targets)
        self.episode_loss += loss.item()
        loss.backward()
        self.optimizer.step()

    def replay_munchhausen(self, memory, replay_size, gamma):

        self.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = self.sample(memory, replay_size)

        q_values_next = self.model_target(next_states).detach()

        logsum = torch.logsumexp(
            (q_values_next - q_values_next.max(1)[0].unsqueeze(-1)) / self.entropy_tau,
            1,
        ).unsqueeze(-1)
        tau_log_next = (
            q_values_next
            - q_values_next.max(1)[0].unsqueeze(-1)
            - self.entropy_tau * logsum
        )
        pi_target = softmax(q_values_next / self.entropy_tau, dim=1)
        q_target = (
            gamma * (pi_target * (q_values_next - tau_log_next) * (1 - dones)).sum(1)
        ).unsqueeze(-1)
        q_values = self.model_target(states).detach()
        v_values = q_values.max(1)[0].unsqueeze(-1)
        logsum = torch.logsumexp((q_values - v_values) / self.entropy_tau, 1).unsqueeze(
            -1
        )
        log_pi = q_values - v_values - self.entropy_tau * logsum

        munchausen_addon = log_pi.gather(1, actions)
        munchausen_reward = rewards + self.alpha * torch.clamp(
            munchausen_addon, min=-1, max=0
        )

        q_targets = munchausen_reward + q_target

        q_k = self.model(states)
        Q_expected = q_k.gather(1, actions)
        loss = mse_loss(Q_expected, q_targets)

        self.episode_loss += loss.item()

        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        with torch.no_grad():
            return self.model(torch.Tensor(s))

    def target_predict(self, s):
        with torch.no_grad():
            return self.model_target(torch.Tensor(s))

    def copy_target(self):
        self.model_target.load_state_dict(self.model.state_dict())

    def save_model(self, path, number=""):
        full_path = os.path.join(path, "dqn" + number + ".pth")
        torch.save(self.model.state_dict(), full_path)
        print(f"Model saved {full_path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded {path}")

    def deploy(self, s):
        if random.random() < 0.01:
            return random.randint(0, 6)
        else:
            with torch.no_grad():
                q_values = self.model(torch.Tensor(s))
                return (torch.argmax(q_values)).item()


def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
            action = torch.argmax(q_values).item()
            return action

    return policy_function
