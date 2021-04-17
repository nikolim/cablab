import torch
from torch.autograd import Variable
from torch.nn.functional import softmax, mse_loss
import numpy as np
import random
import copy
import os

torch.manual_seed(0)

counter = 0

class VDNMixer(torch.nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs):
        return torch.sum(agent_qs, dim=2, keepdim=True)

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
        self.losses = []

        self.mixer = VDNMixer()
        
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
        states2 = (
            torch.from_numpy(np.stack([tmp[1] for tmp in replay_data]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(np.vstack([tmp[2] for tmp in replay_data]))
            .long()
            .to(self.device)
        )
        actions2 = (
            torch.from_numpy(np.vstack([tmp[3] for tmp in replay_data]))
            .long()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(np.stack([tmp[4] for tmp in replay_data]))
            .float()
            .to(self.device)
        )
        next_states2 = (
            torch.from_numpy(np.stack([tmp[5] for tmp in replay_data]))
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(np.vstack([tmp[6] for tmp in replay_data]))
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([tmp[7] for tmp in replay_data]).astype(np.uint8)
            )
            .float()
            .to(self.device)
        )

        return states, states2,actions, actions,rewards, next_states, next_states,dones

    def replay(self, memory, replay_size, gamma):

        self.optimizer.zero_grad()

        states, states2, actions, actions2, rewards, next_states, next_states2,dones = self.sample(memory, replay_size)

        #assert len(states) == len(actions) == len(next_states)

        q_values = self.model(states)
        q_values_next = self.model_target(next_states).detach()
        q_values_next_max1 = q_values_next.max(1)[0].unsqueeze(-1)
        chosen_action_qvals1 = q_values.gather(1, actions)

        q_values2 = self.model(states2)
        q_values_next2 = self.model_target(next_states2).detach()
        q_values_next_max2 = q_values_next2.max(1)[0].unsqueeze(-1)
        chosen_action_qvals2 = q_values2.gather(1, actions2)

        #mix (use VDN later)
        # chosen_action_qvals = chosen_action_qvals1 + chosen_action_qvals2
        # target_max_qvals = q_values_next_max1 + q_values_next_max2

        chosen_action_qvals = torch.sum(torch.cat((chosen_action_qvals1, chosen_action_qvals2),1),1).unsqueeze(-1)
        target_max_qvals = torch.sum(torch.cat((q_values_next_max1, q_values_next_max2),1),1).unsqueeze(-1)

        q_targets_done = dones.view(-1, 1) * rewards
        q_targets_not_done = (1 - dones) * (
            rewards + gamma * target_max_qvals
        )
        q_targets = q_targets_done + q_targets_not_done

        loss = mse_loss(chosen_action_qvals, q_targets)
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
        if random.random() < 0.05:
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
