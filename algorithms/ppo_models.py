import os
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical

from collections import deque

torch.manual_seed(0)


class Memory:
    def __init__(self):
        self.actions = deque()
        self.states = deque()
        self.logprobs = deque()
        self.rewards = deque()
        self.is_terminal = deque()

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminal.clear()


class ActorCritic(nn.Module):
    def __init__(self, n_state, n_actions):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def act(self, state, memory):

        state = torch.Tensor(state).to(self.device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def deploy(self, state):
        state = torch.Tensor(state).to(self.device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()

    def evaluate(self, state, action):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, n_state, n_actions):
        self.lr_actor = 0.0001
        self.lr_critic = 0.001    
        self.betas = (0.9, 0.999)
        self.gamma = 0.9
        self.eps_clip = 0.2
        self.K_epochs = 4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(n_state, n_actions).to(self.device)
        
        #self.optimizer = torch.optim.Adam(
        #    self.policy.parameters(), lr=self.lr, betas=self.betas
        #)

        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
                    ])

        self.policy_old = ActorCritic(n_state, n_actions).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.losses = []

    def update(self, memory, episode):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.is_terminal)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # normalise rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.stack(tuple(memory.states)).to(self.device).detach()
        old_actions = torch.stack(tuple(memory.actions)).to(self.device).detach()
        old_logprobs = torch.stack(tuple(memory.logprobs)).to(self.device).detach()

        for i in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )
            if i == (self.K_epochs -1):
               self.losses.append(torch.mean(loss).item())
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        mean_entropy = torch.mean(dist_entropy).item()
        return mean_entropy

    def save_model(self, path):
        full_path = os.path.join(path, "ppo.pth")
        torch.save(self.policy.state_dict(), full_path)
        print(f"Model saved {full_path}")

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()
        print(f"Model loaded {path}")
