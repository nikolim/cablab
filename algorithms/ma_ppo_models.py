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


class MAActorCritic(nn.Module):
    def __init__(self, n_agents, n_state, n_actions):
        super(MAActorCritic, self).__init__()

        self.n_agents = n_agents
        self.actors = []
        self.actors_optimizers = []
        for _ in range(n_agents):
            actor = nn.Sequential(
                    nn.Linear(n_state, 32),
                    nn.Tanh(),
                    nn.Linear(32, 32),
                    nn.Tanh(),
                    nn.Linear(32, n_actions),
                    nn.Softmax(dim=-1),
                    )

            self.actors.append(actor)
            self.actors_optimizers.append(actor.parameters())

        self.critic = nn.Sequential(
            nn.Linear(n_state, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def act(self, states, memory):

        states = [torch.Tensor(s).to(self.device) for s in states]
        action_probs = [self.actors[i](s) for i, s in enumerate(states)]
        dists = [Categorical(prob) for prob in action_probs]
        actions = [dist.sample() for dist in dists]
        logprobs = [dist.log_prob(action) for dist, action in zip(dists, actions)]

        memory.states.append(states)
        memory.actions.append(actions)
        memory.logprobs.append(logprobs)

        return [action.item() for action in actions]

    def deploy(self, states):
        states = [torch.Tensor(s).to(self.device) for s in states]
        action_probs = [self.actors[i](s) for i, s in enumerate(states)]
        dists = [Categorical(prob) for prob in action_probs]
        actions = [dist.sample() for dist in dists]
        return [action.item() for action in actions]

    def evaluate(self, states, actions):
        action_probs = [self.actors[i](s) for i, s in enumerate(states)]
        dists = [Categorical(prob) for prob in action_probs]
        action_logprobs = [
            dist.log_prob(action) for dist, action in zip(dists, actions)
        ]
        dist_entropys = [dist.entropy() for dist in dists]
        state_values = [self.critic(state) for state in states]
        
        state_values = [torch.squeeze(state_values[i]) for i in range(self.n_agents)]

        return action_logprobs, state_values, dist_entropys


class MAPPO:
    def __init__(self, n_state, n_actions):

        self.n_agents = 2
        self.lr = 0.001
        self.betas = (0.9, 0.999)
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = MAActorCritic(self.n_agents, n_state, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=self.lr, betas=self.betas
        )
        self.policy_old = MAActorCritic(self.n_agents, n_state, n_actions).to(
            self.device
        )
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory, episode):

        rewards = []
        discounted_rewards = [0] * self.n_agents

        for rewards_tmp, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminal)):
            if is_terminal:
                discounted_rewards = [0] * self.n_agents
            discounted_rewards = [reward + (self.gamma * discounted_reward) for reward, discounted_reward in zip(rewards_tmp, discounted_rewards)]
            rewards.insert(0, discounted_rewards)

        tmp_rewards = [list(tmp[i] for tmp in rewards) for i in range(self.n_agents)]

        rewards = [torch.tensor(reward, dtype=torch.float32).to(self.device) for reward in tmp_rewards]
        rewards = [(reward - reward.mean()) / (reward.std() + 1e-5) for reward in rewards]
        
        tmp_states = [(tmp[i] for tmp in memory.states) for i in range(self.n_agents)]
        tmp_actions = [(tmp[i] for tmp in memory.actions) for i in range(self.n_agents)]
        tmp_logprobs = [(tmp[i] for tmp in memory.actions) for i in range(self.n_agents)]

        old_states = [torch.stack(tuple(state)).to(self.device).detach() for state in tmp_states]
        old_actions = [torch.stack(tuple(action)).to(self.device).detach() for action in tmp_actions]
        old_logprobs = [torch.stack(tuple(logprob)).to(self.device).detach() for logprob in tmp_logprobs]

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropys = self.policy.evaluate(
                old_states, old_actions
            )
            ratios = [
                torch.exp(logprob - old_logprob.detach())
                for logprob, old_logprob in zip(logprobs, old_logprobs)
            ]
            advantages = [
                reward - state_value.detach()
                for reward, state_value in zip(rewards, state_values)
            ]
            surr1s = [
                (ratio * advantage) for ratio, advantage in zip(ratios, advantages)
            ]
            surr2s = [
                (torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage)
                for ratio, advantage in zip(ratios, advantages)
            ]
            losses = [
                (
                    -torch.min(surr1, surr2)
                    + 0.5 * self.MseLoss(state_value, reward)
                    - 0.01 * dist_entropy
                )
                for surr1, surr2, state_value, reward, dist_entropy in zip(
                    surr1s, surr2s, state_values, rewards, dist_entropys
                )
            ]
            self.optimizer.zero_grad()
            for loss in losses:
                loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        mean_entropy = torch.mean(sum(dist_entropys)).item()
        return mean_entropy

    def save_model(self, path):
        full_path = os.path.join(path, "mappo.pth")
        torch.save(self.policy.state_dict(), full_path)
        print(f"Model saved {full_path}")

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()
        print(f"Model loaded {path}")
