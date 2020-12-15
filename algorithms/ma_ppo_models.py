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
    def __init__(self):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(11, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 5),
            nn.Softmax(dim=-1),
        )

        self.actor2 = nn.Sequential(
            nn.Linear(11, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 5),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(11, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def act(self, states, memory):

        state, state2 = states

        state = torch.Tensor(state).to(self.device)
        state2 = torch.Tensor(state2).to(self.device)

        action_probs = self.actor(state)
        action_probs2 = self.actor2(state2)

        dist = Categorical(action_probs)
        dist2 = Categorical(action_probs2)

        action = dist.sample()
        action2 = dist2.sample()

        memory.states.append((state, state2))
        memory.actions.append((action, action2))
        memory.logprobs.append((dist.log_prob(action), dist2.log_prob(action2)))

        return action.item(), action2.item()

    def evaluate(self, states, actions):

        state, state2 = states
        action, action2 = actions

        action_probs = self.actor(state)
        action_probs2 = self.actor(state2)

        dist = Categorical(action_probs)
        dist2 = Categorical(action_probs2)

        action_logprobs = dist.log_prob(action)
        action_logprobs2 = dist.log_prob(action2)

        dist_entropy = dist.entropy()
        dist_entropy2 = dist2.entropy()

        # stacked_state = torch.cat((state, state2),1)

        state_value = self.critic(state)
        state_value2 = self.critic(state2)

        return (
            (action_logprobs, action_logprobs2),
            (torch.squeeze(state_value), torch.squeeze(state_value2)),
            (dist_entropy, dist_entropy2),
        )


class MAPPO:
    def __init__(self):

        self.lr = 0.001
        self.betas = (0.9, 0.999)
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic().to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.lr, betas=self.betas
        )
        self.optimizer2 = torch.optim.Adam(
            self.policy.parameters(), lr=self.lr, betas=self.betas
        )

        self.policy_old = ActorCritic().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory, episode):

        rewards = []
        rewards2 = []
        discounted_reward = 0
        discounted_reward2 = 0

        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.is_terminal)
        ):
            if is_terminal:
                discounted_reward = 0
                discounted_reward2 = 0

            discounted_reward = reward[0] + (self.gamma * discounted_reward)
            discounted_reward2 = reward[1] + (self.gamma * discounted_reward2)
            rewards.insert(0, discounted_reward)
            rewards2.insert(0, discounted_reward2)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        rewards2 = torch.tensor(rewards2, dtype=torch.float32).to(self.device)
        rewards2 = (rewards2 - rewards2.mean()) / (rewards2.std() + 1e-5)

        tmp1 = memory.states
        tmp2 = memory.actions
        tmp3 = memory.logprobs

        states_1 = [tmp[0] for tmp in memory.states]
        states_2 = [tmp[1] for tmp in memory.states]

        actions_1 = [tmp[0] for tmp in memory.actions]
        actions_2 = [tmp[1] for tmp in memory.actions]

        logprobs_1 = [tmp[0] for tmp in memory.logprobs]
        logprobs_2 = [tmp[1] for tmp in memory.logprobs]

        old_states = (
            torch.stack(states_1).to(self.device).detach(),
            torch.stack(states_2).to(self.device).detach(),
        )
        old_actions = (
            torch.stack(actions_1).to(self.device).detach(),
            torch.stack(actions_2).to(self.device).detach(),
        )
        old_logprobs = (
            torch.stack(logprobs_1).to(self.device).detach(),
            torch.stack(logprobs_2).to(self.device).detach(),
        )

        for _ in range(self.K_epochs):
            logprobs, (state_value, state_value2), dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )
            ratios = torch.exp(logprobs[0] - old_logprobs[0].detach())
            ratios2 = torch.exp(logprobs[1] - old_logprobs[1].detach())

            advantages = rewards - state_value.detach()
            advantages2 = rewards2 - state_value2.detach()

            surr1 = ratios * advantages
            surr12 = ratios2 * advantages2

            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            surr22 = (
                torch.clamp(ratios2, 1 - self.eps_clip, 1 + self.eps_clip) * advantages2
            )

            loss1 = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_value, rewards)
                - 0.01 * dist_entropy[0]
            )

            loss2 = (
                -torch.min(surr12, surr22)
                + 0.5 * self.MseLoss(state_value, rewards2)
                - 0.01 * dist_entropy[1]
            )

            loss = torch.add(loss1, loss2)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        uncertainty = (
            torch.sum(torch.squeeze(torch.add(logprobs[0], logprobs[1])))
        ).item() * -1
        self.policy_old.load_state_dict(self.policy.state_dict())
        return uncertainty
