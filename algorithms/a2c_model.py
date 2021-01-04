import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticModel(nn.Module):
    def __init__(self, n_state, n_actions):
        super(ActorCriticModel, self).__init__()

        self.fc1 = nn.Linear(n_state, 32)

        self.action1 = nn.Linear(32, 32)
        self.action2 = nn.Linear(32, n_actions)

        self.value1 = nn.Linear(32, 32)
        self.value2 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))

        action_x = F.relu(self.action1(x))
        action_probs = F.softmax(self.action2(action_x), dim=-1)

        value_x = F.relu(self.value1(x))
        state_values = self.value2(value_x)

        return action_probs, state_values


class PolicyNetwork:
    def __init__(self, n_state, n_actions):
        self.model = ActorCriticModel(n_state, n_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 0.01)

    def predict(self, s):
        return self.model(torch.Tensor(s))

    def update(self, returns, log_probs, state_values, episode):
        loss = 0
        for log_prob, value, Gt in zip(log_probs, state_values, returns):
            advantage = Gt - value.item()
            policy_loss = -log_prob * advantage
            value_loss = F.smooth_l1_loss(value[0], Gt)
            loss += policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, s):
        action_probs, state_value = self.predict(s)
        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs[action])
        return action, log_prob, state_value

    def save_model(self, path):
        full_path = os.path.join(path, "a2c.pth")
        torch.save(self.model.state_dict(), full_path)
        print(f"Model saved {full_path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded {path}")
