import torch
from torch.autograd import Variable
from torch.nn.functional import softmax, mse_loss
import numpy as np
import random
import copy
import os

# random.seed(0)
torch.manual_seed(0)


class AdvNet:
    def __init__(self, n_input=6, n_msg=2, n_hidden=16, lr=0.001):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_msg),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")


    def sample(self, memory, replay_size):

        if len(memory) < replay_size:
            return

        replay_data = random.sample(memory, replay_size)
        states = (torch.from_numpy(
            np.stack([tmp[0] for tmp in replay_data])).float().to(self.device))
        actions = (torch.from_numpy(
            np.vstack([tmp[1] for tmp in replay_data])).long().to(self.device))
        rewards = (torch.from_numpy(
            np.vstack([tmp[2] for tmp in replay_data])).float().to(self.device))

        return states, actions, rewards

    def replay(self, memory, replay_size):

        self.optimizer.zero_grad()

        states, actions, rewards = self.sample(
            memory, replay_size)

        q_values = self.model(states)
        q_values_pred = q_values.gather(1, actions)

        loss = mse_loss(q_values_pred, rewards)

        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        with torch.no_grad():
            return self.model(torch.Tensor(s))

    def save_model(self, path, number=''):
        full_path = os.path.join(path, "adv" + number + ".pth")
        torch.save(self.model.state_dict(), full_path)
        print(f"Model saved {full_path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded {path}")

    def deploy(self, s):
        with torch.no_grad():
            q_values = self.model(torch.Tensor(s))
            return (torch.argmax(q_values)).item()
