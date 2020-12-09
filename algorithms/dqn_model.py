import os
import torch
from torch.autograd import Variable
import math
import gym
import gym_cabworld
from torch.utils.tensorboard import SummaryWriter

class DqnEstimator():
    def __init__(self, n_feat, n_action, lr, writer):
        self.n_feat = n_feat
        self.models = []
        self.optimizers = []
        self.criterion = torch.nn.MSELoss()
        self.writer = writer
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        for _ in range(n_action):
            model = torch.nn.Sequential(
                torch.nn.Linear(int(n_feat), int(64)),
                torch.nn.ReLU(),
                torch.nn.Linear(int(64), int(64)),
                torch.nn.ReLU(),
                torch.nn.Linear(int(64), 1)
            )
            model.to(self.device)
            self.models.append(model)
            optimizer = torch.optim.Adam(model.parameters(), lr)
            self.optimizers.append(optimizer)
        writer.add_graph(self.models[0], torch.ones(n_feat))

    def update(self, s, a, y, episode):
        features = torch.Tensor(s, device=self.device)
        y_pred = self.models[a](features)
        loss = self.criterion(y_pred, Variable(torch.Tensor([y])))
        self.optimizers[a].zero_grad()
        loss.backward()
        self.optimizers[a].step()

    def predict(self, s):
        features = torch.Tensor(s, device=self.device)
        with torch.no_grad():
            return torch.tensor([model(features) for model in self.models], device=self.device)


