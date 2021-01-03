import torch
from torch.autograd import Variable
import random
import copy

random.seed(0)
torch.manual_seed(0)

class DQN():
    def __init__(self, n_state, n_action, n_hidden=32, lr=0.01):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(n_state, n_hidden),
                        torch.nn.ReLU(),
                        torch.nn.Linear(n_hidden, n_action),
                )

    def update(self, s, y):
        y_pred = self.model(torch.Tensor(s))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        with torch.no_grad():
            return self.model(torch.Tensor(s))


def calc_q_values(s): 
    q_values = []
    for i in range(4): 
        if s[i] == 1:
            q_values.append(-1)
        else: 
            q_values.append(-5)
    for j in range(4,6): 
        if s[j] == 1:
            q_values.append(100)
        else: 
            q_values.append(-10)
    return q_values

def create_test_state(): 
    state = []
    for i in range(4): 
        tmp = 1 if random.random < 0.
        state.append(tmp)
