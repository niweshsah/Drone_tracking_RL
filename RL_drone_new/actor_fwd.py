from torch import nn
import torch.nn.functional as F
import torch

class Actor(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, max_action=6.0, hidden_dim=64):
        super(Actor, self).__init__()
        # Architecture: 2 hidden layers, 64 neurons each [cite: 621, 626]
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        # Activation: ReLU for hidden, tanh for output [cite: 322, 323]
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        # Scaled to max_action (6 m/s) [cite: 324, 626]
        return self.max_action * torch.tanh(self.l3(x))