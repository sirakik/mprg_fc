import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MLPModel, self).__init__()
        # shared layer
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.tanh = nn.Tanh()

        # actor
        self.linear_actor = nn.Linear(256, action_dim)
        self.softmax = nn.Softmax(dim=-1)

        # critic
        self.linear_value = nn.Linear(256, 1)

        self.train()

    def forward(self, inputs):
        # shared layer
        x = self.tanh(self.linear1(inputs))
        x = self.tanh(self.linear2(x))
        x = self.tanh(self.linear3(x))
        x = self.tanh(self.linear4(x))

        # actor
        action_porbs = self.softmax(self.linear_actor(x))

        # critic
        state_value = self.linear_value(x)

        return action_porbs, state_value