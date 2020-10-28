import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(m):
    weight = m.weight.data
    weight.normal_(0, 1)
    weight *= 1.0 / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
    nn.init.constant_(m.bias.data, 0)
    return m


class MLP(nn.Module):
    def __init__(self, num_inputs):
        super(MLP, self).__init__()

        self.linear1 = init_layer(nn.Linear(num_inputs, 64))
        self.linear2 = init_layer(nn.Linear(64, 128))
        self.linear3 = init_layer(nn.Linear(128, 128))
        self.linear4 = init_layer(nn.Linear(128, 128))
        self.actor_hidden = init_layer(nn.Linear(128, 128))
        self.critic = init_layer(nn.Linear(128, 1))
        self.tanh = nn.Tanh()

        self.train()

    def forward(self, inputs):
        x = self.tanh(self.linear1(inputs))
        x = self.tanh(self.linear2(x))
        x = self.tanh(self.linear3(x))
        x = self.tanh(self.linear4(x))
        actor_feature = self.tanh(self.actor_hidden(x))
        critic = self.critic(x)

        return actor_feature, critic
