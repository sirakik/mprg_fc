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
    def __init__(self, num_inputs, action_space):
        super(MLP, self).__init__()

        self.linear1 = init_layer(nn.Linear(num_inputs, 256))
        self.linear2 = init_layer(nn.Linear(256, 512))
        self.linear3 = init_layer(nn.Linear(512, 512))
        self.linear4 = init_layer(nn.Linear(512, 512))
        self.linear5 = init_layer(nn.Linear(512, 256))
        self.actor = init_layer(nn.Linear(256, action_space))
        self.critic = init_layer(nn.Linear(256, 1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

        self.train()

    def forward(self, inputs):
        x = self.tanh(self.linear1(inputs))
        x = self.tanh(self.linear2(x))
        x = self.tanh(self.linear3(x))
        x = self.tanh(self.linear4(x))
        x = self.tanh(self.linear5(x))
        policy = self.softmax(self.actor(x))
        value = self.critic(x)

        return policy, value
