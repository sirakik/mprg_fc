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
    def __init__(self, num_inputs, action_space, num_agents):
        super(MLP, self).__init__()
        # shared
        self.linear1 = init_layer(nn.Linear(num_inputs, 256))
        self.linear2 = init_layer(nn.Linear(256, 512))
        self.linear3 = init_layer(nn.Linear(512, 512))
        self.linear4 = init_layer(nn.Linear(512, 256))

        # actor
        actor_layers = []
        for _ in range(num_agents):
            actor_layers.append(nn.Sequential(
                init_layer(nn.Linear(256, 128)),
                nn.Tanh(),
                init_layer(nn.Linear(128, action_space))
            ))
        self.actor_layers = nn.ModuleList(actor_layers)

        # critic
        critic_layers = []
        for _ in range(num_agents):
            critic_layers.append(nn.Sequential(
                init_layer(nn.Linear(256, 128)),
                nn.Tanh(),
                init_layer(nn.Linear(128, 1))
            ))
        self.critic_layers = nn.ModuleList(critic_layers)

        # activation
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

        self.train()

    def forward(self, inputs):
        # shared
        x = self.tanh(self.linear1(inputs))
        x = self.tanh(self.linear2(x))
        x = self.tanh(self.linear3(x))
        x = self.tanh(self.linear4(x))

        # actor
        policies = torch.cat([self.softmax(layer(x)).unsqueeze(0) for layer in self.actor_layers], dim=0)

        # critic
        values = torch.cat([layer(x).unsqueeze(0) for layer in self.critic_layers], dim=0)

        return policies, values
