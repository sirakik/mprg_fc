import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_conv import GraphConv
from .soccer_edge import get_A


def init_layer(m):
    weight = m.weight.data
    weight.normal_(0, 1)
    weight *= 1.0 / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
    nn.init.constant_(m.bias.data, 0)
    return m


class ActorLayer(nn.Module):
    def __init__(self, in_channels, hidden_size, A, s_kernel_size, action_space):
        super(ActorLayer, self).__init__()
        self.gc = GraphConv(in_channels, hidden_size, A, s_kernel_size)
        self.fc = init_layer(nn.Linear(hidden_size, action_space))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.gc(x)
        x = self.relu(x)
        x = F.avg_pool1d(x, x.size()[2:])
        x = x.squeeze()
        x = self.fc(x)
        x = self.softmax(x)
        return x


class CriticLayer(nn.Module):
    def __init__(self, in_channels, hidden_size, A, s_kernel_size):
        super(CriticLayer, self).__init__()
        self.gc = GraphConv(in_channels, hidden_size, A, s_kernel_size)
        self.fc = init_layer(nn.Linear(hidden_size, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.gc(x)
        x = self.relu(x)
        x = F.avg_pool1d(x, x.size()[2:])
        x = x.squeeze()
        x = self.fc(x)
        return x


class GCN(nn.Module):
    def __init__(self, num_inputs, action_space, num_agents, multi_A=False):
        super(GCN, self).__init__()
        A = get_A(multi_A=multi_A)
        A = torch.tensor(A, dtype=torch.float32, requires_grad=False).cuda()
        s_kernel_size = A.size()[0]

        # shared
        self.gc1 = GraphConv(num_inputs, 64, A, s_kernel_size)
        self.gc2 = GraphConv(64, 128, A, s_kernel_size)
        self.gc3 = GraphConv(128, 256, A, s_kernel_size)
        self.gc4 = GraphConv(256, 512, A, s_kernel_size)
        self.gc5 = GraphConv(512, 512, A, s_kernel_size)
        self.gc6 = GraphConv(512, 256, A, s_kernel_size)

        # actor
        actor_layers = []
        for _ in range(num_agents):
            actor_layers.append(nn.Sequential(
                ActorLayer(256, 128, A, s_kernel_size, action_space)
            ))
        self.actor_layers = nn.ModuleList(actor_layers)

        # critic
        critic_layers = []
        for _ in range(num_agents):
            critic_layers.append(nn.Sequential(
                CriticLayer(256, 128, A, s_kernel_size)
            ))
        self.critic_layers = nn.ModuleList(critic_layers)

        # activation
        self.relu = nn.ReLU()

        self.train()

    def forward(self, inputs):
        # shared
        x = self.relu(self.gc1(inputs))
        x = self.relu(self.gc2(x))
        x = self.relu(self.gc3(x))
        x = self.relu(self.gc4(x))
        x = self.relu(self.gc5(x))
        x = self.relu(self.gc6(x))

        # actor
        policies = torch.cat([layer(x).unsqueeze(0) for layer in self.actor_layers], dim=0)

        # critic
        values = torch.cat([layer(x).unsqueeze(0) for layer in self.critic_layers], dim=0)

        return policies, values
