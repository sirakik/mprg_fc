import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def init_layer(m):
    weight = m.weight.data
    weight.normal_(0, 1)
    weight *= 1.0 / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
    nn.init.constant_(m.bias.data, 0)
    return m


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_space, model_name):
        super(ActorCritic, self).__init__()

        if model_name == 'mlp':
            from tools.models.mlp import MLP as Model
        else:
            raise ValueError('# Error   :Not expected model [{}].'.format(model_name))

        self.actor_critic = Model(obs_shape, action_space)

    def action(self, inputs):
        policy, value = self.actor_critic(inputs)
        dist = Categorical(action_policy)
        action = dist.sample()
        action_log_probs = dist.log_prob(action)

        return value, action, action_log_probs

    def get_value(self, inputs):
        _, value = self.actor_critic(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        policy, value = self.actor_critic(inputs)
        dist = Categorical(policy)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy
