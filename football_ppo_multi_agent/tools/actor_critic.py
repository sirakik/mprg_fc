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
    def __init__(self, obs_shape, action_space, model_name, num_agents):
        super(ActorCritic, self).__init__()

        if model_name == 'mlp':
            from tools.models.mlp import MLP as Model
        else:
            raise ValueError('# Error   :Not expected model [{}].'.format(model_name))

        self.actor_critic = Model(obs_shape, action_space, num_agents)

    def action(self, inputs):
        policies, value = self.actor_critic(inputs)
        dists = Categorical(policies)
        actions = dists.sample()
        action_log_probs = dists.log_prob(actions)

        return value, actions, action_log_probs

    def get_value(self, inputs):
        _, value = self.actor_critic(inputs)
        return value

    def evaluate_actions(self, inputs, actions):
        policies, value = self.actor_critic(inputs)
        dists = [Categorical(policy) for policy in policies]
        actions = actions.transpose(1, 0).unsqueeze(-1)
        action_log_probs = [e[0].log_prob(e[1]) for e in zip(dists, actions)]
        action_log_probs = torch.stack(action_log_probs)  # tensor in list -> tensors
        dist_entropies = [dist.entropy() for dist in dists]
        dist_entropy = torch.stack(dist_entropies).mean()

        return value, action_log_probs, dist_entropy
