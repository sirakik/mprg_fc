import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_space, model_name):
        super().__init__()

        if model_name == 'mlp':
            from models.mlp import MLP as Model
        else:
            raise ValueError('# Not expected model [{}].'.format(model_name))

        self.actor_critic = Modl(obs_shape)
        self.action_mean = init_layer(nn.Linear(128, action_space))
        self.action_log_std = nn.Parameter(torch.zeros(1, action_space))

    def _get_dist(self, actor_feature):
        action_mean = self.action_mean(actor_feature)
        action_log_std = self.action_log_std.expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_log_std.exp())

    def action(self, inputs, deterministic=False):
        actor_features, value = self.actor_critic(inputs)
        dist = self._get_dist(actor_features)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        action_log_probs = dist.log_prob(action).sum(-1, keepdim==True)

        return value, action, action_log_probs

    def get_value(self, inputs):
        _, value = self.actor_critic(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        actor_feature, value = self.actor_critic(inputs)
        dist = self._get_dist(actor_feature)

        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()

        return value, action_log_probs, dist_entropy
