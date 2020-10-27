import torch
from torch.utils.data.sample import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, per_steps, num_processes, obs_shape, action_space, start_obs):
        self.observations = torch.zeros(per_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(per_steps, num_processes, 1)
        self.value_preds = torch.zeros(per_steps + 1, num_processes, 1)
        self.returns = torch.zeros(per_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(per_steps, num_processes, 1)
        self.actions = torch.zeros(per_steps, num_processes, action_space)
        self.masks = torch.ones(per_steps + 1, num_processes, 1)

        self.per_steps = per_steps
        self.step = 0
        self.observations[0].copy_(start_obs)

    def cuda(self, device):
        self.observations = self.observations.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, current_obs, action, action_log_prob, value_pred, reward, mask):
        self.observations[self.step + 1].copy_(current_obs)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.per_steps

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    # Generalized advantage estimator
    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        # Subtraction of value_{s_t} is not done here
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.rewards[step] + self.masks[step + 1] * gamma * self.returns[step + 1]

    def sample(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch

        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            observations_batch = self.observations[:-1].view(-1, *self.observations.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv = advantages.view(-1, 1)[indices]

            yield observations_batch, actions_batch, return_batch, masks_batch, old_action_log_probs_batch, adv
