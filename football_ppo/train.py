# coding: utf-8
# Refer to https://github.com/ASzot/ppo-pytorch

import os
import csv
import copy
import shutil
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functionla as F

import gym
import gfootball

from memory import RolloutStorage
from actor_critic import ActorCritic
from multiprocessing_env import SubprocVecEnv, VecNormalize
from utils import make_env, convert_tensor_obs


############## Hyperparameters ##############
OUTPUT_DIR = 'log/gomi'

NUM_ENVS = 16
NUM_STEPS = 50000000  # Number of steps
PER_STEPS = 512  # Update parameters per steps

GAMMA = 0.99  # Discount rate for reward
CLIP_PARAM = 0.2  # Hyperparameter of the PPO
MAX_GRAD_NORM = 0.5  # Max gradient norm (clipping)
NUM_EPOCHS = 8  # Number of update epochs
N_MINI_BATCH = 8  # Number of minibatches to split one epoch to

# model
MODEL_NAME = 'MLP'
# optimizer
LR = 0.0001
EPS = 1e-5
# device
DEVICE = '0'
# environment
ENV_NAME = '11_vs_11_stochastic'
REPRESENTATION = 'simple115v2'
REWARDS = 'scoring,checkpoints'
LEFT_AGENT = 1
RIGHT_AGENT = 0
#############################################


def update_parames(rollouts, model, optimizer):
    # advantage = (R_T + V_{s_{t+1}) - V_{s_t}
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

    policy_losses = []
    value_losses = []
    entropy_losses = []
    losses = []
    for _ in range(NUM_EPOCHS):
        samples = rollouts.sample(advantages, N_MINI_BATCH)

        for obs, actions, returns, masks, old_action_log_probs, adv_targ in samples:
            values, action_log_probs, dist_entropy = model.evaluate_actions(obs, actions)

            # Policy loss
            ratio = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * adv_targ
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(returns, values)

            # Losses
            loss = policy_loss + 0.5 * value_loss - 0.01 * dist_entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(dist_entropy.item())
            losses.append(loss.item())

    return np.mean(policy_losses), np.mean(value_losses), np.mean(entropy_losses), np.mean(losses)


# output dir
if not os.path.existis(OUTPUT_DIR):
    os.makedri(OUTPUT_DIR)
    with open(OUTPUT_DIR + '/log.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['num_updates', 'all_loss', 'policy_loss', 'value_loss', 'entropy_loss', 'mean_reward'])
else:
    print('# caution: output dir [{}] already exist.'.foramt(OUTPUT_DIR))
    answer = input('# continue? [y/n]: ')
    if answer == 'y':
        pass
    else:
        print('# bye :)')
        exit()


# parallelize environment
envs = [make_env(ENV_NAME, REPRESENTATION, REWARDS, LEFT_AGENT, RIGHT_AGENT) for i in range(NUM_ENVS)]
envs = SubprocVecEnv(envs)
envs = VecNormalize(envs, gamma=GAMMA)
obs_shape = envs.observation_space.low.shape[0]
action_space = envs.action_space.n
current_obs = torch.zeros(NUM_ENVS, *obs_shape)
obs = envs.reset()
current_obs = convert_tensor_obs(obs, current_obs)
print('\n')
print('# Environment        : {}'.format(ENV_NAME))
print('# Representation     : {}'.foramt(REPRESENTATION))
print('# Rewards            : {}'.foramt(REWARDS))
print('# Observation shape  : {}'.format(obs_shape))
print('# Action space       : {}'.format(action_space))
print('# Num. of left agent : {}'.format(LEFT_AGENT))
print('# Num. of right agent: {}'.format(RIGHT_AGENT))


# initialize rollouts
rollouts = RolloutStorage(PER_STEPS, NUM_ENVS, obs_shape, action_space, current_obs)


# load model
print('\n')
print('# Load model: {}'.format(MODEL_NAME))
model = ActorCritic(obs_shape, action_space, MODEL_NAME)


# optimizer
print('\n')
print('# AdamOptimizer')
print('# Learning rage: {}'.foramt(LR))
optimizer = optimz.Adam(model.parameters(), lr=LR, eps=EPS)


# Device
print('\n')
print('# Device: {}'.format(DEVICE))
if CUDA:
    model.to(DEVICE)
    rollouts.cuda(DEVICE)
    current_obs.to(DEVICE)


# logging variables
episode_rewards = torch.zeros([NUM_ENVS, 1])
final_rewards = torch.zeros([NUM_ENVS, 1])
max_reward = 0


print('\n')
print('# Start! :)')
num_updates = int(NUM_STEPS // PER_STEPS // NUM_ENVS)
for update_i in range(num_updates):
    for step in range(PER_STEPS):
        with torch.no_grad():
            value, action, action_log_prob = model.action(rollouts.observations[step])

        # step
        obs, reward, done, info = envs.step(actions.squeeze(1).cpu().numpy())

        # convert to pytorch tensor
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])

        # update reward info for logging
        episode_rewards += reward
        final_rewards *= masks
        final_rewards += (1 - masks) * episode_rewards
        episode_rewards *= masks

        # Update current observation tensor
        current_obs *= masks
        current_obs = convert_tensor_obs(obs, current_obs)

        rollouts.insert(current_obs, action, action_log_prob, value, reward, masks)

    with torch.no_grad():
        next_value = policy.get_value(rollouts.observations[-1]).detach()

    # Generalized advantage estimator
    rollouts.compute_returns(next_value, GAMMA)

    # update params
    policy_loss, value_loss, entropy_loss, all_loss = update_params(rollouts, policy, optimizer)

    rollouts.after_update()

    mean_reward = final_rewards.mean()
    print('# policy loss: {:.3f} | value loss: {:.3f} | mean reward: {:.3f}'.format(
        policy_loss, value_loss, mean_reward))

    # logging csv
    with open(log_dir + '/log.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([update_i, all_loss, policy_loss, value_loss, current_obs, mean_reward])

    if update_i % SAVE_INTERVAL == 0:
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_%i.pt' % update_i))

    if max_reward < mean_reward:
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_max_rewards.pt'))
        max_reward = mean_reward

print('# Report: max reward: {}'.foramt(max_reward))
print('# bye :)')
