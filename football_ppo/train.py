import os
import csv
import copy
import shutil
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
gym.logger.set_level(40)
import gfootball

from tools.memory import RolloutStorage
from tools.actor_critic import ActorCritic
from tools.utils import make_env, convert_tensor_obs, print_log
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


############## Hyperparameters ##############
OUTPUT_DIR = 'log/gomi'

NUM_ENVS = 32
NUM_STEPS = 500000000  # Number of steps
PER_STEPS = 512  # Update parameters per steps

GAMMA = 0.99  # Discount rate for reward
CLIP_PARAM = 0.2  # Hyperparameter of the PPO
MAX_GRAD_NORM = 0.5  # Max gradient norm (clipping)
NUM_EPOCHS = 2  # Number of update epochs
N_MINI_BATCH = 8  # Number of minibatches to split one epoch to

SAVE_INTERVAL = 500 # Save per params update
PRINT_INTERVAL = 10
# model
MODEL_NAME = 'mlp'
# optimizer
LR = 0.0001
EPS = 1e-5
# device
DEVICE = 'cuda:0'
# environment
ENV_NAME = '11_vs_11_stochastic'
REPRESENTATION = 'simple115v2'
REWARDS = 'scoring,checkpoints'
LEFT_AGENT = 1
RIGHT_AGENT = 0
#############################################


def update_params(rollouts, model, optimizer):
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

def main():
    # output dir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print('# Report  : make dir -> [{}]'.format(OUTPUT_DIR))
    else:
        print('# Caution : output dir [{}] already exist.'.format(OUTPUT_DIR))
        answer = input('# Asking  : continue? [y/n]: ')
        if answer == 'y':
            pass
        else:
            print('# John Doe: bye.')
            exit()

    # make csv file
    with open(OUTPUT_DIR + '/loss.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['time', 'num_updates', 'all_loss', 'policy_loss', 'value_loss', 'entropy_loss'])
    print('# Report  : make csv file -> [{}]'.format(OUTPUT_DIR + '/loss.csv'))
    with open(OUTPUT_DIR + '/reward.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['reward'])
    print('# Report  : make csv file -> [{}]'.format(OUTPUT_DIR + '/reward.csv'))

    print('\n# John Doe: Prepare to train...')
    # parallelize environment
    envs = [(lambda _i=i: make_env(ENV_NAME, REPRESENTATION, REWARDS, LEFT_AGENT, RIGHT_AGENT, _i)) for i in range(NUM_ENVS)]
    envs = SubprocVecEnv(envs, context=None)
    obs_shape = envs.observation_space.shape[0]
    action_space = envs.action_space.n
    current_obs = torch.zeros(NUM_ENVS, obs_shape)
    obs = envs.reset()
    current_obs = convert_tensor_obs(obs, current_obs)

    # initialize rollouts
    rollouts = RolloutStorage(PER_STEPS, NUM_ENVS, obs_shape, current_obs)

    # load model
    model = ActorCritic(obs_shape, action_space, MODEL_NAME)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=EPS)

    # Device
    if DEVICE is not None:
        model.to(DEVICE)
        rollouts.cuda(DEVICE)
        current_obs.to(DEVICE)

    # logging variables
    episode_rewards = torch.zeros([NUM_ENVS, 1])
    final_rewards = torch.zeros([NUM_ENVS, 1])
    max_reward = -100
    num_updates = int(NUM_STEPS // PER_STEPS // NUM_ENVS)

    print_log('### Output dir: {}'.format(OUTPUT_DIR), OUTPUT_DIR, mode='w')
    print_log('### Football', OUTPUT_DIR)
    print_log('# Environment        : {}'.format(ENV_NAME), OUTPUT_DIR)
    print_log('# Num. of envs       : {}'.format(NUM_ENVS), OUTPUT_DIR)
    print_log('# Representation     : {}'.format(REPRESENTATION), OUTPUT_DIR)
    print_log('# Rewards            : {}'.format(REWARDS), OUTPUT_DIR)
    print_log('# Observation shape  : {}'.format(obs_shape), OUTPUT_DIR)
    print_log('# Action space       : {}'.format(action_space), OUTPUT_DIR)
    print_log('# Num. of left agent : {}'.format(LEFT_AGENT), OUTPUT_DIR)
    print_log('# Num. of right agent: {}'.format(RIGHT_AGENT), OUTPUT_DIR)
    print_log('### Model', OUTPUT_DIR)
    print_log('# Base model   : {}'.format(MODEL_NAME), OUTPUT_DIR)
    print_log('# Learning rage: {}'.format(LR), OUTPUT_DIR)
    print_log('# Device       : {}'.format(DEVICE), OUTPUT_DIR)
    print_log('### Proximal Policy Optimization', OUTPUT_DIR)
    print_log('# Num. of steps       : {}'.format(NUM_STEPS), OUTPUT_DIR)
    print_log('# Update per steps    : {}'.format(PER_STEPS), OUTPUT_DIR)
    print_log('# Num. of updates     : {}'.format(num_updates), OUTPUT_DIR)
    print_log('# Dis. rate for reward: {}'.format(GAMMA), OUTPUT_DIR)
    print_log('# Clip param          : {}'.format(CLIP_PARAM), OUTPUT_DIR)
    print_log('# Max gradient norm   : {}'.format(MAX_GRAD_NORM), OUTPUT_DIR)
    print_log('# Num. of epochs      : {}'.format(NUM_EPOCHS), OUTPUT_DIR)
    print_log('# Batch size          : {}'.format(N_MINI_BATCH), OUTPUT_DIR)

    print('\n# John Doe: Start!')

    for update_i in range(num_updates):
        for step in range(PER_STEPS):
            with torch.no_grad():
                value, action, action_log_prob = model.action(rollouts.observations[step])
            action = action.unsqueeze(1)
            action_log_prob = action_log_prob.unsqueeze(1)

            # step
            obs, reward, done, info = envs.step(action.cpu().numpy())

            # convert to pytorch tensor
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])

            # update reward info for logging
            episode_rewards += reward
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            if True in done:
                with open(OUTPUT_DIR + '/reward.csv', 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([episode_rewards.mean().item()])
            episode_rewards *= masks

            # Update current observation tensor
            current_obs *= masks
            current_obs = convert_tensor_obs(obs, current_obs)

            rollouts.insert(current_obs, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = model.get_value(rollouts.observations[-1]).detach()

        # Generalized advantage estimator
        rollouts.compute_returns(next_value, GAMMA)

        # update params
        policy_loss, value_loss, entropy_loss, all_loss = update_params(rollouts, model, optimizer)

        rollouts.after_update()

        mean_reward = final_rewards.mean().item()

        # logging csv
        with open(OUTPUT_DIR + '/loss.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.datetime.now(), update_i, all_loss, policy_loss, value_loss, entropy_loss])

        if update_i % PRINT_INTERVAL == 0:
            print_log('# Log     : update: [{}/{}] | policy loss: {:.7f} | value loss: {:.7f} | mean reward: {:.3f}'.format(
                update_i, num_updates, policy_loss, value_loss, mean_reward), OUTPUT_DIR)

        if update_i % SAVE_INTERVAL == 0:
            print('# Report  : Save model -> [{}]'.format(OUTPUT_DIR + '/model_%i.pt' % update_i))
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_%i.pt' % update_i))

        if max_reward < mean_reward:
            print('# Report  : Updated max reward.')
            print('# Report  : Save model -> [{}]'.format(OUTPUT_DIR + '/model_max_reward.pt'))
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_max_reward.pt'))
            max_reward = mean_reward

    print('# Report  : max reward: {}'.format(max_reward))
    print('# John Doe: bye.')


if __name__ == "__main__":
    print('\n# John Doe: Hello.')

    main()
