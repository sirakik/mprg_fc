import matplotlib.pyplot as plt
from matplotlib import animation
import cv2
import datetime
import numpy as np

import torch
import gym
import gfootball


from tools.actor_critic import ActorCritic
from tools.utils import make_env, convert_tensor_obs

def modify_obs(obs):
    obs = obs[0] # (agents, 115) -> (115)
    obs[97:108] = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    return obs


MODEL_NAME = 'mlp'

OUTPUT_DIR = 'log/gomi'
LOAD_PATH = 'log/gomi/model_john.pt'

# environment
ENV_NAME = '11_vs_11_stochastic'
REPRESENTATION = 'simple115v2'
REWARDS = 'scoring,checkpoints'
LEFT_AGENT = 11
RIGHT_AGENT = 0
env = make_env(ENV_NAME, REPRESENTATION, REWARDS, LEFT_AGENT, RIGHT_AGENT, iprocess=1)
obs_shape = env.observation_space.shape[1]
action_space = env.action_space.nvec[0]
current_obs = torch.zeros(1, obs_shape)

# model
model = ActorCritic(obs_shape, action_space, MODEL_NAME, LEFT_AGENT)
model.load_state_dict(torch.load(LOAD_PATH))

img = []
episode_reward = 0.0

print('# start! :)')
obs = env.reset()
obs = modify_obs(obs)
current_obs = convert_tensor_obs(obs, current_obs)
for i in range(3000):
    with torch.no_grad():
        _, action, _ = model.action(current_obs)

    action = action.transpose(1, 0).squeeze()

    obs, reward, _, _ = env.step(action.numpy())

    episode_reward += reward.mean()

    obs = modify_obs(obs)  # add
    current_obs = convert_tensor_obs(obs, current_obs)

    # render
    bgr = env.render('rgb_array')
    img.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) # BGR -> RGB

print('# Episode reward: {}'.format(episode_reward))


print('# saving video. . .')
dpi = 72
interval = 50 # ms
plt.figure(figsize=(img[0].shape[1]/dpi,img[0].shape[0]/dpi),dpi=dpi)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
patch = plt.imshow(img[0])
plt.axis=('off')
animate = lambda i: patch.set_data(img[i])
ani = animation.FuncAnimation(plt.gcf(),animate,frames=len(img),interval=interval)
now = datetime.datetime.now()
filename = OUTPUT_DIR + '/' + now.strftime('%Y%m%d_%H%M%S') + '.mp4'
ani.save(filename, writer="ffmpeg")
plt.close()
print('# done. output: -> [{}]'.format(filename))

env.close()