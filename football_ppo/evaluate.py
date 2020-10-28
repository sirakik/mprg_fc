import torch
import gym
import gfootball

from football_ppo.tools.actor_critic import ActorCritic
from football_ppo.tools.utils import make_env, convert_tensor_obs
from multiprocessing_env import DummyVecEnv, VecNormalize


MODEL_NAME = 'mlp'
LOAD_PATH = 'log/gomi/john_due.pt'


# environment
ENV_NAME = '11_vs_11_stochastic'
REPRESENTATION = 'simple115v2'
REWARDS = 'scoring,checkpoints'
LEFT_AGENT = 1
RIGHT_AGENT = 0
env = make_env(ENV_NAME, REPRESENTATION, REWARDS, LEFT_AGENT, RIGHT_AGENT)
env = DummyVecEnv([env])
obs_shape = envs.observation_space.low.shape[0]
action_space = envs.action_space.n
current_obs = torch.zeros(1, *obs_shape)

# model
model = ActorCritic(obs_shape, action_space, MODEL_NAME)
model.load_state_dict(torch.load(LOAD_PATH))

print('# start! :)')
for i in range(3000):
    obs = env.reset()
    current_obs = convert_tensor_obs(obs, current_obs)
    done = False
    episode_reward = 0.0
    while not done:
        with torch.no_grad():
            _, action, _ = model.action(current_obs)

        action = action.unsqueeze(1)

        obs, reward, done, _ = env.step(action.numpy())

        episode_reward += reward

        current_obs = convert_tensor_obs(obs, current_obs)
        env.render()

    print('# Episode reward: {}'.format(episode_reward))
