import gym
import gfootball
import torch


def make_env(ENV_NAME, REPRESENTATION, REWARDS, LEFT_AGENT, RIGHT_AGENT, iprocess):
    env = gfootball.env.create_environment(
        env_name=ENV_NAME,
        stacked=False,
        representation=REPRESENTATION,
        rewards=REWARDS,
        logdir='',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        write_video=False,
        dump_frequency=0,
        number_of_left_players_agent_controls=LEFT_AGENT,
        number_of_right_players_agent_controls=RIGHT_AGENT)

    return env


def convert_tensor_obs(obs, current_obs):
    obs = torch.from_numpy(obs).float()
    current_obs[:, :] = obs

    return current_obs


def print_log(text, output_dir, is_print=True, is_log=True, mode='a'):
    if is_print:
        print(text)
    if is_log:
        with open('{}/log.txt'.format(output_dir), mode) as f:
            f.write(text + '\n')
