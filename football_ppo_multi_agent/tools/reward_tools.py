import numpy as np


def ceil(a, precision=0):
    return np.round(a + 0.5 * 10 ** (-precision), precision)


def floor(a, precision=0):
    return np.round(a - 0.5 * 10 ** (-precision), precision)


def my_reward_ball(reward, ball_x):
    for i in range(len(reward)):
        reward[i] *= 100  # 100 points for a goal.
        if ball_x[i] > 1.0:
            r = 1.0
        elif ball_x[i] < -1.0:
            r = -1.0
        elif ball_x[i] >= 0:
            r = ceil(ball_x[i], precision=1)
        else:
            r = floor(ball_x[i], precision=1)

        reward[i] += r

    return reward


def my_reward_player(reward, obs):
    if obs['ball_owned_team'] == 0:  # 0 no toki zibun

        ball = obs['ball'][0:2]
        min_dis = 100
        for idx, player in enumerate(obs['left_team']):
            dis = np.linalg.norm(player - ball)
            if min_dis >= dis:
                min_dis = dis
                active = idx

        x = obs['left_team'][active][0]
        if x >= 0:
            r = ceil(x, precision=1)
        else:
            r = floor(x, precision=1)

        reward[active] += r
        print(reward)
        exit()

    return reward


if __name__ == '__main__':
    reward = my_reward_player(reward, env.unwrapped.observation()[0])
    reward = my_reward_ball(reward, state[88])