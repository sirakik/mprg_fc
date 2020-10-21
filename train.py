# coding: utf-8
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
#gym.logger.set_level(40) # gymのwarningを出力しない
import gfootball
import random



############## Hyperparameters ##############
seed = 0
device = 'cuda:0'
render = False

max_episodes = 1000 # 決め打ち 無限にしたいやつ
max_steps = 3000 # 3000stepで1試合(90分)
update_timestep = 2000 # update_timestepごとにパラメタを更新
#############################################

env = gfootball.env.create_environment(
            env_name='11_vs_11_stochastic',
            stacked=False,
            representation='simple115v2',
            rewards='scoring,checkpoints',
            logdir='./temp_log',
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False,
            write_video=False,
            dump_frequency=0,
            number_of_left_players_agent_controls=1,
            number_of_right_players_agent_controls=0)

state_dim = env.observation_space.shape[0]
action_space = env.action_space.n


class Memory():
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_dones = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_dones[:]


from mlp_model import MLPModel
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_space):
        super(ActorCritic, self).__init__()
        self.model = MLPModel(state_dim, action_space)

    def forward(self):
        raise NotImplementedError

    def get_action(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_porbs, _ = self.model(state)
        dist = Categorical(action_porbs)
        action = dist.sample()
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        return action.item()

    def evaluate(self, state, action):
        action_probs, state_value = self.model(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_space, LR=0.002, BETAS=(0.9, 0.999)):
        self.policy = ActorCritic(state_dim, action_space).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR, betas=BETAS)
        self.policy_old = ActorCritic(state_dim, action_space).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update(self, memory, gamma=0.99, K_epochs=4, eps_clip=0.2):
        rewards = []
        discounted_reward = 0
        for reward, is_done in zip(reversed(memory.rewards), reversed(memory.is_dones)):
            if is_done:
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs
        for _ in range(K_epochs):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy_old.evaluate(old_states, old_actions)

            # Finding the ratio
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0-eps_clip, 1.0+eps_clip) # clamp: (tensor, min, max) tensorをmin, maxでクリップ
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # Loss = L^{CLIP} -c_1 * L^{VF} + c_2 * L^s : 最小化問題なので符号反転
            # L^{CLIP} : 方策 得られる報酬の期待値を最小化している感じ? 方策(行動の確率)とアドバンテージA() の積 policyベース的な考え
            # L^{VF}   : 状態価値 (old状態価値 - targetの状態価値)^2  いわゆるA3CのAdvantage ちょっと先まで考えましょうねぇ
            # L^s      : つまりはランダム性　
            #   方策が決定的であるほど小さく, 全ての行動を選ぶ確率がほぼ等しくなる

            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()


def main():
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    torch.cuda.manual_seed(seed)

    memory = Memory()
    ppo = PPO(state_dim, action_space)

    # logging variables
    reward_last = -1
    reward_sum = 0
    timestep = 0

    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for i_step in range(max_steps):
            timestep += 1

            # running policy old:
            action = ppo.policy_old.get_action(state, memory)
            state, reward, done, _ = env.step(action)

            # saving reward and done
            memory.rewards.append(reward)
            memory.is_dones.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            reward_sum += reward

            if render:
                env.render()

            if done:
                break

        #if reward_last < reward_sum:
        #    torch.save(ppo.policy.state_dict(), )

        print('# Episode: {} | Total reward: {}'.format(i_episode, reward_sum))
        reward_last = reward_sum
        reward_sum = 0


if __name__ == '__main__':
    main()