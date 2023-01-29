import gymnasium as gym
from stable_baselines import PPO1
from stable_baselines.common.policies import CnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

# environment_name = 'CarRacing-v2'
# # env = gym.make(environment_name, render_mode="human")
# #
# # episodes = 5
# # for episode in range(1, episodes + 1):
# #     state = env.reset()
# #     done = False
# #     score = 0
# #
# #     while not done:
# #         action = env.action_space.sample()
# #         n_state, reward, done, info, _ = env.step(action)
# #         score += reward
# #     print(f'Episode:{episode} Score:{score}')
# # env.close()
#
# fun = gym.make(environment_name, render_mode="human")
# env = DummyVecEnv([lambda: fun])
#
# model = PPO("CnnPolicy", env, verbose=1)
#
# model.learn(total_timesteps=5)

env = gym.make('CarRacing-v2')

model = PPO1(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=5)
model.save("ppo1_cartpole")
