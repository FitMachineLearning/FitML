import gym
import pybullet, pybullet_envs
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
# env = gym.make('LunarLanderContinuous-v2')

env = gym.make('BipedalWalker-v3')
# env.render(mode="human")

policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=[512, 512])
# Instantiate the agent
model = PPO('MlpPolicy', env,learning_rate=0.0003,policy_kwargs=policy_kwargs, verbose=1)
# Train the agent
for i in range(8000):
    print("Training itteration ",i)
    model.learn(total_timesteps=10000)
    # Save the agent
    model.save("ppo_Ant")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
    print("mean_reward ", mean_reward)
    if mean_reward >= 270:
        print("***Agent Trained with average reward ", mean_reward)
        break

del model  # delete trained model to demonstrate loading
# Load the trained agent
# model = PPO.load("ppo_Ant")

# Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
# obs = env.reset()
# for i in range(100):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
