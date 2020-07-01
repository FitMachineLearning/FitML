import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = gym.make('LunarLanderContinuous-v2')

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)
# Train the agent
for i in range(10):
    print("Training itteration ",i)
    model.learn(total_timesteps=10000)
    # Save the agent
    model.save("ppo_lunar")


del model  # delete trained model to demonstrate loading
# Load the trained agent
model = DQN.load("ppo_lunar")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
# obs = env.reset()
# for i in range(100):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
