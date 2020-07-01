import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = gym.make('LunarLanderContinuous-v2')

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=2)


del model  # delete trained model to demonstrate loading
# Load the trained agent
model = PPO.load("ppo_lunar")

# # Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(100):
    dones = False
    game_score = 0
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        # import ipdb;ipdb.set_trace()
        game_score+=rewards
        env.render()
    print("game ", i , " game score %.3f"%game_score)
    obs = env.reset()
    # break
