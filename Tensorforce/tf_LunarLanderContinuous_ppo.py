from tensorforce import Agent, Environment
from tensorforce.agents import PPOAgent
from tensorforce.environments import OpenAIGym

# Pre-defined or custom environment
# environment = Environment.create(
#     environment='gym', level='CartPole', max_episode_timesteps=500
# )

# Network as list of layers
network_spec = [
    # dict(type='dense', size=32, activation='tanh'),
    dict(type='dense', size=128, activation='tanh')
]

# environment = OpenAIGym('CartPole-v0', visualize=True, max_episode_steps=500)
environment = OpenAIGym('LunarLanderContinuous-v2', visualize=True, max_episode_steps=500)
# environment = OpenAIGym('BipedalWalker-v3', visualize=False, max_episode_steps=500)


# Instantiate a Tensorforce agent
# agent = Agent.create(
#     agent='tensorforce',
#     environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
#     memory=10000,
#     update=dict(unit='timesteps', batch_size=64),
#     optimizer=dict(type='adam', learning_rate=3e-4),
#     policy=dict(network='auto'),
#     objective='policy_gradient',
#     reward_estimation=dict(horizon=20)
# )

agent = Agent.create(
    agent='ppo', environment=environment, batch_size=10,
    network=[
        dict(type='dense', size=64),
        dict(type='dense', size=64)
    ],
    learning_rate=1e-3

)

# # Instantiate a Tensorforce agent
# agent = PPOAgent(
#     environment=environment,
#     network=[
#         dict(type='dense', size=64),
#         dict(type='dense', size=64)
#     ],
#     learning_rate=1e-3
# )


running_score = 0.0
# Train for 300 episodes
for i_epoch in range(50000):
    game_score = 0.0
    # Initialize episode
    states = environment.reset()
    terminal = False

    while not terminal:
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        game_score+=reward
        agent.observe(terminal=terminal, reward=reward)

    running_score = 0.95*running_score + 0.05*game_score
    if i_epoch%5==0:
        print("Game ", i_epoch, "       game score %.2f"%game_score,"       running score %.2f"%running_score)


agent.close()
environment.close()
