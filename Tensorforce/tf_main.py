from tensorforce import Agent, Environment
from tensorforce.agents import PPOAgent
from tensorforce.environments import OpenAIGym

# Pre-defined or custom environment
# environment = Environment.create(
#     environment='gym', level='CartPole', max_episode_timesteps=500
# )

# Network as list of layers
network_spec = [
    dict(type='dense', size=32, activation='tanh'),
    dict(type='dense', size=32, activation='tanh')
]

environment = OpenAIGym('CartPole-v0', visualize=True, max_episode_steps=500)


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
    agent='ppo', environment=environment, batch_size=10, learning_rate=1e-3
)

# agent = PPOAgent(
#     states_spec=environment.states,
#     actions_spec=environment.actions,
#     network_spec=network_spec,
#     batch_size=4096,
#     # BatchAgent
#     keep_last_timestep=True,
#     # PPOAgent
#     step_optimizer=dict(
#         type='adam',
#         learning_rate=1e-3
#     ),
#     optimization_steps=10,
#     # Model
#     scope='ppo',
#     discount=0.99,
#     # DistributionModel
#     distributions_spec=None,
#     entropy_regularization=0.01,
#     # PGModel
#     baseline_mode=None,
#     baseline=None,
#     baseline_optimizer=None,
#     gae_lambda=None,
#     # PGLRModel
#     likelihood_ratio_clipping=0.2,
#     # summary_spec=None,
#     # distributed_spec=None
# )

# Train for 300 episodes
for _ in range(300):

    # Initialize episode
    states = environment.reset()
    terminal = False

    while not terminal:
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

agent.close()
environment.close()
