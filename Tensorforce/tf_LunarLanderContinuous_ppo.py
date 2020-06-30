from tensorforce import Agent, Environment
from tensorforce.agents import PPOAgent
from tensorforce.environments import OpenAIGym

# Pre-defined or custom environment
# environment = Environment.create(
#     environment='gym', level='CartPole', max_episode_timesteps=500
# )


# environment = OpenAIGym('CartPole-v0', visualize=True, max_episode_steps=500)
environment = OpenAIGym('LunarLanderContinuous-v2', visualize=False, max_episode_steps=500)
# environment = OpenAIGym('BipedalWalker-v3', visualize=False, max_episode_steps=500)


agent = Agent.create(
    agent='ppo', environment=environment, batch_size=10,
    network=[
        dict(type='dense', size=64),
        dict(type='dense', size=64)
    ],
    learning_rate=1e-3

)


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

    if i_epoch%10==0 and i_epoch>20:
        agent.save()
    if running_score >= 250:
        agent.save()
        break()

agent.close()
environment.close()
