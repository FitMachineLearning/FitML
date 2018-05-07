

import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE



game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
#agent = myAgentHere(allowed_actions=p.getActionSet())
print("getActionSet -- ",p.getActionSet())


p.init()
reward = 0.0

for i in range(1000):
   if p.game_over():
           p.reset_game()

   observation = p.getScreenRGB()
  # action = agent.pickAction(reward, observation)
   action = 118 + np.random.randint(2)
   print("action",action)
   reward = p.act(action)
   print("r",reward)
