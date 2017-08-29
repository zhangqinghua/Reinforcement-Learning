from command_line import *
from RL_brain import *


RL = RL = DeepQNetwork(2, 1, memory_size=1)
env = CommandLine()

while True:
    action = RL.choose_action([env.S])
    S, R = env.step(action)
    if env.isDone():
        break





