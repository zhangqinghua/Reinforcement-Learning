from command_line import *
from RL_brain import *


RL = RL = DeepQNetwork(2, 1, memory_size=1)
env = CommandLine()

while True:
    S = [env.S]
    action = RL.choose_action(S)
    R = env.step(action)

    RL.store_transition(S, action, R, env.S)

    if env.isDone():
        break





