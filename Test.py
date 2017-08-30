from command_line import *
from RL_brain import *


RL = RL = DeepQNetwork(2, 1, memory_size=1)
env = CommandLine()

for episode in range(100):
    count = 0
    env.reset()
    for i in range(6):
        i = np.array([i])[np.newaxis, :]
        actions_value = RL.sess.run(RL.q_eval, feed_dict={RL.s: i})
        print(i, ' : \n', actions_value)
    while True:
        S = [env.S]
        action = RL.choose_action(S)
        R = env.step(action)

        RL.store_transition(S, action, R, env.S)

        count += 1
        if env.isDone():
            for i in range(6):
                i = np.array([i])[np.newaxis, :]
                actions_value = RL.sess.run(RL.q_eval, feed_dict={RL.s: i})
                print(i, ' : \n', actions_value)
            print('\ngame over: episode{} count{}'.format(episode, count))
            break





