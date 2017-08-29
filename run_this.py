from maze_env import Maze
from RL_brain import DeepQNetwork
import time


def run_maze():
    step = 0
    for episode in range(100):
        # initial observation
        s = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action, action_value = RL.choose_action(s)

            # RL take action and get next observation and reward
            s_, reward, done = env.step(action)

            print('action: {}, reward:{}, action_value:{}'.format(RL.get_action(action), reward, action_value))

            RL.store_transition(s, action, reward, s_)

            # swap observation
            s = s_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features, memory_size=1)
    env.after(200, run_maze)
    env.mainloop()
