import time


class CommandLine(object):
    def __init__(self):
        self.n_features = 1
        self.n_actions = 2

        self.N_STATES = 6   # the length of the 1 dimensional world
        self.FRESH_TIME = 0.5
        self.ACTIONS = ['left', 'right']     # available actions
        self.S = 0
        self.render()

    def reset(self):
        self.S = 0

    def render(self):
        env_list = ['-']*(self.N_STATES-1) + ['T']   # '---------T' our environment
        if self.S == 'terminal':
            time.sleep(2)
            print('\r                                ')
        else:
            env_list[self.S] = 'o'
            interaction = ''.join(env_list)
            print('\r{}'.format(interaction), end='')
            time.sleep(self.FRESH_TIME)

    def step(self, action):
        # This is how agent will interact with the environment
        if action == 1:    # move right
            self.S += 1
            if self.S == self.N_STATES - 1:   # terminate
                R = 100
            else:
                R = 0
        else:   # move left
            R = 0
            if self.S == 0:
                self.S = self.S  # reach the wall
            else:
                self.S -= 1
        self.render()
        return R

    def isDone(self):
        return self.S == self.N_STATES - 1
