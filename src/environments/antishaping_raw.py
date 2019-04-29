import numpy as np
from scipy.stats import norm
from environments.antishaping import Environment as Antishaping


class Environment(Antishaping):
    def __init__(self):
        super().__init__()

    def env_start(self):
        self.pos = np.random.uniform(0.45, 0.55)
        return np.array([self.pos])

    def env_step(self, action):
        self.pos += self.actions(action)

        terminal = not 0 < self.pos < 1
        reward = norm.pdf(self.pos, 0.5, 0.15)
        reward += self.terminal_reward if terminal else 0

        return reward / 10000, np.array([self.pos]), terminal

    def numstates(self):
        return 1
