import numpy as np
from environments.riverswim import Environment as Riverswim


class Environment(Riverswim):
    def __init__(self):
        super().__init__()

    def env_start(self):
        self.pos = np.random.uniform(0.2, 0.4)
        return np.array([self.pos])

    def env_step(self, action):
        # move the agent
        self.pos += self.actions(action)

        # calculate rewards
        reward = 10000 if self.pos > 1 else 5 if self.pos < 0 else 0

        # move agent back in bounds
        if not 0 < self.pos < 1:
            self.pos = min(1, max(0, self.pos))

        terminal = False

        return reward, np.array([self.pos]), terminal

    def numstates(self):
        return 1
