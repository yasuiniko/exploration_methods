import numpy as np

from tilecoding import TileCoder


class Environment:
    def __init__(self):
        self.pos = None

        # 0 is right, 1 is left
        mean0, sd0 = -1 / 5, 0.1 / 5
        self.actions = lambda a: (np.random.normal(mean0, sd0)
                                  if a else
                                  (np.random.gamma(342.8, 0.01) - 3.378) / 1.6)
        self.nactions = 2

        # tilecoding stuff
        self.num_tiles = np.asarray([2])
        self.num_tilings = 32
        self.limits = [[0, 1]]
        self.tiler = TileCoder(dims=self.num_tiles,
                               limits=self.limits,
                               tilings=self.num_tilings)

    def env_init(self):
        self.tiler = TileCoder(dims=self.num_tiles,
                               limits=self.limits,
                               tilings=self.num_tilings)

    def get_phi(self, state):
        return self.tiler[state]

    def env_start(self):
        self.pos = np.random.uniform(0.2, 0.4)
        return self.get_phi(self.pos)

    def env_step(self, action):
        # move the agent
        self.pos += self.actions(action)

        # calculate rewards
        reward = 1 if self.pos > 1 else 0.0005 if self.pos < 0 else 0
        # reward = self.pos if self.pos > 0.5 else 0 if self.pos > 0 else 0.0005
        # reward = self.pos if self.pos > 0 else 0.0005

        # move agent back in bounds
        if not 0 < self.pos < 1:
            self.pos = min(1, max(0, self.pos))

        terminal = False

        return reward, self.get_phi(self.pos), terminal

    def numactions(self):
        return self.nactions

    def numstates(self):
        # number of cells
        return self.tiler.n_tiles
