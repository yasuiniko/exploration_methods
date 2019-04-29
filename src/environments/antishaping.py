import numpy as np
from scipy.stats import norm

from tilecoding import TileCoder


class Environment:
    """
    Note: requires gamma to be set to maintain reward structure.
    """
    def __init__(self, gamma=0.99):
        self.pos = None
        self.terminal_reward = (norm.pdf(0.5, 0.5, 0.15)
                                / (1 - gamma)
                                / np.power(gamma, 100)  # 100 steps to edge
                                * 1.005)

        # 0 is left by N(0.005, 0.001), 1 is right by N(0.005, 0.001)
        self.actions = lambda a: (np.random.normal(np.power(-1, 1 - a) * 5) /
                                  1000)
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
        self.pos = np.random.uniform(0.45, 0.55)
        return self.get_phi(self.pos)

    def env_step(self, action):
        self.pos += self.actions(action)

        terminal = not 0 < self.pos < 1
        reward = norm._pdf((self.pos - 0.5) / 0.15) / 0.15

        # normalize to 1 terminal reward
        reward /= self.terminal_reward
        reward += 1 if terminal else 0

        return reward, self.get_phi(self.pos), terminal

    def numactions(self):
        return self.nactions

    def numstates(self):
        # number of cells
        return self.tiler.n_tiles
