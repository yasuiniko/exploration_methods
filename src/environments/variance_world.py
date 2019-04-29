import numpy as np

from tilecoding import TileCoder


class Environment:
    def __init__(self):
        self.pos = None

        # 0 is left by N(0.05, 0.01), 1 is right by N(0.05, 0.01)
        self.actions = lambda a: np.random.normal(np.power(-1, 1 - a) * 5) / 100
        self.nactions = 2

        self.noisy_reward = lambda: np.random.choice([2, -2, 0.1, -0.1, 1])

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

        noisy = self.pos > 1
        stable = self.pos < 0

        terminal = noisy or stable
        reward = self.noisy_reward() if noisy else 0.02 if stable else 0

        return reward, self.get_phi(self.pos), terminal

    def numactions(self):
        return self.nactions

    def numstates(self):
        # number of cells
        return self.tiler.n_tiles
