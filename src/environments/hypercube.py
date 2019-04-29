import numpy as np

from tilecoding import TileCoder


class Environment:
    """
    Note: requires gamma to be set to maintain reward structure.
    """
    def __init__(self, n=5, gamma=0.99):
        self.pos = None
        self.dim = n
        self.radius = 10

        # 0 is left, or N(-1, 0.15), 1 is right, or N(+1, 0.15)
        self.actions = lambda a: np.random.normal(np.power(-1, 1 - a), 0.15)
        self.nactions = 2 * n

        # reward code
        self.rewards = np.zeros(self.dim + 1)
        self.rewards[-1] = 1
        for i in reversed(range(1, self.dim)):
            self.rewards[i] = self.rewards[i + 1] * (1 - gamma) * 0.9
        # make average reward 100x higher when agent chooses to receive final
        # terminating reward
        self.rewards[-1] = (111 * self.rewards[-2] * self.radius * self.dim
                            - (self.rewards[1:self.dim] * 10).sum())

        # tilecoding stuff
        self.num_tiles = np.asarray([2] * n)
        self.num_tilings = 32
        self.limits = [[-self.radius, self.radius]] * n

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
        self.pos = np.zeros(self.dim)
        return self.get_phi(self.pos)

    def env_step(self, action):
        # make the action vector from the integer action selected
        action_dim = action // 2
        action_dir = action % 2
        action_vector = np.zeros(self.dim)
        action_vector[action_dim] = self.actions(action_dir)

        # move the agent and make sure it stays within bounds
        self.pos += action_vector
        if abs(self.pos[action_dim]) > self.radius:
            self.pos[action_dim] = min(self.radius,
                                       max(-self.radius,
                                           self.pos[action_dim]))

        num_walls = (np.abs(self.pos) == self.radius).sum()
        terminal = num_walls == self.dim
        reward = self.rewards[num_walls]

        return reward, self.get_phi(self.pos), terminal

    def numactions(self):
        return self.nactions

    def numstates(self):
        # number of cells
        return self.tiler.n_tiles
