import numpy as np
from environments.hypercube import Environment as Hypercube


class Environment(Hypercube):
    def __init__(self):
        super().__init__()

    def env_start(self):
        self.pos = np.zeros(self.dim)
        # print("env_start", np.array(self.pos))
        return np.array(self.pos)

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
                                       max(-self.radius, self.pos[action_dim]))

        num_walls = (np.abs(self.pos) == self.radius).sum()
        terminal = num_walls == self.dim
        reward = self.rewards[num_walls]

        # print("env_step", np.array(self.pos))
        return reward, np.array(self.pos), terminal

    def numstates(self):
        return self.dim
