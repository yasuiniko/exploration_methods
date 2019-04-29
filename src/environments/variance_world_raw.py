import numpy as np
from environments.variance_world import Environment as VarianceWorld

class Environment(VarianceWorld):
    def __init__(self):
        super().__init__()

    def env_start(self):
        self.pos = np.random.uniform(0.45, 0.55)
        return np.array([self.pos])

    def env_step(self, action):
        self.pos += self.actions(action)

        noisy = self.pos > 1
        stable = self.pos < 0

        terminal = noisy or stable
        reward = self.noisy_reward() if noisy else 1 if stable else 0

        return reward, np.array([self.pos]), terminal

    def numstates(self):
        return 1
