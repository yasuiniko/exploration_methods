"""Random agent with discrete actions
"""

import random

from agents.base_agents import Agent as BaseAgent


class Agent(BaseAgent):

    def agent_learn(self, s, a, r, sp=None, ap=None, verbose=False):
        pass

    def policy(self, state):
        return random.randrange(self.numactions)
