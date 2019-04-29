"""
Linear implementation of Osband et al (2016) bootstrap DQN
"""

import numpy as np
import os

from agents.base_agents import QlambdaAgent, egreedy


class QAgent(QlambdaAgent):
    def __init__(self, num_actions, num_states, alpha,
                 agentlambda, epsilon, gamma, *args, **kwargs):
        del args, kwargs

        kwargs = {'numactions': num_actions,
                  'numstates': num_states,
                  'epsilon': epsilon,
                  'alpha': alpha,
                  'gamma': gamma,
                  'agentlambda': agentlambda}
        super().__init__(**kwargs)

        self.num_agents = 10
        self.agents = [QlambdaAgent(**kwargs) for _ in range(self.num_agents)]
        self.policy = None

    def save(self, path):
        qs = np.asarray([a.Q for a in self.agents])
        np.save(os.path.join(f'{path}-{self.num_agents}'), qs)

    def load(self, path):
        _, tail = os.path.split(path)
        self.num_agents = int(tail.split('-')[1][:-4])
        for a, Q in zip(self.agents, np.load(path)):
            a.Q = Q

    def enable_target_policy(self):
        for k in range(self.num_agents):
            self.agents[k].policy = lambda s: \
                egreedy(0, self.agents[k].actionvalues(s))
        self.agent_learn = lambda *args, **kwargs: None

    def agent_init(self):
        super().agent_init()

        for agent in self.agents:
            agent.agent_init()
            # agent.Q = np.random.rand(*agent.Q.shape)

    def agent_start(self, state, k=None):
        if k is None:
            k = np.random.randint(len(self.agents))
        self.policy = self.agents[k].policy

        return super().agent_start(state)

    def agent_learn(self, s, a, r, sp=None, ap=None, verbose=False):
        for agent in self.agents:
            mask = np.random.poisson(1)

            # check if s is feature vector or single integer
            try:
                assert agent.Q[s].sum(0).size == agent.numactions
                max_s = agent.Q[s].sum(0).max()
                max_sp = 0 if sp is None else agent.Q[sp].sum(0).max()
            except AssertionError:
                max_s = agent.Q[s].max()
                max_sp = 0 if sp is None else agent.Q[sp].max()

            next_val = 0 if sp is None else agent.gamma * max_sp
            td_error = (r + next_val - agent.Q[s, a].sum())

            # use traces if action was greedy
            if agent.agentlambda:
                if max_s == agent.Q[s, a].sum():
                    agent.z *= agent.gamma * agent.agentlambda
                    agent.z[s, a] += mask
                else:
                    agent.z = np.zeros(agent.z.shape)
                agent.Q += agent.alpha * td_error * agent.z
            else:
                agent.Q[s, a] += agent.alpha * td_error * mask


class SARSAAgent(QAgent):
    def agent_learn(self, s, a, r, sp=None, ap=None, verbose=False):
        for agent in self.agents:
            mask = np.random.poisson(1)

            next_val = 0 if sp is None else agent.gamma * agent.Q[sp, ap].sum()
            td_error = (r + next_val - agent.Q[s, a].sum())

            if agent.agentlambda:
                agent.z *= agent.gamma * agent.agentlambda
                agent.z[s, a] += mask

                agent.Q += agent.alpha * td_error * agent.z
            else:
                agent.Q[s, a] += agent.alpha * td_error * mask
