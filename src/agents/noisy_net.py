"""
Linear implementation of Fortunato et al. (2015) noisy network
"""
import numpy as np

from agents.base_agents import QlambdaAgent


class QAgent(QlambdaAgent):
    def __init__(self, num_actions, num_states, alpha,
                 agentlambda, epsilon, gamma, *args, **kwargs):
        del args, kwargs
        super().__init__(**{'numactions': num_actions,
                            'numstates': num_states,
                            'epsilon': epsilon,
                            'alpha': alpha,
                            'gamma': gamma,
                            'agentlambda': agentlambda})

    def agent_init(self):
        super().agent_init()

        # h = np.sqrt(0.03 / self.numstates)
        # mean_init = np.random.uniform(-h, h, self.Q.size)

        self.Q = np.zeros(list(self.Q.shape) + [2])
        # self.Q[:, :, 0] = mean_init.reshape(self.Q.shape[0], -1)
        # self.Q[:, :, 1] = 0.0001

    def actionvalues(self, s):
        ε = np.random.normal(size=self.numactions)

        try:
            len(s)
            q_s = self.Q[s].sum(0)
        except TypeError:
            q_s = self.Q[s]

        return q_s[:, 0] + (q_s[:, 1] * ε).flatten()

    def agent_learn(self, s, a, r, sp=None, ap=None, verbose=False):
        num_s = np.asarray(s).size

        # check if s is feature vector or single integer
        max_sp = 0
        if sp is not None:
            q_sp = self.Q[sp]
            if len(q_sp.shape) == 3:
                q_sp = q_sp.sum(0)
            max_sp = np.max(q_sp[:, 0])

        next_val = 0 if sp is None else self.gamma * max_sp
        curr_val = self.Q[s, a, 0].sum()
        td_error = r + next_val - curr_val

        self.Q[s, a, 0] += self.alpha * td_error
        self.Q[s, a, 1] += self.alpha * td_error * np.random.normal(size=num_s)


class SARSAAgent(QAgent):
    def agent_learn(self, s, a, r, sp=None, ap=None, verbose=False):
        num_s = np.asarray(s).size

        next_val = 0 if sp is None else self.gamma * self.Q[sp, ap, 0].sum()
        curr_val = self.Q[s, a, 0].sum()
        td_error = r + next_val - curr_val

        self.Q[s, a, 0] += self.alpha * td_error
        self.Q[s, a, 1] += self.alpha * td_error * np.random.normal(size=num_s)
