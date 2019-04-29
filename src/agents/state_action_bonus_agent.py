"""
Implemented a linear agent with state-action count-based reward bonuses. 
"""

import numpy as np

from agents.base_agents import QlambdaAgent, SarsaLambdaAgent, egreedy


class StateCounter:
    def __init__(self, num_states, num_actions):
        self.N = np.zeros((num_states, num_actions), dtype=np.int)

    def __call__(self, s, a=0):
        self.N[s, a] += 1
        return self.N[s, a]


class FeatureCounter:
    def __init__(self, num_states, num_actions):
        # Krichevsky-Trofimov estimator initialization
        self.logt = 0
        self.logN = np.log(np.ones((num_states, num_actions)) / 2)

    def __call__(self, s, a=0):
        p = np.exp(np.sum(self.logN[s, a]))

        # update incremental log average
        self.logt = np.log1p(np.exp(self.logt))
        self.logN += np.log1p(-np.exp(-self.logt))
        self.logN[s, a] = np.logaddexp(self.logN[s, a], -self.logt)

        pp = np.exp(np.sum(self.logN[s, a]))
        count = p * (1 - pp) / (pp - p)

        return count


def state_action(Parent):
    class Mixin(Parent):
        def __init__(self, num_actions, num_states, alpha, beta, init_val,
                     agentlambda, epsilon, gamma, *args, **kwargs):
            del args, kwargs
            super().__init__(numactions=num_actions,
                             numstates=num_states,
                             epsilon=epsilon,
                             alpha=alpha,
                             gamma=gamma,
                             initialvalue=init_val,
                             agentlambda=agentlambda)
            self.beta = beta
            self.model = None

            self.last_action = None
            self.last_state = None

        def compute_bonus(self, s, a):
            if self.model is None:
                try:
                    len(s)
                    model_class = FeatureCounter
                except TypeError:
                    model_class = StateCounter
                self.model = model_class(self.numstates, self.numactions)

            return self.beta / np.sqrt(self.model(s, a))

        def agent_step(self, reward, sprime, verbose=False):
            s = self.last_state
            a = self.last_action

            aprime = self.policy(sprime)

            # compute reward
            reward += self.compute_bonus(s, a)

            self.agent_learn(s, a, reward, sprime, aprime, verbose)

            self.last_state = sprime
            self.last_action = aprime

            return aprime

        def target_agent_step(self, reward, sprime, verbose=False):
            return self.policy(sprime)

        def enable_target_policy(self):
            self.policy = lambda s: egreedy(0, self.actionvalues(s))
            self.agent_step = self.target_agent_step

    return Mixin


QAgent = state_action(QlambdaAgent)
SARSAAgent = state_action(SarsaLambdaAgent)
