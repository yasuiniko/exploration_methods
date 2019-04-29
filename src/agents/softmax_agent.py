import numpy as np

from agents.base_agents import QlambdaAgent, SarsaLambdaAgent


"""
Softmax agents that chose actions according to a softmax distribution over
action-values
"""


def make_agent(Parent):
    class Mixin(Parent):
        """Softmax action choice initialization"""
        def __init__(self, num_actions, num_states, alpha, init_val,
                     agentlambda, epsilon, gamma, *args, **kwargs):
            del args, kwargs

            super().__init__(numactions=num_actions,
                             numstates=num_states,
                             epsilon=epsilon,
                             alpha=alpha,
                             gamma=gamma,
                             initialvalue=init_val,
                             agentlambda=agentlambda)

            # extra
            self.recentsensations = None
            self.recentactions = None
            self.changedstates = None

        def softmax(self, s, *args, **kwargs):
            del args, kwargs
            values = self.Q[s] if type(s) is int else self.Q[s].sum(0)
            e_v = np.exp(values - np.max(values))
            return e_v / e_v.sum()

        def policy(self, state):
            return np.random.choice(a=self.numactions, p=self.softmax(state))

        def enable_target_policy(self):
            self.agent_learn = lambda *args, **kwargs: None
            self.policy = lambda s: np.argmax(self.softmax(s))

    return Mixin


QAgent = make_agent(QlambdaAgent)
SARSAAgent = make_agent(SarsaLambdaAgent)
