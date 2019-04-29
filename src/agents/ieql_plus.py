"""
Linear implementation of Meuleau et al. (1999) IEQL+
"""

import numpy as np
from scipy import special

from agents.state_action_bonus_agent import (QAgent as QBonusAgent,
                                             SARSAAgent as SBonusAgent,
                                             StateCounter,
                                             FeatureCounter)


def ieql(Parent):
    class Mixin(Parent):
        def __init__(self, num_actions, num_states, alpha, init_val,
                     agentlambda, epsilon, C, beta, gamma, *args, **kwargs):
            del args, kwargs
            super().__init__(num_actions=num_actions,
                             num_states=num_states,
                             epsilon=epsilon,
                             alpha=alpha,
                             gamma=gamma,
                             init_val=init_val,
                             agentlambda=agentlambda,
                             beta=0)

            self.alpha = alpha
            self.θ = C
            self.σ_max = beta
            self.counter = None
            self.Q = None

        def δ_0(self, n):
            return self.σ_max * special.ndtri(1 - self.θ / 2) / np.sqrt(n)

        def compute_bonus(self, s, a):
            if self.counter is None:
                scale = self.δ_0(1) / (1 - self.gamma)
                try:
                    # check if state is a list of indices or integer state
                    scale /= len(s)
                    counter_class = FeatureCounter
                except TypeError:
                    counter_class = StateCounter

                # optimistically initalize Q-values
                self.Q = np.ones_like(self.Q) * scale / np.asarray(s).size

                # initialize the state-action counter
                self.counter = counter_class(self.numstates, self.numactions)

            # calculate bonus
            bonus = self.δ_0(self.counter(s, a))

            # # don't multiply by 1-γ because σ_max isn't divided by 1-γ
            # return bonus * (1 - self.gamma)
            return bonus
    return Mixin


QAgent = ieql(QBonusAgent)
SARSAAgent = ieql(SBonusAgent)
