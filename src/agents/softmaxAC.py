from typing import List

import numpy as np

from agents.softmax_agent import QAgent as SoftmaxAgent

"""
Softmax Actor-critic
"""


class Agent(SoftmaxAgent):
    def __init__(self, num_actions, num_states, alpha, init_val,
                 agentlambda, beta, epsilon, gamma, *args, **kwargs):
        del args, kwargs
        super().__init__(num_actions=num_actions,
                         num_states=num_states,
                         epsilon=epsilon,
                         alpha=alpha,
                         gamma=gamma,
                         init_val=init_val,
                         agentlambda=agentlambda)

        self.w = np.zeros(num_states)
        self.e_w = np.zeros(num_states)
        self.θ = np.zeros((num_states, num_actions))
        self.e_θ = np.zeros((num_states, num_actions))
        self.v_old = 0
        self.α = alpha
        self.beta = beta
        self.γ = gamma
        self.λ = agentlambda
        self.δ = None

        # extras
        self.Q = None
        self.changedstates = None
        self.recentactions = None
        self.recentsensations = None

    def softmax(self, state, *args, **kwargs):
        del args, kwargs
        values = self.θ[state] if type(state) is int else self.θ[state].sum(0)
        e_v = np.exp(values - np.max(values))
        return e_v / e_v.sum()

    def enable_target_policy(self):
        self.agent_learn = lambda *args, **kwargs: None
        self.policy = lambda s: np.random.choice(a=self.numactions,
                                                 p=self.softmax(s))

    def agent_learn(self, s, a, r, sp=None, ap=None, verbose=None):
        del ap, verbose

        # for readability
        α, w, e_w, λ, γ = self.α, self.w, self.e_w, self.λ, self.γ
        θ, e_θ = self.θ, self.e_θ

        # get TD error
        v = w[s].sum()
        vp = 0 if sp is None else w[sp].sum()
        self.δ = r + γ * vp - v

        # critic update with true online TD
        if λ:
            e_wx = e_w[s].sum()
            e_w *= γ * λ
            e_w[s] += 1 - α * γ * λ * e_wx

            w += α * (self.δ + v - self.v_old) * e_w
            w[s] -= α * (v - self.v_old)
            self.v_old = vp
        else:
            w[s] += α * self.δ

        # actor update
        # e_θ *= γ * λ
        # e_θ += self.partial(s, a)

        # replaces partial
        βδ = self.beta * self.δ
        phi_b = self.softmax(s)
        θ[s, a] += βδ
        θ[s] -= βδ * phi_b

        # θ += self.beta * self.δ * self.partial(s, a)

    def partial(self, s: List, a: int):
        action_probs = self.softmax(s)

        # make x(s, b) for all b
        phis = np.zeros([self.numactions, self.numstates, self.numactions])
        for b in range(self.numactions):
            phis[b, s, b] = 1

        sum_b = sum(xbi * bpi for xbi, bpi in zip(phis, action_probs))

        # has the same shape as self.θ
        return phis[a] - sum_b


