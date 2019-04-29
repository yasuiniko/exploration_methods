
import numpy as np


def egreedy(epsilon, valuelist):
    if epsilon and np.random.random() < epsilon:
        action = np.random.randint(len(valuelist))
    else:
        shuf = np.random.permutation(len(valuelist))
        action = shuf[np.argmax(np.array(valuelist)[shuf])]

    return action


class Agent:
    def __init__(self, numactions, numstates, epsilon=0, alpha=0.5,
                 gamma=.9, initialvalue=0.0, agentlambda=0.8, *args, **kwargs):
        self.alpha = alpha
        self.initialvalue = initialvalue
        self.gamma = gamma
        self.epsilon = epsilon
        self.agentlambda = agentlambda
        self.numactions = numactions
        self.numstates = numstates
        self.last_state = None
        self.last_action = None
        self.Q = None
        self.z = None

    def agent_init(self):
        self.Q = (np.zeros((self.numstates, self.numactions))
                  + self.initialvalue)
        self.z = np.zeros((self.numstates, self.numactions))

    def agent_start(self, state):
        self.last_state = state
        self.last_action = self.policy(state)

        return self.last_action

    def actionvalues(self, s):
        try:
            len(s)
            return self.Q[s].sum(0)
        except TypeError:
            return self.Q[s]

    def statevalue(self, s):
        invalid_state = s is None or (type(s) is str and s == 'terminal')
        return 0 if invalid_state else np.max(self.actionvalues(s))

    def policy(self, state):
        return egreedy(self.epsilon, self.actionvalues(state))

    def agent_learn(self, s, a, r, sp=None, ap=None):
        # qlearning
        next_val = 0 if sp is None else self.gamma * self.statevalue(sp)
        self.Q[s, a] += self.alpha * (r + next_val - self.Q[s, a])

    def agent_step(self, reward, sprime, verbose=False):
        s = self.last_state
        a = self.last_action

        aprime = self.policy(sprime)

        self.agent_learn(s, a, reward, sprime, aprime, verbose)

        self.last_state = sprime
        self.last_action = aprime

        return aprime

    def agent_end(self, reward):
        self.agent_learn(self.last_state, self.last_action, reward)

    def enable_target_policy(self):
        self.policy = lambda s: egreedy(0, self.actionvalues(s))
        self.agent_learn = lambda s, a, r, *args, **kwargs: None


class SarsaLambdaAgent(Agent):
    """accumulating traces"""
    def agent_learn(self, s, a, r, sp=None, ap=None, verbose=False):

        next_val = 0 if sp is None else self.gamma * self.Q[sp, ap].sum()

        if self.agentlambda:
            self.z *= self.gamma * self.agentlambda
            self.z[s, a] += 1
            # self.z[s, a] = 1  # replacing traces

            self.Q += self.alpha * (r + next_val - self.Q[s, a].sum()) * self.z
        else:
            self.Q[s, a] += self.alpha * (r + next_val - self.Q[s, a].sum())


class QlambdaAgent(Agent):
    """accumulating traces"""
    def agent_learn(self, s, a, r, sp=None, ap=None, verbose=False):

        # check if s is feature vector or single integer
        try:
            assert self.Q[s].sum(0).size == self.numactions
            max_s = self.Q[s].sum(0).max()
            max_sp = 0 if sp is None else self.Q[sp].sum(0).max()
        except AssertionError:
            max_s = self.Q[s].max()
            max_sp = 0 if sp is None else self.Q[sp].max()

        next_val = 0 if sp is None else self.gamma * max_sp

        if self.agentlambda:
            if max_s == self.Q[s, a].sum():
                self.z *= self.gamma * self.agentlambda
                self.z[s, a] += 1
                # self.z[s, a] = 1  # replacing traces

                self.Q += (self.alpha
                           * (r + next_val - self.Q[s, a].sum())
                           * self.z)
            else:
                self.z = np.zeros(self.z.shape)
        else:
            self.Q[s, a] += self.alpha * (r + next_val - self.Q[s, a].sum())
