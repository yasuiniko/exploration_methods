#!/usr/bin/env python
"""
Next-state prediction error reward bonuses, implemented by predicting the next
tile-coded feature vector.
"""

import numpy as np

from agents.base_agents import QlambdaAgent, SarsaLambdaAgent

from tilecoding import TileCoder


# Referenced from each env. files
def choose_tile_coding_setting(env_num):
    if env_num == 0.0:  # Sparse/Dense MC, 288
        num_tiles = np.array([2, 2])
        num_tilings = 32
        limits = [[-0.7, 0.7], [-1.2, 0.5]]

    elif env_num == 1.0:  # variance_world, 96
        num_tiles = np.asarray([2])
        num_tilings = 32
        limits = [[0, 1]]

    elif env_num == 2.0:  # antishaping, 96
        num_tiles = np.asarray([2])
        num_tilings = 32
        limits = [[0, 1]]

    elif env_num == 3.0:  # hypercube, 7776 = 3^5 * 32
        n = 5  # assuming default setting
        radius = 10

        num_tiles = np.asarray([2] * n)
        num_tilings = 32
        limits = [[-radius, radius]] * n

    elif env_num == 4.0:  # riverswim
        num_tiles = np.asarray([2])
        num_tilings = 32
        limits = [[0, 1]]
    else:
        raise ValueError('Invalid tile coding setting')
    return num_tiles, num_tilings, limits


class OGPredictionLinearModel:
    def __init__(self, tc_size, numstates, numactions, learningrate):
        super(OGPredictionLinearModel, self).__init__()
        self.w = np.zeros((numstates, tc_size, numactions))
        self.lr = learningrate

    def forward(self, x, a):
        pred = self.w[:, x, a]
        # print('pred shape:', np.shape(pred))
        pred = np.sum(pred, axis=1)
        # print('pred shape:', np.shape(pred))
        # input()
        return pred

    def update(self, phi_s, a, target, prediction):
        loss = target - prediction
        # print('loss:', np.shape(loss[0]))
        # print('w:', np.shape(self.w[0, phi_s, a]))
        # print('lr:', self.lr)
        # input()
        for i in range(len(loss)):
            self.w[i, phi_s, a] += self.lr * loss[i]


def state_prediction(Parent):
    class Mixin(Parent):
        def __init__(self, num_actions, num_states, alpha, init_val, beta, gamma_e,
                     agentlambda, epsilon, gamma, C, alpha_e, task_lambda, *args, **kwargs):

            self.num_tiles, self.num_tilings, self.limits = choose_tile_coding_setting(alpha_e)
            self.tiler = TileCoder(dims=self.num_tiles,
                                   limits=self.limits,
                                   tilings=self.num_tilings)

            self.tc_size = self.tiler.n_tiles

            super().__init__(numactions=num_actions,
                             numstates=self.tc_size,
                             epsilon=epsilon,
                             alpha=alpha,
                             gamma=gamma,
                             initialvalue=init_val,
                             agentlambda=agentlambda)

            self.numrealstates = num_states

            self.beta = beta  # 1.0, 10, 100, 1000
            self.omega = gamma_e  # 0.125, 0.015625, 0.0078125
            self.model_update_steps = int(C)  # just 1

            self.net = OGPredictionLinearModel(self.tc_size, self.numrealstates, self.numactions, self.omega)
            self.criterion = None
            self.optimizer = None

            self.num_steps = None
            self.max_pred_error = None

        def agent_init(self):
            super().agent_init()

            self.num_steps = 0
            self.max_pred_error = None

        def learn_linear_model(self, phi_s, a, r, sp=None):

            self.num_steps += 1

            if sp is not None:

                # psi_s_a = np.append(self.get_feature_vector(phi_s), a - 1.0)

                assert self.predict_tc is False

                sp_prediction = self.net.forward(phi_s, a)
                # next_pred_error = self.get_l2_distance(sp_prediction, sp)
                next_pred_error = self.mse_loss(sp_prediction, sp)

                if self.max_pred_error is None:
                    self.max_pred_error = next_pred_error
                elif next_pred_error > self.max_pred_error:
                    self.max_pred_error = next_pred_error

                normalized_next_pred_error = (next_pred_error / self.max_pred_error) / self.num_steps  # * 100

                # print('sp', sp)
                # print('sp_pred', sp_prediction)
                # print('======= error', next_pred_error, normalized_next_pred_error)

                # update every step
                self.net.update(phi_s, a, sp, sp_prediction)

                bonus = self.beta * normalized_next_pred_error

            else:
                bonus = 0

            return bonus

        def agent_start(self, state):
            # print('start', state)
            # input()
            return super().agent_start(self.get_phi(state))

        def agent_step(self, reward, sprime, verbose=False):
            # print('step', sprime)
            # input()
            phi_s = self.last_state  # already tilecoded
            a = self.last_action

            phi_sprime = self.get_phi(sprime)
            aprime = self.policy(phi_sprime)

            if self.use_nn:
                bonus = self.learn_model(phi_s, a, reward, sprime)
            else:
                bonus = self.learn_linear_model(phi_s, a, reward, sprime)

            reward += bonus
            self.agent_learn(phi_s, a, reward, phi_sprime, aprime, verbose)

            self.last_state = phi_sprime
            self.last_action = aprime

            return aprime

        def target_agent_step(self, reward, sprime, verbose=False):
            return self.policy(self.get_phi(sprime))

        def enable_target_policy(self):
            self.policy = lambda s: egreedy(0, self.actionvalues(s))
            self.agent_step = self.target_agent_step

        def get_phi(self, state):
            return self.tiler[state]

        # Turn phi(s) into giant binary 32-hot vector
        def get_feature_vector(self, phi_state):
            feature_vec = np.zeros(self.tc_size)
            feature_vec[phi_state] = 1  # 32 hot vector!

            return feature_vec

        def get_l2_distance(self, vec_1, vec_2):
            return np.sqrt(np.sum(np.square(vec_1 - vec_2)))

        def mse_loss(self, pred, label):
            loss = 0.5 * np.sum(np.square(pred - label))

            return loss
    return Mixin


QAgent = state_prediction(QlambdaAgent)
SARSAAgent = state_prediction(SarsaLambdaAgent)
