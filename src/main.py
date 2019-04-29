
from copy import copy, deepcopy
import fcntl
import os
import sys
import time
from io import BytesIO

import numpy as np

from rl_glue import RLGlue
from agents import random_agent, softmax_agent, state_bonus_agent, \
    state_action_bonus_agent, ieql_plus, softmaxAC, tc_state_prediction_agent, \
    bootstrap_q, noisy_net
from agents.base_agents import QlambdaAgent, SarsaLambdaAgent

from environments import variance_world, \
    variance_world_raw, antishaping, antishaping_raw, \
    hypercube, hypercube_raw, riverswim, riverswim_raw


def save_results(data, data_size, filename):
    # data: floating point, data_size: integer, filename: string
    with open(filename, "w") as data_file:
        for i in range(data_size):
            data_file.write("{0}\n".format(data[i]))


def experiment(glue, num_steps, result_file, info_string):

    record_step = 20000
    data_size = int(num_steps / record_step) + 1
    rmse = np.zeros(data_size)
    pval = np.zeros(data_size)

    glue.rl_init()

    term = True
    for step in range(num_steps + 1):
        if term:
            glue.rl_start()
        _, _, _, term = glue.rl_step()
        if step % record_step == 0:
            try:
                rmse[int(step/record_step)] = get_nrmse(glue)
            except TypeError:
                pass
            pval[int(step/record_step)] = expensive_policy_value(glue)

    with open(result_file, "a+") as data_file:
        fcntl.flock(data_file, fcntl.LOCK_EX)
        for i in range(2):
            data_file.write('{}, '.format(info_string))

            if i == 0:
                data_file.write('NRMSE, ')
                dat = rmse
            else:
                data_file.write('Policy Value, ')
                dat = pval

            s = BytesIO()
            np.savetxt(s, dat, newline=",")
            data_file.write("{}\n".format(s.getvalue().decode()[:-1]))

        fcntl.flock(data_file, fcntl.LOCK_UN)


def main(func=experiment):
    np.set_printoptions(threshold=8000)

    start = time.time()

    agent_names = {1: "random",
                   22: 'softmax_Q',
                   26: 'state_bonus_Q',
                   28: 'state_action_bonus_Q',
                   30: 'optimistic_init_Q',
                   31: 'epsilon_greedy_Q',
                   32: 'IEQL+_Q',
                   34: 'softmax_AC',
                   38: 'tc_state_prediction_Q',
                   39: 'bootstrap_Q',
                   40: 'noisy_net_Q',
                   }

    agents = {1: random_agent.Agent,
              22: softmax_agent.QAgent,
              26: state_bonus_agent.QAgent,
              28: state_action_bonus_agent.QAgent,
              30: QlambdaAgent,
              31: QlambdaAgent,
              32: ieql_plus.QAgent,
              34: softmaxAC.Agent,
              38: tc_state_prediction_agent.QAgent,
              39: bootstrap_q.QAgent,
              40: noisy_net.QAgent,
              }

    agent_tag = int(sys.argv[1])
    agent_name = agent_names[agent_tag]

    envs = {13: variance_world.Environment,
            14: antishaping.Environment,
            15: hypercube.Environment,
            16: riverswim.Environment,
            19: variance_world_raw.Environment,
            20: antishaping_raw.Environment,
            21: hypercube_raw.Environment,
            22: riverswim_raw.Environment
            }

    env_names = {13: 'variance_world',
                 14: 'antishaping',
                 15: 'hypercube',
                 16: 'riverswim',
                 19: 'variance_world_raw',
                 20: 'antishaping_raw',
                 21: 'hypercube_raw',
                 22: 'riverswim_raw'
                 }

    alpha = float(sys.argv[2])
    beta = float(sys.argv[3])
    alpha_e = float(sys.argv[4])
    init_val = float(sys.argv[5])
    epsilon = float(sys.argv[6])
    behaviour_gamma = float(sys.argv[7])
    agent_gamma = float(sys.argv[8])
    agentlambda = float(sys.argv[9])
    task_lambda = float(sys.argv[10])
    c = float(sys.argv[11])
    env_tag = int(sys.argv[12])
    environment = envs[env_tag]()
    steps = int(sys.argv[13])
    run = int(sys.argv[14])
    filename = 'data/{}'.format(sys.argv[15])

    agent_info = {"alpha": alpha,
                  "beta": beta,
                  "alpha_e": alpha_e,
                  "init_val": init_val,
                  'initialvalue': init_val,
                  'num_actions': environment.numactions(),
                  'num_states': environment.numstates(),
                  'numactions': environment.numactions(),
                  'numstates': environment.numstates(),
                  'agent_name': agent_name,
                  'task_lambda': task_lambda,
                  'run': run,
                  'epsilon': epsilon,
                  'C': c,
                  'gamma': behaviour_gamma,
                  'gamma_e': agent_gamma,
                  'agentlambda': agentlambda,
                  }

    agent = agents[agent_tag](**agent_info)

    # python 3.5 friendly
    datastring = ('{agent_name}, {alpha}, {beta}, {init_val}, {epsilon}, ' +
                  '{gamma}, {gamma_e}, {agentlambda}, {task_lambda}, {C}, ' +
                  '{run}').format(**agent_info) + ', {}'.format(env_tag)
    infostring = ('{agent_name}, ' + 'env={}, '.format(env_names[env_tag]) +
                  'alpha={alpha}, beta={beta}, ' +
                  'lambda={agentlambda}, ' +
                  'init={init_val}, epsilon={epsilon}, ' +
                  'run={run}, task_lambda={task_lambda}, ' +
                  'gamma={gamma}, agent_gamma={gamma_e}').format(**agent_info)

    rl_glue = RLGlue(environment, agent)
    func(rl_glue, steps, filename, datastring)

    print("{} {} in {}s.".format("Done", infostring, time.time()-start))


if __name__ == "__main__":
    if 'src' not in os.path.split(os.getcwd())[1]:
        os.chdir('src')
    main(experiment)
