""" driver code for experiments

run:
[small toy example]  python main.py -P 10 -N 10 -T 50 -B 3 -E 1
[default]  python main.py -P 100 -N 100 -T 500 -B 20 -E 30
"""

import numpy as np
import pandas as pd
import random
import time, datetime
import sys, os
import argparse

import matplotlib.pyplot as plt

from simulator import RMABSimulator, random_valid_transition, random_valid_transition_round_down, synthetic_transition_small_window
from uc_whittle import UCWhittle
from ucw_value import UCWhittle_value
from baselines import optimal_policy, random_policy, WIQL


def smooth(rewards, weight=0.7):
    """ smoothed exponential moving average """
    prev = rewards[0]
    smoothed = np.zeros(len(rewards))
    for i, val in enumerate(rewards):
        smoothed_val = prev * weight + (1 - weight) * val
        smoothed[i] = smoothed_val
        prev = smoothed_val

    return smoothed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms',         '-N', help='num beneficiaries (arms)', type=int, default=8)
    parser.add_argument('--episode_len',    '-H', help='episode length', type=int, default=20)
    parser.add_argument('--n_episodes',     '-T', help='num episodes', type=int, default=30)
    parser.add_argument('--budget',         '-B', help='budget', type=int, default=3)
    parser.add_argument('--data',           '-D', help='dataset to use {synthetic, real}', type=str, default='synthetic')

    parser.add_argument('--n_epochs',       '-E', help='number of epochs (num_repeats)', type=int, default=10)
    parser.add_argument('--discount',       '-d', help='discount factor', type=float, default=0.9)
    parser.add_argument('--alpha',          '-a', help='alpha: for conf radius', type=float, default=3)

    parser.add_argument('--n_states',       '-S', help='num states', type=int, default=2)
    parser.add_argument('--n_actions',      '-A', help='num actions', type=int, default=2)
    parser.add_argument('--seed',           '-s', help='random seed', type=int, default=42)
    parser.add_argument('--verbose',        '-V', help='if True, then verbose output (default False)', action='store_true')
    parser.add_argument('--local',          '-L', help='if True, running locally (default False)', action='store_true')
    parser.add_argument('--prefix',         '-p', help='prefix for file writing', type=str, default='')


    args = parser.parse_args()


    # problem setup
    n_arms      = args.n_arms
    budget      = args.budget
    n_states    = args.n_states
    n_actions   = args.n_actions

    # solution/evaluation setup
    discount    = args.discount
    alpha       = args.alpha #7 - too pessimistic #0.1 - too optimistic

    # experiment setup
    seed        = args.seed
    VERBOSE     = args.verbose
    LOCAL       = args.local
    prefix      = args.prefix
    n_episodes  = args.n_episodes
    episode_len = args.episode_len
    n_epochs    = args.n_epochs
    data        = args.data
    # real_data   = args.real_data


    # separate out things we don't want to execute on the cluster
    if LOCAL:
        import matplotlib as mpl
        mpl.use('tkagg')

    np.random.seed(seed)
    random.seed(seed)


    if not os.path.exists(f'figures/{data}'):
        os.makedirs(f'figures/{data}')

    if not os.path.exists(f'results/{data}'):
        os.makedirs(f'results/{data}')


    # -------------------------------------------------
    # initialize RMAB simulator
    # -------------------------------------------------

    if data in ['real']:
        if data == 'real':
            print('real data')
            transitions = get_armman_data()

        assert n_arms <= transitions.shape[0]
        assert transitions.shape[1] == n_states
        assert transitions.shape[2] == n_actions
        all_population_size = transitions.shape[0]

        if all_population_size < transitions.shape[0]:
            transitions = transitions[0:all_population_size, :]

        all_transitions = np.zeros((all_population_size, n_states, n_actions, n_states))
        all_transitions[:,:,:,1] = transitions
        all_transitions[:,:,:,0] = 1 - transitions

    elif data == 'synthetic':
        print('synthetic data')
        all_population_size = 100 # number of random arms to generate
        all_transitions = random_valid_transition(all_population_size, n_states, n_actions)

    else:
        raise Exception(f'dataset {data} not implemented')


    all_features = np.arange(all_population_size)

    if VERBOSE: print(f'transitions ----------------\n{np.round(all_transitions, 2)}')
    simulator = RMABSimulator(all_population_size, all_features, all_transitions,
            n_arms, episode_len, n_epochs, n_episodes, budget, number_states=n_states)


    # -------------------------------------------------
    # run comparisons
    # -------------------------------------------------

    use_algos = ['optimal', 'ucw_value', 'ucw_qp', 'ucw_extreme', 'wiql', 'random'] # 'ucw_qp_min', 'ucw_ucb',

    rewards  = {}
    runtimes = {}
    colors   = {'optimal': 'purple', 'ucw_value': 'b', 'ucw_qp': 'c', 'ucw_qp_min': 'goldenrod', 'ucw_ucb': 'darkorange',
                'ucw_extreme': 'r', 'wiql': 'limegreen', 'random': 'brown'}

    if 'optimal' in use_algos:
        print('-------------------------------------------------')
        print('optimal policy')
        print('-------------------------------------------------')
        start                 = time.time()
        rewards['optimal']    = optimal_policy(simulator, n_episodes, n_epochs, discount)
        runtimes['optimal']   = time.time() - start

    if 'ucw_value' in use_algos: # UCW-value (P_V, main approach)
        print('-------------------------------------------------')
        print('UCWhittle - value-based QP')
        print('-------------------------------------------------')
        start                 = time.time()
        rewards['ucw_value']  = UCWhittle_value(simulator, n_episodes, n_epochs, discount, alpha=alpha)
        runtimes['ucw_value'] = time.time() - start

    if 'ucw_qp' in use_algos: # UCW-penalty (P_m, heuristic approach)
        print('-------------------------------------------------')
        print('UCWhittle QP')
        print('-------------------------------------------------')
        start              = time.time()
        rewards['ucw_qp']  = UCWhittle(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='QP')
        runtimes['ucw_qp'] = time.time() - start

    if 'ucw_qp_min' in use_algos: # quadratic constraint program (QCP) that directly minimizes lambda (penalty)
        print('-------------------------------------------------')
        print('UCWhittle QP - minimizing')
        print('-------------------------------------------------')
        start                  = time.time()
        rewards['ucw_qp_min']  = UCWhittle(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='QP-min')
        runtimes['ucw_qp_min'] = time.time() - start

    if 'ucw_ucb' in use_algos: # use upper confidence bounds
        print('-------------------------------------------------')
        print('UCWhittle UCB')
        print('-------------------------------------------------')
        start               = time.time()
        rewards['ucw_ucb']  = UCWhittle(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='UCB')
        runtimes['ucw_ucb'] = time.time() - start

    if 'ucw_extreme' in use_algos: # UCW-extreme (extreme points)
        print('-------------------------------------------------')
        print('UCWhittle extreme points')
        print('-------------------------------------------------')
        start                   = time.time()
        rewards['ucw_extreme']  = UCWhittle(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='extreme')
        runtimes['ucw_extreme'] = time.time() - start

    if 'wiql' in use_algos: # WIQL from Biswas et al.
        print('-------------------------------------------------')
        print('WIQL')
        print('-------------------------------------------------')
        start                  = time.time()
        rewards['wiql']        = WIQL(simulator, n_episodes, n_epochs)
        runtimes['wiql']       = time.time() - start

    if 'random' in use_algos: # random policy
        print('-------------------------------------------------')
        print('random policy')
        print('-------------------------------------------------')
        start                  = time.time()
        rewards['random']      = random_policy(simulator, n_episodes, n_epochs)
        runtimes['random']     = time.time() - start

    print('-------------------------------------------------')
    print('runtime')
    for algo in use_algos:
        print(f'  {algo}:   {runtimes[algo]:.2f} s')


    x_vals = np.arange(n_episodes * episode_len + 1)

    def get_cum_sum(reward):
        cum_sum = reward.cumsum(axis=1).mean(axis=0)
        cum_sum = cum_sum / (x_vals + 1)
        return smooth(cum_sum)

    exp_name_out = f'{data}_n{n_arms}_b{budget}_s{n_states}_a{n_actions}_H{episode_len}_L{n_episodes}_epochs{n_epochs}'

    str_time = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')


    # -------------------------------------------------
    # write out CSV
    # -------------------------------------------------
    for algo in use_algos:
        data_df = pd.DataFrame(data=rewards[algo], columns=x_vals)

        runtime = runtimes[algo] / n_epochs
        prepend_df = pd.DataFrame({'seed': seed, 'n_arms': n_arms, 'budget': budget,
                                    'n_states': n_states, 'n_actions': n_actions,
                                    'discount': discount, 'n_episodes': n_episodes, 'episode_len': episode_len,
                                    'n_epochs': n_epochs, 'runtime': runtime, 'time': str_time}, index=[0])

        prepend_df = pd.concat([prepend_df]*n_epochs, ignore_index=True)

        out_df = pd.concat([prepend_df, data_df], axis=1)

        filename = f'results/{data}/reward_{exp_name_out}_{algo}.csv'

        with open(filename, 'a') as f:
            # write header, if file doesn't exist
            if f.tell() == 0:
                print(f'creating file {filename}')
                out_df.to_csv(f)

            # write results (appending) and no header
            else:
                print(f'appending to file {filename}')
                out_df.to_csv(f, mode='a', header=False)

    # -------------------------------------------------
    # visualize
    # -------------------------------------------------
    # plot average cumulative reward
    plt.figure()
    for algo in use_algos:
        plt.plot(x_vals, get_cum_sum(rewards[algo]), c=colors[algo], label=algo)
    plt.legend()
    plt.xlabel(f'timestep $t$ ({n_episodes} episodes of length {episode_len})')
    plt.ylabel('average cumulative reward')
    plt.title(f'{data} - N={n_arms}, B={budget}, discount={discount}, {n_epochs} epochs')
    plt.savefig(f'figures/{data}/cum_reward_{exp_name_out}_{str_time}.pdf')
    if LOCAL: plt.show()

    # plot average reward
    plt.figure()
    for algo in use_algos:
        plt.plot(x_vals, smooth(rewards[algo].mean(axis=0)), c=colors[algo], label=algo)
    plt.legend()
    plt.xlabel(f'timestep $t$ ({n_episodes} episodes of length {episode_len}')
    plt.ylabel('average reward')
    plt.title(f'{data} - N={n_arms}, budget={budget}, discount={discount}, {n_epochs} epochs')
    plt.savefig(f'figures/{data}/avg_reward_{exp_name_out}_{str_time}.pdf')
    if LOCAL: plt.show()
