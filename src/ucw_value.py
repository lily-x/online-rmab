""" UC Whittle

QP per episode
"""

import warnings
import numpy as np
import heapq

import gurobipy as gp
from gurobipy import GRB

from compute_whittle import arm_compute_whittle
from utils import Memoizer, get_valid_lcb_ucb, get_ucb_conf

def UCWhittle_value(env, n_episodes, n_epochs, discount, alpha, VERBOSE=False):
    """
    value-based UCWhittle
    discount = discount factor
    alpha = for confidence radius """
    N           = env.cohort_size
    n_states    = env.number_states
    n_actions   = env.all_transitions.shape[2]
    budget      = env.budget
    T           = env.episode_len * n_episodes
    episode_len = env.episode_len

    env.reset_all()

    print(f'solving UCWhittle using method: VALUE-BASED')

    memoizer_p_s = Memoizer(method='p_s')
    memoizer_lcb_ucb_s_lamb = Memoizer(method='lcb_ucb_s_lamb')

    all_reward = np.zeros((n_epochs, T + 1))

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()

        n_pulls  = np.zeros((N, n_states, n_actions))  # number of pulls
        cum_prob = np.zeros((N, n_states, n_actions))  # transition probability estimates for going to ENGAGING state

        print('first state', env.observe())
        all_reward[epoch, 0] = env.get_reward()

        t = 1  # total timestep across all episodes
        lamb_val = 0 # initialize subsidy for acting to 0
        for episode in range(n_episodes):

            ####################################################
            # run each episode
            ####################################################

            for l in range(episode_len):
                ####################################################
                # precompute
                ####################################################

                est_p, conf_p = get_ucb_conf(cum_prob, n_pulls, t, alpha, env.episode_count)

                # compute value for each arm, for each state
                opt_values, opt_p = QP_value_step(est_p, conf_p, discount, lamb_val, budget, memoizer_lcb_ucb_s_lamb)

                ####################################################
                # solve value iteration to get WI, using opt values
                ####################################################

                state = env.observe()

                # compute whittle index for each arm
                state_WI = np.zeros(N)
                top_WI   = []
                min_chosen_subsidy = 0
                for i in range(N):
                    arm_transitions = opt_p[i, state[i], :, :]
                    # memoize to speed up
                    check_set_val = memoizer_p_s.check_set(arm_transitions, state[i])
                    if check_set_val != -1:
                        state_WI[i] = check_set_val
                    else:
                        state_WI[i] = arm_compute_whittle(arm_transitions, state[i], discount, subsidy_break=min_chosen_subsidy)
                        memoizer_p_s.add_set(arm_transitions, state[i], state_WI[i])

                    if len(top_WI) < budget:
                        heapq.heappush(top_WI, (state_WI[i], i))
                    else:
                        # add state_WI to heap
                        heapq.heappushpop(top_WI, (state_WI[i], i))
                        min_chosen_subsidy = top_WI[0][0]  # smallest-valued item

                ####################################################
                # determine action and act
                ####################################################

                # get action corresponding to optimal subsidy
                # pull K highest indices
                sorted_WI = np.argsort(state_WI)[::-1]

                action = np.zeros(N, dtype=np.int8)
                action[sorted_WI[:budget]] = 1

                # execute chosen policy; observe reward and next state
                next_state, reward, done, _ = env.step(action)

                if done and t < T: env.reset()

                # update estimates
                for i in range(N):
                    for s in range(n_states): # pick states with B largest WI
                        a = action[s]
                        n_pulls[i, s, a] += 1
                        if next_state[i] == 1:
                            cum_prob[i, s, a] += 1

                # print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward, '   | Q_active ', np.round(Q_active[:, state], 3), ' | Q_passive ', np.round(Q_passive, 3), ' | WI ', np.round(state_WI, 3))
                if t % 100 == 0:
                    print('---------------------------------------------------')
                    print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward, '   | WI ', np.round(state_WI, 3))

                all_reward[epoch, t] = reward
                lamb_val = min_chosen_subsidy  # update lambda value used

                t += 1

    return all_reward



def QP_value_step(est_p, conf_p, discount, lamb_val, budget, memoizer_lcb_ucb_s_lamb):
    """ step of UCWhittle using QP solver """

    N, n_states, n_actions = est_p.shape

    # compute UCB and LCB estimates of transition probabilities
    p_ucb = est_p + conf_p
    p_lcb = est_p - conf_p
    p_ucb[p_ucb > 1] = 1  # keep within valid range
    p_lcb[p_lcb < 0] = 0

    opt_values = np.zeros((N, n_states))  # opt values from QP (objective of QP)
    opt_p      = np.zeros((N, n_states, n_states, n_actions))  # opt transition probabilities from QP for each (arm, state) pair

    print('--------------------------')

    min_chosen_subsidy = 0
    for i in range(N):
        arm_p_lcb, arm_p_ucb = get_valid_lcb_ucb(p_lcb[i, :, :], p_ucb[i, :, :])
        for s in range(n_states):
            # memoize to speed up
            check_set_val = memoizer_lcb_ucb_s_lamb.check_set((arm_p_lcb, arm_p_ucb), (s, lamb_val))
            if check_set_val != -1:
                opt_values[i][s] = check_set_val
                # print(f'  >> skipping! arm {i} WI {state_WI[i]:.3f} lcb = {arm_p_lcb.flatten().round(3)}, ucb = {arm_p_ucb.flatten().round(3)}')

            else:
                opt_values[i][s], opt_p[i, s, :, :] = ucw_value_qp(arm_p_lcb, arm_p_ucb, s, lamb_val, discount, subsidy_break=min_chosen_subsidy)
                # print(f'  arm {i} WI: {state_WI[i]:.3f}, state {state[i]}, lcb = {arm_p_lcb.flatten().round(3)}, ucb = {arm_p_ucb.flatten().round(3)}')
                memoizer_lcb_ucb_s_lamb.add_set((arm_p_lcb, arm_p_ucb), (s, lamb_val), opt_values[i][s])

            print(f'      arm {i} state {s}, opt val {opt_values[i][s]:.2f}, opt p {np.round(opt_p[i, s, :, :].flatten(), 3)}')

    return opt_values, opt_p


def ucw_value_qp(p_lcb, p_ucb, s0, lamb_val, discount, subsidy_break):
    """ solve QP to compute Whittle index for each arm independently
    s0: initial state of arm """

    def reward(s): return s

    # add padding to address numerical instability
    epsilon = 1e-4
    p_ucb = p_ucb + epsilon
    p_lcb = p_lcb - epsilon

    # TODO: see if we can incorporate subsidy_break (stop when model.ub?)

    def early_termination(what, where):
        """ callback for gurobi early termination, to incorporate subsidy_break
        callback codes: https://www.gurobi.com/documentation/9.5/refman/cb_codes.html
        """

        if where == GRB.Callback.MIP:
            objbnd = what.cbGet(GRB.Callback.MIP_OBJBND)
            if objbnd < subsidy_break:
                # print(f'  gurobi terminate! {objbnd:.3f}')
                what.terminate()
        elif where == GRB.Callback.MIPSOL:
            objbnd = what.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if objbnd < subsidy_break:
                # print(f'  gurobi terminate! {objbnd:.3f}')
                what.terminate()

    max_iterations = 100 # max number of simplex iterations

    n_states, n_actions = p_lcb.shape

    # set up Gurobi optimizer --------------------------------------------------
    model = gp.Model('UCWhittle_QP')
    model.setParam('OutputFlag', 0) # silence output
    model.setParam('NonConvex', 2)  # nonconvex constraints
    model.setParam('IterationLimit', max_iterations) # limit number of simplex iterations
    # model.setParam('TimeLimit', 100) # seconds
    # model.setParam('DualReductions', 1) # if we get Gurobi error status = 4: https://www.gurobi.com/documentation/9.5/refman/dualreductions.html


    # define variables ---------------------------------------------------------
    # variables to estimate transition probability (within LCB and UCB)
    p       = [[model.addVar(vtype=GRB.CONTINUOUS, lb=p_lcb[s][a], ub=p_ucb[s][a], name=f'p_{s}{a}')
                for a in range(n_actions)] for s in range(n_states)]

    # variable for Q-value
    q_val   = [[model.addVar(vtype=GRB.CONTINUOUS, name=f'q_{s}_{a}')
                for a in range(n_actions)] for s in range(n_states)]
    value_s = [model.addVar(vtype=GRB.CONTINUOUS, name=f'v_{s}')
                for s in range(n_states)]


    # define objective ---------------------------------------------------------
    model.setObjective(value_s[s0], GRB.MAXIMIZE)


    # define constraints -------------------------------------------------------
    # Q function (s, a)
    model.addConstrs(((q_val[s][a] == -lamb_val * a + reward(s) + discount * (value_s[1] * p[s][a] +
                                                                              value_s[0] * (1 - p[s][a])))
        for s in range(n_states) for a in range(n_actions)), 'q_val')

    # value function (s)
    model.addConstrs(((value_s[s] == gp.max_([q_val[s][0], q_val[s][1]]))
                       # value_s[s] == gp.max_([value_sa[s][a] for a in range(n_actions)]))
        for s in range(n_states)), 'value_s')

    # add constraints that enforce probability validity
    model.addConstrs((p[s][1] >= p[s][0] for s in range(n_states)), 'good_to_act')  # always good to act
    model.addConstrs((p[1][a] >= p[0][a] for a in range(n_actions)), 'good_state')  # always good to be in good state

    # valid subsidy
    # model.addConstr((value_sa[s0][0] <= value_sa[s0][1]), 'valid_subsidy')


    # optimize -----------------------------------------------------------------
    model.optimize(early_termination)  # early_termination - set callback

    if model.status != GRB.OPTIMAL:
        warnings.warn(f'Uh oh! Model is not optimal; status is {model.status}')

    try:
        opt_value_s0 = value_s[s0].x
    except:
        # https://www.gurobi.com/documentation/9.5/refman/attributes.html#sec:Attributes
        print(f'can\'t get subsidy value. model status is {model.status}')
        print(f'  state {s0}, lcb = {p_lcb.flatten().round(3)}, ucb = {p_ucb.flatten().round(3)}')
        # print(f'objective value {model.objVal}')
        # print(f'objective bound {model.objBound}') # Best available objective bound (lower bound for minimization, upper bound for maximization)

        opt_value_s0 = -1
        # print(f'objective value {model.getObjective().getValue()}')
        # raise

    try:
        opt_p = np.array([[p[s][a].x for a in range(n_actions)] for s in range(n_states)])
    except:
        print(f'can\'t get p values. model status is {model.status}')
        print(f'  state {s0}, lcb = {p_lcb.flatten().round(3)}, ucb = {p_ucb.flatten().round(3)}')

        opt_p = (p_lcb + p_ucb) / 2


    # print('subsidy  ', subsidy.x)
    # print('prob     ', p[0][0].x, p[0][1].x, p[1][0].x, p[1][1].x)
    # print('value_sa ', value_sa[0][0].x, value_sa[0][1].x, value_sa[1][0].x, value_sa[1][1].x)
    # print('value_s  ', value_s[0].x, value_s[1].x)
    return opt_value_s0, opt_p
