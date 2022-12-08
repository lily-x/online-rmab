import numpy as np


class Memoizer:
    """ improve performance of memoizing solutions (to QP and WI value iteration) """
    def __init__(self, method):
        self.method = method
        self.solved_p_vals = {}

    def to_key(self, input1, input2):
        """ convert inputs to a key

        QP: inputs: LCB and UCB transition probabilities
        UCB and extreme: inputs - estimated transition probabilities and initial state s0 """
        if self.method in ['lcb_ucb', 'QP', 'QP-min']:
            lcb, ucb = input1, input2
            p_key = (np.round(lcb, 4).tobytes(), np.round(ucb, 4).tobytes())
        elif self.method in ['p_s', 'optimal', 'UCB', 'extreme', 'ucw_value']:
            transitions, state = input1, input2
            p_key = (np.round(transitions, 4).tobytes(), state)
        elif self.method in ['lcb_ucb_s_lamb']:
            lcb, ucb = input1
            s, lamb_val = input2
            p_key = (np.round(lcb, 4).tobytes(), np.round(ucb, 4).tobytes(), s, lamb_val)
        else:
            raise Exception(f'method {self.method} not implemented')

        return p_key

    def check_set(self, input1, input2):
        p_key = self.to_key(input1, input2)
        if p_key in self.solved_p_vals:
            return self.solved_p_vals[p_key]
        return -1

    def add_set(self, input1, input2, wi):
        p_key = self.to_key(input1, input2)
        self.solved_p_vals[p_key] = wi

def get_valid_lcb_ucb(arm_p_lcb, arm_p_ucb):
    n_states, n_actions = arm_p_lcb.shape

    # enforce validity constraints
    assert n_actions == 2  # these checks only valid for two-action
    for s in range(n_states):
        # always better to act
        if arm_p_ucb[s, 0] > arm_p_ucb[s, 1]:  # move passive UCB down
            arm_p_ucb[s, 0] = arm_p_ucb[s, 1]
        if arm_p_lcb[s, 1] < arm_p_lcb[s, 1]:  # move active LCB up
            arm_p_lcb[s, 1] = arm_p_lcb[s, 1]

    assert n_states == 2  # these checks only valid for two-state
    for a in range(n_actions):
        # always better to start in good state
        if arm_p_ucb[0, a] > arm_p_ucb[1, a]:  # move bad-state UCB down
            arm_p_ucb[0, a] = arm_p_ucb[1, a]
        if arm_p_lcb[1, a] < arm_p_lcb[0, a]:  # move good-state LCB up
            arm_p_lcb[1, a] = arm_p_lcb[0, a]

    # these above corrections may lead to LCB being higher than UCBs... so make the UCB the optimistic option
    if arm_p_ucb[0, 0] < arm_p_lcb[0, 0]:
        print(f'ISSUE 00!! lcb {arm_p_lcb[0, 0]:.4f} ucb {arm_p_ucb[0, 0]:.4f}')
        arm_p_ucb[0, 0] = arm_p_lcb[0, 0] # p_ucb[i, 0, 0]
    if arm_p_ucb[0, 1] < arm_p_lcb[0, 1]:
        print(f'ISSUE 01!! lcb {arm_p_lcb[0, 1]:.4f} ucb {arm_p_ucb[0, 1]:.4f}')
        arm_p_ucb[0, 1] = arm_p_lcb[0, 1] # p_ucb[i, 0, 1]
    if arm_p_ucb[1, 0] < arm_p_lcb[1, 0]:
        print(f'ISSUE 10!! lcb {arm_p_lcb[1, 0]:.4f} ucb {arm_p_ucb[1, 0]:.4f}')
        arm_p_ucb[1, 0] = arm_p_lcb[1, 0] # p_ucb[i, 1, 0]
    if arm_p_ucb[1, 1] < arm_p_lcb[1, 1]:
        print(f'ISSUE 11!! lcb {arm_p_lcb[1, 1]:.4f} ucb {arm_p_ucb[1, 1]:.4f}')
        arm_p_ucb[1, 1] = arm_p_lcb[1, 1] # p_ucb[i, 1, 1]

    return arm_p_lcb, arm_p_ucb


def get_ucb_conf(cum_prob, n_pulls, t, alpha, episode_count, delta=1e-3):
    """ calculate transition probability estimates """
    n_arms, n_states, n_actions = n_pulls.shape

    with np.errstate(divide='ignore'):
        n_pulls_at_least_1 = np.copy(n_pulls)
        n_pulls_at_least_1[n_pulls == 0] = 1
        est_p               = cum_prob / n_pulls_at_least_1
        est_p[n_pulls == 0] = 1 / n_states  # where division by 0

        conf_p = np.sqrt( 2 * n_states * np.log( 2 * n_states * n_actions * n_arms * ((episode_count+1)**4 / delta) ) / n_pulls_at_least_1 )
        conf_p[n_pulls == 0] = 1
        conf_p[conf_p > 1]   = 1  # keep within valid range

    # if VERBOSE: print('conf', np.round(conf_p.flatten(), 2))
    # if VERBOSE: print('est p', np.round(est_p.flatten(), 2))

    return est_p, conf_p
