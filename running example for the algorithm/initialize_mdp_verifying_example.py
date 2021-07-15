import numpy as np

from DFA import DFA
from DFA import Action
from MDP_learner_gridworld_cleaned import MDP
import pickle


def inv_dict(target):
    result = {}
    for key, value in target.items():
        if value not in result:
            result[value] = [key]
        else:
            if key not in result[value]:
                result[value].append(key)
    return result


def inv_dict2(target):
    result = {}
    for key, value in target.items():
        # if len(value) == 1:
        #     print(value)
        #     if value[0] not in result:
        #         result[value] = [key]
        #     else:
        #         if key not in result[value]:
        #             result[value].append(key)
        # elif len(value) > 1:
        for element in value:
            if element not in result:
                result[element] = [key]
            else:
                if key not in result[element]:
                    result[element].append(key)
    return result


def dfa_add_trans_weak(dfa):
    # dfa.add_transition(whole.display(), sink, sink)
    dfa.add_transition(b.display(), 3, 1)  # set transitin 3->b->1
    dfa.add_transition(b.display(), 2, 0)

    dfa.add_transition(c.display(), 1, 0)  # set transitin 1->b->0
    dfa.add_transition(c.display(), 3, 2)

    dfa.add_transition(nb.display(), 2, 2)  # set transitin 2->b->2
    dfa.add_transition(nc.display(), 1, 1)
    dfa.add_transition(nbnc.display(), 3, 3)
    dfa.add_transition(whole.display(), 0, 0)


def dfa_add_trans_strict(dfa):
    dfa.add_transition(b.display(), 3, 1)  # set transitin 3->b->1
    dfa.add_transition(c.display(), 3, 2)
    dfa.add_transition(whole.display(), 2, 2)  # set transitin 2->b->2
    dfa.add_transition(whole.display(), 1, 1)
    dfa.add_transition(nbnc.display(), 3, 3)


if __name__ == "__main__":

    # filename = 'strict_pref_example.pkl'
    # with open(filename, 'rb') as infile:
    #     dfa = pickle.load(infile)
    dfa_state_dict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}

    b = Action('b')
    c = Action('c')  # set as an globally unsafe obstacle
    nb = Action('!b')
    nc = Action('!c')
    nbnc = Action('!b,!c')
    whole = Action('1')
    # sinks = Action('sink')  # set of all 0 energy states

    dfa = DFA(0, [b, c, nb, nc, whole])  # sink state here means energy used up outside of the station

    dfa.set_final(1)
    dfa.set_final(2)
    dfa.set_final(0)

    dfa.pref_labeling(1, 'X1')  # visiting b
    dfa.pref_labeling(2, 'X2')  # visiting c
    dfa.pref_labeling(0, 'X3')  # visiting b & visiting c in any order
    dfa.pref_labeling(3, 'X3')  # not visiting any places

    dfa.add_pref_trans('X1', 'X2')  # preference edge X2 is more preferred

    dfa.inv_pref_labels = inv_dict(dfa.pref_labels)
    dfa.inv_pref_trans = inv_dict2(dfa.pref_trans)
    dfa.pref_trans['X2'] = []

    # Select one of the following: Constructs Preference DFA for strict or weak formula
    # dfa_add_trans_weak(dfa)
    dfa_add_trans_strict(dfa)

    dfa.toDot("DFA")
    dfa.prune_eff_transition()
    dfa.g_unsafe = 'sink'

    print('done initialize dfa.')

    # brute force labeling
    S = [0, 1, 2]
    S_dict = {0: 'A', 1: 'B', 2: 'C'}  # check google slide page 13

    s_q = dict()    # MDP state -> DFA trans
    q_s = dict()    # DFA trans -> MDP state

    q_s[b.v] = [1]
    q_s[c.v] = [2]
    q_s[nb.v] = [0, 2]
    q_s[nc.v] = [0, 1]
    q_s[nbnc.v] = [0]
    q_s[whole.v] = [0, 1, 2]

    for s in S:
        s_q[s] = []
        if s in q_s[b.v] and b.v not in s_q[s]:
            s_q[s].append(b.v)
        if s in q_s[nb.v] and nb.v not in s_q[s]:
            s_q[s].append(nb.v)
        if s in q_s[c.v] and c.v not in s_q[s]:
            s_q[s].append(c.v)
        if s in q_s[nc.v] and nc.v not in s_q[s]:
            s_q[s].append(nc.v)
        if s in q_s[nbnc.v] and nbnc.v not in s_q[s]:
            s_q[s].append(nbnc.v)
        if s in q_s[whole.v] and whole.v not in s_q[s]:
            s_q[s].append(whole.v)

    # mdp = MDP(transition_file="prod_MDP_gridworld.pkl")
    mdp = MDP()
    mdp.set_S(q_s[whole.v])

    mdp.add_trans_P(0, 'l', 1, 0.5)  # l, r matches p, q in the draft actions
    mdp.add_trans_P(0, 'l', 2, 0.5)
    mdp.add_trans_P(0, 'r', 2, 1.0)

    mdp.add_trans_P(1, 'l', 1, 1.0)
    mdp.add_trans_P(1, 'r', 1, 1.0)

    mdp.add_trans_P(2, 'l', 1, 1.0)
    mdp.add_trans_P(2, 'r', 2, 1.0)

    mdp.set_L(s_q)      # L: labeling function (L: S -> Q), e.g. self.L[(2, 3)] = 'g1'
    mdp.set_Exp(q_s)    # L^-1: inverse of L (L: Q -> S), e.g. self.Exp['g1'] = (2, 3)

    curve = dict()
    result = mdp.product(dfa, mdp, flag='toy')
    result.plotKey = False

    with open("prod_MDP_verifying_example_strict.pkl", 'wb') as pkl_file: #pkl
        pickle.dump(result, pkl_file)







