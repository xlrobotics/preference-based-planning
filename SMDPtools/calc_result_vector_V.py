import yaml
import pickle
import json
from MDP_learner_backup import MDP


IMPROVED = "improved"


def vector_value(mdp, state):
    vector_s = []
    pref_seq = []
    for pref_state in mdp.vector_V:
        pref_seq.append(pref_state)
        vector_s.append(mdp.vector_V[pref_state][state])

    return tuple(vector_s), tuple(pref_seq)


def is_lt(mdp, v1, v2):
    """ Checks if a vector value v1 is less than v2. """
    return not is_gt(mdp, v1, v2) and not is_eq(mdp, v1, v2)


def is_eq(mdp, v1, v2):
    """ Checks equality of vector value v1 is less than v2. """
    return v1[0] == v2[0]


def is_gt(mdp, v1, v2):
    """
    Checks if a vector value v1 is greater than v2.
    Assumption: Vectors v1, v2 are consistent. That is, there does NOT exist any 0 < i < j < |X| s.t.
        Xj is comparable to Xi and vec[i] = vec[j] = 1.
    """
    # Get vector values
    vec1, seq1 = v1
    vec2, seq2 = v2
    assert seq1 == seq2
    seq = seq1

    # Logic to check v1 >= v2
    gt_flag = False
    for i in range(len(vec2)):          # Iterate over vec2
        # Get reachable preference states
        pref_state = seq[i]
        if pref_state in mdp.dfa.pref_trans:
            pref_successors = list(mdp.dfa.pref_trans[pref_state])
        else:
            pref_successors = []

        for succ in pref_successors:
            j = seq.index(succ)
            if vec1[j] < vec2[j]:     # Condition 1: for all j, vec1[j] >= vec2[j]
                return False

            if vec1[j] > vec2[i]:   # Condition 2: there exists j s.t. vec1[j] > vec2[i]
                gt_flag = True

    # Return gt_flag after Condition 1 is checked for all possible cases in the loop.
    return gt_flag


def construct_improvement_mdp(mdp):
    # Construct state set
    states = mdp.S + [(s, IMPROVED) for s in mdp.S]

    # Construct edge set
    ndict = dict()
    for state, act in mdp.P:
        # Main algorithm
        successors = mdp.P[(state, act)].keys()
        for succ_state in successors:
            # Bookkeeping. Add keys to new dictionary ndict.
            if not (state, act) in ndict:
                ndict[(state, act)] = dict()

            if not ((state, act), IMPROVED) in ndict:
                ndict[((state, act), IMPROVED)] = dict()

            # FIXME: Xuan said 'fail' state is for debugging.
            if isinstance(state, str) or isinstance(succ_state, str):
                continue

            # Transition function (Definition 11) from paper.
            if is_gt(mdp, vector_value(mdp, succ_state), vector_value(mdp, state)):
                prob = mdp.P[(state, act)][succ_state]
                ndict[(state, act)].update({(succ_state, IMPROVED): prob})
                ndict[((state, act), IMPROVED)].update({(succ_state, IMPROVED): prob})

            elif is_eq(mdp, vector_value(mdp, succ_state), vector_value(mdp, state)):
                prob = mdp.P[(state, act)][succ_state]
                ndict[(state, act)].update({succ_state: prob})
                ndict[((state, act), IMPROVED)].update({succ_state: prob})

            else:
                pass    # Do not add such an edge.

    # TODO: Copy any relevant attributes from original MDP to improvement MDP.
    imdp = MDP()
    imdp.S = states
    imdp.P = ndict

    return imdp


if __name__ == '__main__':
    # with open("prod_MDP_gridworld.pkl", 'rb') as pkl_file:  # pkl
    #     result = pickle.load(pkl_file)

    # counter = 0
    # total = len(result.S)
    # for s in result.S:
    #     print("processing key:", s, counter, "/", total, " done")
    #     result.init_ASR_V(s)
    #     counter += 1

    # with open("prod_MDP_gridworld_updated.pkl", 'wb') as pkl_file: #pkl
    #     pickle.dump(result, pkl_file)

    mdp = None
    with open("prod_MDP_gridworld_updated.pkl", 'rb') as pkl_file:  # pkl
        mdp = pickle.load(pkl_file)
    
    print(mdp)
    print(vector_value(mdp, ((0, 0, 0, 0, 0), 14)))
    imdp = construct_improvement_mdp(mdp)
    print("okay")


