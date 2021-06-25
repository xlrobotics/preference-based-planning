import sys

import yaml
import pickle
import json
from tqdm import tqdm
from MDP_learner_gridworld_cleaned import MDP
from copy import deepcopy as dcp

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
    for i in range(len(vec2)):  # Iterate over vec2
        # Get reachable preference states
        pref_state = seq[i]
        if pref_state in mdp.dfa.pref_trans:
            pref_successors = list(mdp.dfa.pref_trans[pref_state])
        else:
            pref_successors = []

        for succ in pref_successors:
            j = seq.index(succ)
            if vec1[j] < vec2[j]:  # Condition 1: for all j, vec1[j] >= vec2[j]
                return False

            if vec1[j] > vec2[i]:  # Condition 2: there exists j s.t. vec1[j] > vec2[i]
                gt_flag = True

    # Return gt_flag after Condition 1 is checked for all possible cases in the loop.
    return gt_flag


def merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def construct_improvement_mdp(mdp):
    # Construct state set
    states = mdp.S + [(s, IMPROVED) for s in mdp.S]

    # Construct edge set
    ndict = dict()
    for state, act in tqdm(mdp.P, desc="Construct IMDP Edges."):
        # Main algorithm
        successors = mdp.P[(state, act)].keys()
        for succ_state in successors:
            # Bookkeeping. Add keys to new dictionary ndict.
            if not (state, act) in ndict:
                ndict[(state, act)] = dict()

            if not ((state, act), IMPROVED) in ndict:
                ndict[((state, act), IMPROVED)] = dict()

            # Xuan said 'fail' state is for debugging.
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
                pass  # Do not add such an edge.

    # Copy any relevant attributes from original MDP to improvement MDP. (done)
    imdp = MDP()
    imdp.S = states
    imdp.A = mdp.A
    imdp.P = ndict

    imdp.vector_V = dcp(mdp.vector_V)  # double imdp v vector with improvement tag (done)
    for pref_node in imdp.vector_V:
        temp = {}
        for state in imdp.vector_V[pref_node]:
            improved_state = (state, IMPROVED)
            temp[improved_state] = dcp(imdp.vector_V[pref_node][state])
        imdp.vector_V[pref_node] = merge(temp, imdp.vector_V[pref_node])
        # pass

    imdp.dfa = dcp(mdp.dfa)
    imdp.ASF = dcp(mdp.ASF)

    imdp.ASR = dcp(mdp.ASR)  # double imdp ASW region with improvement tag (done)
    for pref_node in imdp.ASR:
        temp = set()
        for state in imdp.ASR[pref_node]:
            improved_state = (state, IMPROVED)
            temp.add(improved_state)
        imdp.ASR[pref_node] = imdp.ASR[pref_node].union(temp)

    imdp.mdp = mdp

    return imdp


def spi_strategy2(imdp, asw_strategy):
    """
    Alg. 1 from draft.
    imdp: improvement mdp.
    asw_strategy: dict of almost-sure strategies for each Xi \in \mathcal{X}.
        Structure of asw_strategy: {
                                        "X1": {"v1": {a1, a2}, "v2": {a1}...},
                                        "X2": {"v1": {a1, a2}, "v2": {a1}...},
                                    }
    """
    normal_states = [v for v in imdp.S if v[1] != IMPROVED]
    improved_states = [v for v in imdp.S if v[1] == IMPROVED]

    visited = list()
    queue = [v for v in improved_states]  # Helps identify which states have +ve probability of improvement.
    level = {v: 0 for v in improved_states}  # Needed to construct SPI strategy.
    # FIXME: level with only improved_states only, might come across normal states later returned by the successors
    pi = dict()

    # Initialize SPI strategy for all states in imdp.
    print(f"starting loop 1")
    for v in tqdm(imdp.S):
        # Assuming imdp construction assigns same vector values to v and (v, IMPROVED)
        val, seq = vector_value(imdp, v)  # Copy vector value information to imdp. (duplicated, done)
        if 1 in val:
            idx = val.index(1)  # Choice of ASW strategy is arbitrary.
            # if type(v[1]) == str:
            #     print(v, v in normal_states)
            if type(v[1]) != str and v in normal_states: #FIXME: add type(v[1]) != str to avoid value error
                pi[v] = asw_strategy[seq[idx]][v]
            else:
                pi[v] = asw_strategy[seq[idx]][v[0]]
        else:
            pi[v] = set()

    # Iteratively synthesize SPI strategy
    print(f"starting while loop")
    while len(queue) > 0:
        print(len(queue), len(visited), "/", len(normal_states)*2)
        v = queue.pop(0)    # FIXME: only contains improved states? (initially, yes. while loop may add other states.)
        visited.append(v)
        # pred = set(imdp.predecessors(v)) - set.union(set(queue), set(visited))
        pred = imdp.predecessors(v) - set.union(set(queue), set(visited))  # predecessor function needed. (done) set returned
        # FIXME: if pred is empty, this reduction will return empty set, is that correct? (yes.)

        # Each predecessor has positive probability to reach improved_states. Add it to queue.
        for p in pred:
            # Book keeping for levels
            k = level[v]
            level[p] = k + 1
            # If predecessor is not visited or queued, then add it to queue.
            if p not in queue and p not in visited:
                queue.append(p)

        # Design SPI strategy at current state: v.
        if type(v[1]) != str and v in normal_states: #FIXME: add type(v[1]) != str to avoid value error
            spi_actions = set()

            # An enabled action a at state v is SPI action iff all successors satisfy at least one of 2 conditions:
            #   (i) level of successor s is smaller than v
            #   (ii) successor s and v are ASW in same objective Xi, for which vector-value(v) = 1.

            for a in imdp.enabled_actions(v):  # FIXME: enabled_actions function needed.
                successors = imdp.successors(v, a)  # FIXME: successors function needed.
                for s in successors:
                    vec_s, _ = vector_value(imdp, s)
                    vec_v, _ = vector_value(imdp, v)
                    cond1 = True in [vec_s[i] == vec_v[i] for i in range(len(vec_v))]
                    cond2 = level[s] < level[v]
                    if cond1 or cond2:
                        spi_actions.add(a)

            # If spi actions are available, update them for both v and (v, IMPROVED) states.
            # Else, pi[v] = pi[(v, IMPROVED)] = ASW(Xi) ... where ASW(Xi) is already initialized above.
            if len(spi_actions) > 0:
                pi[v] = spi_actions
                pi[(v, IMPROVED)] = spi_actions

    return pi


def spi_strategy3(imdp, asw_strategy):
    # Separate MDP states and improved-labeled states.
    mdp_states = {v for v in imdp.S if v[1] != IMPROVED}
    imp_states = {v for v in imdp.S if v[1] == IMPROVED}
    print(f"mdp_states={len(mdp_states)}, imp_states={len(imp_states)}")

    # Initialize local variables
    visited = set()
    frontier = list(dcp(imp_states))
    level = {v: float("inf") for v in mdp_states}
    level.update({v: 0 for v in imp_states})
    pi = {v: set() for v in set.union(mdp_states, imp_states)}
    print(f"visited={len(visited)}, frontier={len(frontier)}, level={len(level.keys())}")

    while len(frontier) > 0:
        # Select a state for visiting in this loop.
        v = frontier.pop(0)
        visited.add(v)

        # Get predecessors of state. Update level of each predecessor.
        pred = [p for p in imdp.predecessors(v) if (p not in frontier and p not in visited)]
        for p in pred:
            level[p] = level[v] + 1
            frontier.append(p)

        # If state is MDP state, process it further.
        if v in mdp_states:
            # Initialize variables to check conditions for identifying SPI actions
            pos_imp_actions = set()
            flag_cond1 = False
            val_v, seq = vector_value(imdp, v)

            # Identify SPI actions
            for a in imdp.enabled_actions(v):
                succ = imdp.successors(v, a)
                for s in succ:
                    val_s, _ = vector_value(imdp, s)

                    if level[s] < level[v]:
                        flag_cond1 = True
                        pos_imp_actions.add(a)
                    elif any([val_s[i] == val_v[i] for i in range(len(val_s))]):
                        pos_imp_actions.add(a)
                    else:
                        pos_imp_actions.remove(a)
                        flag_cond1 = False
                        break       # continue exploring next action

            if len(pos_imp_actions) > 0 and flag_cond1:
                pi[v] = pos_imp_actions
            elif any([val_v[i] == 1 for i in range(len(val_v))]):
                idx = val_v.index(1)
                pi[v] = asw_strategy[seq[idx]][v]
            else:
                pi[v] = set()

        # Define SPI strategy for "improve"-labeled states
        for v, imp in imp_states:
            pi[(v, imp)] = pi[v]

        # Return SPI strategy
        return pi


def spi_strategy(imdp):
    # Initialize strategy, level dictionaries
    spi = dict()
    level = dict()
    for v in imdp.S:
        spi[v] = set()
        if v[1] == IMPROVED:
            level[v] = 0
        else:
            level[v] = float("inf")

    # Initialize queue and visited set
    visited = set()
    frontier = [v for v in imdp.S if v[1] == IMPROVED]

    print(f"Starting BFS-visitation: |frontier|={len(frontier)}, |visited|={len(visited)}, |V|={len(imdp.S)}")
    pbar = tqdm(imdp.S)
    # while len(frontier) > 0:
    for _ in pbar:
        pbar.set_description(f"|frontier|={len(frontier)}, |visited|={len(visited)}, |V|={len(imdp.S)}")
        v = frontier.pop(0)
        visited.add(v)
        pre_v = imdp.predecessors(v)
        if len(pre_v) > 0:
            print(f"Visiting: {v}, pred: {pre_v}")

        for u, a in pre_v:
            if level[u] == float("inf") and u[1] != IMPROVED:
                level[u] = level[v] + 1
                frontier.append(u)
                print(f"Visiting: {v}, Add {u} to frontier.")

            if level[v] < level[u] != float("inf"):
                spi[u].add(a)
                spi[(u, IMPROVED)].add(a)
                print(f"Visiting: {v}, spi[{u}]={spi[u]}")

    print(f"Generate SPI strategy for States from which Positively Improving Actions do NOT exist.")
    for v in imdp.S:
        if len(spi[v]) == 0:
            spi[v] = imdp.enabled_actions(v)
            spi[(v, IMPROVED)] = imdp.enabled_actions(v)

    return spi


def calc_asw_pi(mdp):
    """
    Structure of asw_strategy: {
                                    "X1": {"v1": {a1, a2}, "v2": {a1}...},
                                    "X2": {"v1": {a1, a2}, "v2": {a1}...},
                                }
    """
    asw_strategy = {pref_st: dict() for pref_st in mdp.ASR.keys()}
    for pref_st in asw_strategy:
        for v in mdp.ASR[pref_st]:
            for a in mdp.enabled_actions(v):
                successors = mdp.successors(v, a)
                if len(successors):
                    if set.issubset(successors, set(mdp.ASR[pref_st])):
                        if v not in asw_strategy[pref_st]:
                            asw_strategy[pref_st][v] = set()
                        asw_strategy[pref_st][v].add(a)
    return asw_strategy


def prune_mdp(mdp):
    pruned_P = dict()
    for u, a in mdp.P.keys():
        if all(val == 0 for val in mdp.P[(u, a)].values()):
            continue
        pruned_P[(u, a)] = mdp.P[(u, a)]
    mdp.P = pruned_P
    return mdp



if __name__ == '__main__':
    # mdp = None
    with open("prod_MDP_gridworld.pkl", 'rb') as pkl_file:  # pkl
        mdp = pickle.load(pkl_file)

    mdp = prune_mdp(mdp)
    print(mdp)
    print(vector_value(mdp, ((0, 0, 0, 0, 0), 14)))
    imdp = construct_improvement_mdp(mdp)
    print("okay")

    # asw_strategy = calc_asw_pi(imdp)
    # print("asw_strategy done.")
    # pi = spi_strategy(imdp, asw_strategy)
    pi = spi_strategy(imdp)
    print("spi_strategy done.")

    # with open("PI_prod_MDP_gridworld.pkl", 'wb') as pkl_file: #pkl
    #     pickle.dump(pi, pkl_file)