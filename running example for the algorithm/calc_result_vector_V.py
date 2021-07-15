import pickle
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
    for state, act in tqdm(mdp.P):
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
    # frontier = frontier[1000:1400]    # debug

    # print(f"Starting BFS-visitation: |frontier|={len(frontier)}, |visited|={len(visited)}, |V|={len(imdp.S)}")
    pbar = tqdm(imdp.S)
    # while len(frontier) > 0:
    for _ in pbar:
        # pbar.update()
        if len(frontier) == 0:
            break

        pbar.set_description(f"|frontier|={len(frontier)}, |visited|={len(visited)}, |V|={len(imdp.S)}")

        v = frontier.pop(0)
        visited.add(v)
        pre_v = imdp.predecessors(v)
        # if len(pre_v) > 0:
        #     print(f"Visiting: {v}, pred: {pre_v}")

        for u, a in pre_v:
            if level[u] == float("inf") and u[1] != IMPROVED:
                level[u] = level[v] + 1
                frontier.append(u)

                # print(f"Visiting: {v}, Add {u} to frontier.")

            if level[v] < level[u] != float("inf"):
                spi[u].add(a)
                spi[(u, IMPROVED)].add(a)
                # print(f"Visiting: {v}, spi[{u}]={spi[u]}")

    print(f"Generate SPI strategy for States from which Positively Improving Actions do NOT exist.")
    for v in imdp.S:
        if len(spi[v]) == 0:
            spi[v] = imdp.enabled_actions(v)
            spi[(v, IMPROVED)] = imdp.enabled_actions(v)

    return spi


if __name__ == '__main__':
    with open("prod_MDP_verifying_example_strict.pkl", 'rb') as pkl_file:  # pkl
        mdp = pickle.load(pkl_file)

    # Construct improvement MDP
    imdp = construct_improvement_mdp(mdp)
    print("Construction of improvement MDP: SUCCESS!")

    # Construction of SPI strategy
    # Remark. pi((x, y)) -> x: 0: A, 1: B, 2: C and y: (0,0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3.
    pi = spi_strategy(imdp)
    print("Construction of SPI strategy: SUCCESS!")

    with open("PI_prod_MDP_verifying_example.pkl", 'wb') as pkl_file:  # pkl
        pickle.dump(pi, pkl_file)

    pass