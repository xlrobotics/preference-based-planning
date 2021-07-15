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
    # PATCH: remove X3
    return v1[0][:2] == v2[0][:2]


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

    # PATCH: remove X3
    vec1 = vec1[:2]
    vec2 = vec2[:2]

    # Logic to check v1 >= v2
    gt_flag = False
    for i in range(len(vec2)):  # Iterate over vec2
        # Get reachable preference states
        pref_state = seq[i]
        if pref_state in mdp.dfa.inv_pref_trans:
            pref_successors = list(mdp.dfa.inv_pref_trans[pref_state])
        else:
            pref_successors = []

        if vec2[i] == vec1[i] == 0 or vec2[i] == vec1[i] == 1 or (vec2[i] == 0 and vec1[i] == 1):
            continue

        # else: vec2[i] == 1 and vec1[i] == 0 --> means there must a higher preferred vec1[j] = 1.
        if not any(vec1[seq.index(succ)] == 1 for succ in pref_successors):
            return False
        gt_flag = True

    # Return gt_flag after Condition 1 is checked for all possible cases in the loop.
    return gt_flag


def merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def construct_improvement_mdp(mdp):
    # Construct state set
    improved_states = {s + (IMPROVED, ) for s in mdp.S}
    states = mdp.S + list(improved_states)

    # Construct edge set
    ndict = dict()
    for state, act in tqdm(mdp.P):
        # Main algorithm
        successors = mdp.P[(state, act)].keys()
        for succ_state in successors:
            # Bookkeeping. Add keys to new dictionary ndict.
            if not (state, act) in ndict:
                ndict[(state, act)] = dict()

            if not (state + (IMPROVED, ), act) in ndict:
                ndict[(state + (IMPROVED, ), act)] = dict()

            # Transition function (Definition 11) from paper.
            if is_gt(mdp, vector_value(mdp, succ_state), vector_value(mdp, state)):
                prob = mdp.P[(state, act)][succ_state]
                ndict[(state, act)].update({succ_state + (IMPROVED, ): prob})
                ndict[(state + (IMPROVED, ), act)].update({(succ_state + (IMPROVED, )): prob})

            elif is_eq(mdp, vector_value(mdp, succ_state), vector_value(mdp, state)):
                prob = mdp.P[(state, act)][succ_state]
                ndict[(state, act)].update({succ_state: prob})
                ndict[(state + (IMPROVED, ), act)].update({succ_state: prob})

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
            improved_state = state + (IMPROVED, )
            temp[improved_state] = dcp(imdp.vector_V[pref_node][state])
        imdp.vector_V[pref_node] = merge(temp, imdp.vector_V[pref_node])
        # pass

    imdp.dfa = dcp(mdp.dfa)
    imdp.ASF = dcp(mdp.ASF)

    imdp.ASR = dcp(mdp.ASR)  # double imdp ASW region with improvement tag (done)
    for pref_node in imdp.ASR:
        temp = set()
        for state in imdp.ASR[pref_node]:
            improved_state = state + (IMPROVED, )
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
        if len(v) == 3:
            level[v] = 0
        else:
            level[v] = float("inf")

    # Initialize queue and visited set
    visited = set()
    frontier = [v for v in imdp.S if len(v) == 3]
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
            if level[u] == float("inf") and len(u) != 3:
                level[u] = level[v] + 1
                frontier.append(u)

                # print(f"Visiting: {v}, Add {u} to frontier.")

            if level[v] < level[u] != float("inf"):
                spi[u].add(a)
                spi[u + (IMPROVED,)].add(a)
                # print(f"Visiting: {v}, spi[{u}]={spi[u]}")

    print(f"Generate SPI strategy for States from which Positively Improving Actions do NOT exist.")
    for v in imdp.S:
        if len(spi[v]) == 0:
            spi[v] = imdp.enabled_actions(v)
            if len(v) != 3:
                spi[v + (IMPROVED,)] = imdp.enabled_actions(v)

    return spi


if __name__ == '__main__':
    with open("prod_MDP_verifying_example_strict.pkl", 'rb') as pkl_file:  # pkl
        mdp = pickle.load(pkl_file)

    print()
    print("DEBUG: Vector Values")
    dfa_state_dict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    S_dict = {0: 'A', 1: 'B', 2: 'C'}  # check google slide page 13
    for state in mdp.S:
        print(f"Val({(S_dict[state[0]], dfa_state_dict[state[1]])} = {vector_value(mdp, state)[0]}")

    # Construct improvement MDP
    imdp = construct_improvement_mdp(mdp)
    print()
    print("DEBUG: Improvement MDP transition function")
    for (state, act), trans_dict in imdp.P.items():
        for next_state, prob in trans_dict.items():
            print(f"{(S_dict[state[0]], dfa_state_dict[state[1]]) + state[2:]} -- {act}, {prob} --> "
                  f"{(S_dict[next_state[0]], dfa_state_dict[next_state[1]]) + next_state[2:]}")

    print()
    print("DEBUG: Comparing vector values")
    u = (2, 2)
    v = (1, 2)
    print(f"Val({(S_dict[u[0]], dfa_state_dict[u[1]])} > Val({(S_dict[v[0]], dfa_state_dict[v[1]])} = "
          f"{is_gt(mdp, vector_value(mdp, u), vector_value(mdp, v))}")

    # Construction of SPI strategy
    # Remark. pi((x, y)) -> x: 0: A, 1: B, 2: C and y: (0,0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3.
    pi = spi_strategy(imdp)
    print()
    print("DEBUG: SPI strategy")
    for state, strategy in pi.items():
        print(f"{(S_dict[state[0]], dfa_state_dict[state[1]]) + state[2:]}: {strategy}")

    with open("PI_prod_MDP_verifying_example.pkl", 'wb') as pkl_file:  # pkl
        pickle.dump(pi, pkl_file)

    pass