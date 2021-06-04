# %matplotlib inline
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
# %matplotlib.inline
# import seaborn as sns
import pickle
# from pandas import *
from copy import deepcopy as dcp

# # plotly dependencies
# import plotly.plotly as py
# import plotly.graph_objs as go
# import plotly
# from plotly import tools
# plotly.tools.set_credentials_file(username='SeanLau', api_key='d1fuOtEPoJU7V6GhntKN')

'''
Comment tags:
    Init:       used for initializing the object parameters
    Tool:       inner function called by API, not transparent to the user
    API:        can be called by the user
    VIS:        visualization
    DISCARD:    on developing or discarded functions in old version
'''
# Tool: geometric series function, called by self.transition_matrix()
def GeoSeries(P, it):
    sum = np.identity(len(P)) + dcp(P)  # + np.identity(len(P))
    for k in range(2, it + 1):
        sum += np.linalg.matrix_power(dcp(P), k)
    return sum

# API: calculating transition matrix for an option
def transition_matrix_(S, goal, unsafe, interruptions, gamma, P, Pi, V, row, col): # write the correct transition probability according to the policy
    '''
    :param S: grid state
    :param goal: goal state
    :param unsafe: obstacles
    :param interruptions: obstacles
    :param gamma: discounting factor, usually 0.9
    :param P: stocasticity of the grid world, P(s'|s, a)
    :param Pi:  generated policy     pi(a|s)
    :param V:   value funciton  V(s)
    :param row: graph height
    :param col:  graph width
    :return:    final: dictionary for stationary distribution;  H: entropy sum over infinite horizon
    '''
    H = {}
    tau = 1.0
    # row, col = 6, 8
    size = row * col + 1  # l * w + sink
    PP = np.zeros((size, size))
    Vn = np.zeros(size)

    HH = np.zeros(size)
    PP[0, 0] = 1.0
    HH[0] = 0.0
    Vn[0] = 0.0
    for state in S:
        s = tuple(state)
        n = (s[0] - 1) * col + s[1]  # (1,1) = 1, (2,1)=9
        Vn[n] = V[s]
        if s not in goal and s not in unsafe and s not in interruptions:
            PP[n, 0] = 1 - gamma

            for a in Pi[s]:

                HH[n] += -tau * Pi[s][a] * np.log(Pi[s][a])

                for nb in P[s, a]:
                    n_nb = (nb[0] - 1) * col + nb[1]
                    PP[n, n_nb] += gamma * Pi[s][a] * P[s, a][nb]
        else:
            PP[n, n] = 1.0
            HH[n] = 0.0

    sums = GeoSeries(
        dcp(PP),
        100)

    sum_entropy = dcp(sums).dot(HH)

    final = {}

    result = np.linalg.matrix_power(dcp(PP), 100)

    # # print  result
    for state in S:
        s = tuple(state)
        n = (s[0] - 1) * col + s[1]

        final[s] = {}
        line = []
        for state_ in S:
            s_ = tuple(state_)
            n_ = (s_[0] - 1) * col + s_[1]
            line.append(result[n, n_])
        for g in goal:
            ng = (g[0] - 1) * col + g[1]
            # final[s][g] = result[n, ng]/sum(line)
            final[s][g] = result[n, ng]

        H[s] = sum_entropy[n]

    return final, H


class MDP:
    #static properties
    INFTY = 1000000000
    def __init__(self, gamma = 0.9, epsilon = 0.15, gain = 100.0, tau = 1.0):
        '''
        :param gamma: Discounting factor
        :param epsilon: P[s'=f(s, a)|s, a] = 1 - epsilon
        :param gain: Fixed reward value at goal states
        :param tau: Temperature parameter for softmax iteration and composition

        Data structure:
        self.L: labeling function (L: S -> Q),          e.g. self.L[(2, 3)] = 'g1'
        self.Exp: L^-1: inverse of L (L: Q -> S),       e.g. self.Exp['g1'] = (2, 3)
        self.goal, self.T: goal set list,               e.g. self.goal = [(1, 2), (3, 4)], in product MDP where
                                                        e.g. state is S*Q, self.goal = [((1, 2), 1)] = self.T
                                                        e.g. T only serves for product MDP
        <self.unsafe: globally unsafe states>           e.g. see self.goal
        <self.interruptions: temporal unsafe states>    e.g. see self.goal
        self.S: state set list,                         e.g. see self.goal
        <self.originS>                                  e.g. only serves NON product MDP
        self.P: P(s' | s, a) for stochastic MDP,        e.g. self.P[s, a][s']
        self.Pi: Pi(a | s) policy,                      e.g. self.Pi[s][a]
        self.R: R(s, a) Reward function,                e.g. self.R[s][a]
        self.Q: Q(s, a) Q value function,               e.g. self.Q[s][a]
        self.V: V(s) value function,                    e.g. self.V[s], (self.V_[s] stores V[s] in last iteration)
        self.Po: P(s' | s, o) option transition matrix  e.g. self.Po[s][s']
        self.Opt: MDP object, for composed option       e.g. self.Opt[id], id = (('g1', 'g2'), 'disjunction')
        self.AOpt: MDP object, for atomic option        e.g. self.AOpt[id], id = 'g1'
        self.dfa: DFA object
        self.mdp: plain MDP object

        '''

        self.gamma = gamma # discounting factor
        self.epsilon = epsilon # transition probability, set as constant here
        self.tau = tau
        self.constReward = gain
        self.unsafe = {}

        self.ID = 'root' # MDP id to identify the layer of an MDP, set as 'root' if it's original MDP,

        self.S = [] # state space e.g. [[0,0],[0,1]...]
        self.init_S = []
        self.originS = []
        self.wall_cord = [] # wall state space, e.g. if state space is [[1,1]], wall is [[1,2], [2,1], [1,0], [0,1]]
        self.L = {} # labeling function
        self.Exp = {} # another way to store labeling pairs
        self.goal = [] # goal state set
        self.interruptions = [] # unsafe state set
        self.gridSize = []
        self.target_coords = []
        self.station_coords = []
        self.full_tank = 12
        self.success = []

        self.A = {"N":(1, 0), "S":(-1, 0), "W":(0, -1), "E":(0, 1), "stay":(0, 0)} # actions for grid world
        self.A_simple = ['l', 'r'] # 0 means stay
        self.simple_trans = {}

        self.V, self.V_ = {}, {} # value function and V_ denotes memory of the value function in last iteration
        self.init_V, self.init_V_ = {}, {} # store initial value function

        self.Q, self.Q_ = {}, {} # state action value function
        self.T = {} # terminal states
        self.Po = {} # transition matrix

        self.R = {} # reward function
        self.init_R = {} # initial reward function

        self.P = {} # 1 step transition probabilities
        self.H = {}
        self.ASF = {}
        self.ASR = {}

        self.Pi, self.Pi_ = {}, {} # policy and memory of last iteration policy
        self.Pi_opt, self.Pi_opt_ = {}, {}  # policy and memory of last iteration policy
        self.Opt = {} # options library
        self.AOpt = {} # atomic options without production

        self.dfa = None
        self.mdp = None
        self.plotKey = False
        self.svi_plot_key = False

        self.action_special = None
        self.hybrid_special = None
        self.svi_record = 0

    # Init: set labeling function
    def set_L(self, labels):
        self.L = labels

    # Init: set
    def set_Exp(self, exp):
        self.Exp = exp

    def set_targets(self, targets):
        self.target_coords = targets[:]

    def set_stations(self, stations):
        self.station_coords = stations[:]

    def set_full_tank(self, full_tank):
        self.full_tank = full_tank

    # Tool: a small math tool for the state space cross product
    def crossproduct(self, a, b):
        cross_states = [(tuple(y), x) for x in a for y in b]
        print(cross_states)
        return cross_states

    def crossproduct2(self, a, b):
        cross_states = [(y, x) for x in a for y in b]
        print(cross_states)
        return cross_states

    def simple_composition(self, Vlist, ctype):
        result = {}
        tau = 1.0

        if ctype == 'disjunction':
            for state in self.S:
                s = tuple(state)
                result[s] = tau*np.log(sum([np.exp(V[s]/tau) for V in Vlist])) #/ len(Opt_list)

        if ctype == 'conjunction':
            tau *= -1.0
            for state in self.S:
                s = tuple(state)
                result[s] = tau * np.log(sum([np.exp(V[s] / tau) for V in Vlist]))  # / len(Opt_list)

        self.V = dcp(result)

    def bubble(self, target):
        for round in range(len(target)):
            for i in range(len(target)):
                if ('&' in target[i] or '|' in target[i]) and i < len(target) - 1:
                    target[i], target[i+1] = target[i+1], target[i]
        return target

    # secondary API, called by def pos_reachability, used to return next reachable states with probability > 0
    def nextStateSet(self, s, a):
        next_set = []
        if (s, a) not in self.P:
            return set([])

        for key in self.P[s, a].keys():
            if type(key) != str and self.P[s, a][key] > 0 and key not in next_set:
                next_set.append(key)
        return set(next_set)

    # secondary API: called by function def product(), return the ASR of the product mdp
    def pos_reachability(self, F):
        # almost sure reaching the set F.
        # alg: compute the set of states from which the agent can stay with probability one, and has a positive probability of reaching F (in multiple steps).
        X0 = set(self.S)
        X0.remove((0, 1))
        X0.remove((1, 1))
        X0.remove((4, 1))
        X0.remove((5, 1))
        X0.remove((6, 1))
        X0.remove((7, 1))
        Y0 = set(F)
        while True:
            Y0 = set(F)
            Y1 = Y0
            while True:
                for s in X0:
                    for a in self.A_simple: # TODO: action_special is for the toy, for gridworld need to modify it
                        reachableStates = self.nextStateSet(s, a)  # TODO: computing the next states that can be reached with probability >0.

                        # if reachableStates.issubset(X0) and reachableStates.intersect(Y1) != set([]):
                        if reachableStates <= X0 and reachableStates and Y1 is not set([]):
                            Y1.add(s)
                            break
                if Y1 == Y0:  # no state is added, reached the inner fixed point.
                    break
            if X0 != Y1: # shrinking X0
                X0 = Y1
            else:  # reach the outer fixed point.
                break  # outer loop complete.
        return X0

    # API: DFA * MDP product
    def product(self, dfa, mdp):
        result = MDP()
        result.Exp = dcp(mdp.Exp)
        result.L = self.L

        filtered_S = []
        for s in mdp.S:  #new_S
            if mdp.L[s][0] != 'failure' and s not in filtered_S: # new_wall
                filtered_S.append(s)
            # filtered_S.append(s) # no need to filter here

        result.originS = dcp(filtered_S)

        result.init_S = self.crossproduct2(list([dfa.initial_state]), mdp.Exp['phi'])

        new_success = self.crossproduct2(list(dfa.final_states-dfa.sink_states), filtered_S)
        new_fail = self.crossproduct2(list(dfa.sink_states), filtered_S)

        # calculate F states for each pref node:
        for key in dfa.inv_pref_labels.keys():
            if key not in result.ASF:
                # result.ASF[key] = result.crossproduct2(dfa.inv_pref_labels[key], filtered_S)
                result.ASF[key] = [(3, 1)]

        result.success = new_success[:]

        new_P, new_R = {}, {}
        new_V, new_V_ = {}, {}

        true_new_s = []

        sink = "pref node"
        fail = "fail"

        total = len(mdp.P.keys())
        counter = 0
        for p in mdp.P.keys():
            counter += 1
            print("processing key:", p, counter,"/", total," done")

            if p[0]==1 and p[2]==0:
                print(p)
                pass

            for q in dfa.states:
                if p[0]==1 and p[2]==0 and q == 1:
                    pass

                if q not in dfa.final_states and mdp.L[p[0]][0] in dfa.final_transitions: # initial state not allowed with final states of dfa
                    continue

                if q in dfa.sink_states and mdp.L[p[0]][0] != 'failure': # cannot get out after entering sink state
                    continue

                if mdp.L[p[0]][0] == 'failure' and q not in dfa.sink_states:
                    continue

                new_s = (p[0], q)
                new_a = p[1]
                if (new_s, new_a) not in new_P:
                    new_P[new_s, new_a] = {}

                new_V[new_s] = 0
                new_V_[new_s] = 0
                new_R[new_s, new_a] = 0

                for label in mdp.L[p[2]]:
                    if tuple([label, q]) in dfa.state_transitions.keys():
                        q_ = dfa.state_transitions[label, q] # p[0]

                        new_s_ = (p[2], q_)
                        if q == q_: # dfa state equals to transition state after reaching the labelled state in mdp
                            if new_s_ not in new_P[new_s, new_a]:
                                new_P[new_s, new_a][new_s_] = mdp.P[p]
                        else:
                            new_P[new_s, new_a][new_s_] = mdp.P[p]


                        if tuple(new_s_) not in true_new_s:
                            true_new_s.append(tuple(new_s_))

                        if new_s_ in new_success and new_P[new_s, new_a][new_s_]>0 and new_s_[1] in dfa.pref_labels: # tagging, debug step, delete after check correct
                            new_P[new_s, new_a][sink] = dfa.pref_labels[new_s_[1]]

                        if new_s_ in new_fail and new_P[new_s, new_a][new_s_]>0: # tagging, debug step, delete after check correct
                            new_P[new_s, new_a][fail] = 1

                if new_s not in true_new_s:
                    true_new_s.append(tuple(new_s))


        result.set_S(true_new_s)

        result.P = dcp(new_P)

        result.T = dcp(new_success)

        result.dfa = dcp(dfa)
        result.mdp = dcp(mdp)

        for key in result.ASF.keys():
            if key not in result.ASR:
                result.ASR[key] = result.pos_reachability(result.ASF[key])

        return result

    # Init: preparation for 1 step transition probability generation, not important
    def add_wall(self, inners):
        wall_cords = []
        for state in inners:
            # for action in self.A:
            #     temp = list(np.array(state) + np.array(self.A[action]))
            #     if temp not in inners:
            wall_cords.append(state)
        return wall_cords

    # Init: preparation for 1 step transition probability generation, not important
    def set_WallCord(self, wall_cord):
        for element in wall_cord:
            self.wall_cord.append(element)
            # if element not in self.S:
            #     self.S.append(element)

    # Init: init state space
    def set_S(self, in_s = None):
        if in_s == None:
            for i in range(13):
                for j in range(13):
                    self.S.append([i,j])
        else:
            self.S = dcp(in_s) # initial states
            self.originS = dcp(in_s)

    # Init: set grid size
    def set_Size(self, row, col):
        self.gridSize = [row, col]

    # Init: init goal state set
    def set_goal(self, goal=None, in_g=None):
        if goal!=None and in_g == None:
            self.goal.append(goal)
        if goal == None and in_g!=None:
            self.goal = dcp(in_g)

    # Init: init state value function
    def init_value_function(self):
        for state in self.S:
            s = tuple(state)
            if s not in self.V:
                self.V[s], self.V_[s] = 0.0, 0.0
            if s in self.success:
                self.V[s], self.V_[s] = 100.0, 0.0

    # Init: generating 1 step transition probabilities for 2d grid world
    def set_P(self): # add cloud dynamics to the transition matrix
        self.P = {}
        filtered_S = []
        for s in self.S:  # new_S
            if s not in self.wall_cord:  # new_wall
                filtered_S.append(s)
        self.S = filtered_S  # 0 battery cases should be filtered out from this step

        # cloud dynamics random walk, up, down, stay
        cloud_acts = [-1, 0, 1]

        for state in self.S:
            s = tuple(state)
            current_cloud_poses = []
            current_cloud_poses.append(tuple([s[2], 0]))
            current_cloud_poses.append(tuple([s[3], 1]))

            for a in self.A.keys():
                self.P[s, a, s] = 0.0  # not possible to stay at the same state, since energy always -1

                if s[0:2] in current_cloud_poses: # trapped in clouds
                    next_position = s[0:2]
                else:
                    next_position = np.array(s[0:2]) + np.array(self.A[a])
                    if next_position[0] < 0: next_position[0] = 1  # bounce back when touching the boundary
                    if next_position[0] > 3: next_position[0] = 2
                    if next_position[1] < 0: next_position[1] = 1  # bounce back when touching the boundary
                    if next_position[1] > 3: next_position[1] = 2

                for ca1 in cloud_acts:
                    new_s2 = s[2] + ca1  # cloud bounce back
                    if new_s2 > 3:
                        new_s2 = s[2] - 1
                    if new_s2 < 0:
                        new_s2 = s[2] + 1

                    for ca2 in cloud_acts:
                        new_s3 = s[3] + ca2
                        if new_s3 > 3:
                            new_s3 = s[3] - 1
                        if new_s3 < 0:
                            new_s3 = s[3] + 1

                        temp = tuple(list(next_position) + [new_s2] + [new_s3] + [s[4] - 1])  # enumerate next states
                        temp_ = tuple(list(next_position) + [new_s2] + [new_s3] + [self.full_tank])  # enumerate next states
                        if list(next_position) not in self.station_coords:
                            s_ = temp[:]
                        else:
                            s_ = temp_[:]

                if list(s_) in self.originS: # only based on cloud dynamics
                    self.P[s, a, s_] = 1/3 * 1/3

        return

    def add_trans_P(self, s, a, s_, value):
        self.P[s, a, s_] = value

    def set_P_simple(self): # add cloud dynamics to the transition matrix
        # self.simple_trans[0]
        self.P = {}
        self.P[0, 'l', 5] = 1
        self.P[0, 'l', 4] = 0
        self.P[0, 'r', 4] = 1
        self.P[0, 'r', 5] = 0

        self.P[1, 'l', 3] = 1
        self.P[1, 'l', 0] = 0
        self.P[1, 'r', 0] = 1
        self.P[1, 'r', 3] = 0

        self.P[2, 'l', 0] = 0.6
        self.P[2, 'l', 3] = 0.4
        self.P[2, 'r', 0] = 0.4
        self.P[2, 'r', 3] = 0.6

        self.P[3, 'l', 2] = 1
        self.P[3, 'l', 1] = 0
        self.P[3, 'r', 1] = 1
        self.P[3, 'r', 2] = 0

        self.P[4, 'l', 2] = 0.9
        self.P[4, 'l', 6] = 0.1
        self.P[4, 'r', 2] = 0
        self.P[4, 'r', 6] = 1

        self.P[5, 'l', 1] = 0
        self.P[5, 'l', 6] = 1
        self.P[5, 'r', 6] = 0.1
        self.P[5, 'r', 1] = 0.9

        self.P[6, 'l', 6] = 1
        self.P[6, 'l', 3] = 0
        self.P[6, 'r', 6] = 1
        self.P[6, 'r', 3] = 0

        return

    def trans_P(self):
        temp = dcp(self.P)
        self.P = {}
        for key in temp:
            initial = tuple([key[0], key[1]])
            post = key[2]
            if initial not in self.P:
                self.P[initial] = {}
            self.P[initial][post] = temp[key]

    # Tool: turning dictionary structure to vector
    def Dict2Vec(self, V, S):
        # # # print   S
        v = []
        # # # print   "Vkeys",V.keys()
        for s in S:
            v.append(V[tuple(s)])
        return np.array(v)

    # Tool: sum operator in value iteration algorithm, called by self.SVI()
    def Sigma_(self, s, a, V):
        total = 0.0

        if self.ID != "root":
            if s in self.unsafe:
                return total
            elif s in self.interruptions:
                return total

        for s_ in self.P[s, a].keys():
            total += self.P[s, a][s_] * V[s_] #self.V_[s_]
        return total

    # API: action SVI runner
    def SVI(self, threshold):
        # S, A, P, R, sink, V, V_ = inS, inA, inP, inR, in_sink, inV, inV_
        tau = 1.0

        self.Pi, self.Pi_ = {}, {}

        V_current, V_last = self.Dict2Vec(self.V, self.S), self.Dict2Vec(self.V_, self.S)
        it = 1
        diff = []
        val = []
        pi_diff = []
        special = []
        special2 = []

        diff.append(np.inner(V_current - V_last, V_current - V_last))

        while np.inner(V_current - V_last, V_current - V_last) > threshold or it < 2: # np.inner(V_current - V_last, V_current - V_last) > threshold

            if it > 1 and threshold > 10000:
                # print  "break"
                break

            for s in self.S:
                self.V_[tuple(s)] = self.V[tuple(s)]

                if tuple(s) not in self.Pi: # for softmax
                    self.Pi[tuple(s)] = {}
                if tuple(s) not in self.Q:
                    self.Q[tuple(s)] = {}

            for s in self.S:

                if tuple(s) not in self.T and tuple(s) not in self.interruptions:
                    # # # print   s, self.T
                    max_v, max_a = -0.001, None

                    for a in self.A:
                        if (tuple(s), a) in self.P:

                            if (tuple(s), a) not in self.R:
                                self.R[tuple(s), a] = 0

                            v = self.R[tuple(s), a] + self.gamma * self.Sigma_(tuple(s), a, self.V_)
                            # if self.R[tuple(s), a] > 0:
                                # print  s, a, self.R[tuple(s), a]
                            self.Q[tuple(s)][a] = np.exp(v / tau)  # softmax solution
                            # if self.plotKey:
                            #     # # print   v, self.R[tuple(s), a], self.Sigma_(tuple(s), a)

                            if v > max_v:
                                max_v, max_a = v, a

                    sumQ = sum(self.Q[tuple(s)].values())
                    # print (self.Q[tuple(s)])
                    for choice in self.Q[tuple(s)]:
                        self.Pi[tuple(s)][choice] = self.Q[tuple(s)][choice] / sumQ
                    self.V[tuple(s)] = tau * np.log(sumQ)

            V_current, V_last = self.Dict2Vec(self.V, self.S), self.Dict2Vec(self.V_, self.S)
            diff.append(np.inner(V_current - V_last, V_current - V_last))

            it += 1
        print  ("action iterations:", it)

        # if not self.plotKey:
        #     self.svi_record = special[len(special) - 1]
        #
        #     print  ("action special point: ", special)
        #     print  ("action special point2: ", special2)

        self.action_diff = diff
        self.action_val = val
        self.action_special = special

        return special

    def Hardmax_SVI(self, threshold):
        self.Pi, self.Pi_ = {}, {}
        self.V, self.V_ = dcp(self.init_V), dcp(self.init_V_)
        self.R = dcp(self.init_R)
        self.Q = {}

        V_current, V_last = self.Dict2Vec(self.V, self.S), self.Dict2Vec(self.V_, self.S)
        it = 1
        diff = []
        val = []
        special = []

        diff.append(np.inner(V_current - V_last, V_current - V_last))

        # while np.inner(V_current - V_last,
        #                V_current - V_last) > threshold or it < 2:  # np.inner(V_current - V_last, V_current - V_last) > threshold
        while it < len(self.action_special):
            if it > 1 and threshold > 10000:
                # print  "break"
                break
            # self.plot_map(it)

            if tuple([(3, 3), 0]) in self.V:
                special.append(self.V[(3, 3), 0])

            for s in self.S:
                self.V_[tuple(s)] = self.V[tuple(s)]

                if tuple(s) not in self.Pi:  # for softmax
                    self.Pi[tuple(s)] = {}
                if tuple(s) not in self.Q:
                    self.Q[tuple(s)] = {}

            for s in self.S:

                if tuple(s) not in self.T and tuple(s) not in self.interruptions:
                    # # # print   s, self.T
                    max_v, max_a = -1*self.INFTY, None

                    for a in self.A:
                        if (tuple(s), a) in self.P:

                            if (tuple(s), a) not in self.R:
                                self.R[tuple(s), a] = 0

                            v = self.R[tuple(s), a] + self.gamma * self.Sigma_(tuple(s), a, self.V_)

                            self.Q[tuple(s)][a] = v

                            if v > max_v:
                                max_v, max_a = v, a

                    for choice in self.Q[tuple(s)]:
                        self.Pi[tuple(s)][choice] = 0
                    self.V[tuple(s)] = max_v
                    self.Pi[tuple(s)][max_a] = 1.0

            V_current, V_last = self.Dict2Vec(self.V, self.S), self.Dict2Vec(self.V_, self.S)
            diff.append(np.inner(V_current - V_last, V_current - V_last))

            it += 1
        print("action iterations:", it)

        if not self.plotKey:
            self.svi_record = special[len(special) - 1]
            print("hardmax action special point: ", special)

        return special, self.Pi

    def policy_evaluation(self, initV):
        # S, A, P, R, sink, V, V_ = inS, inA, inP, inR, in_sink, inV, inV_
        tau = 1.0

        Q = {}
        Pi, Pi_ = {}, {}
        V, V_ = dcp(initV), {}

        # V_current, V_last = self.Dict2Vec(self.V, self.S), self.Dict2Vec(self.V_, self.S)
        it = 1

        while it < 2: # np.inner(V_current - V_last, V_current - V_last) > threshold

            for s in self.S:
                V_[tuple(s)] = V[tuple(s)]

                if tuple(s) not in Pi: # for softmax
                    Pi[tuple(s)] = {}
                if tuple(s) not in Q:
                    Q[tuple(s)] = {}

            for s in self.S:

                if tuple(s) not in self.T and tuple(s) not in self.interruptions:
                    # # # print   s, self.T
                    max_v, max_a = -0.001, None

                    for a in self.A:
                        if (tuple(s), a) in self.P:
                            if (tuple(s), a) not in self.R:
                                self.R[tuple(s), a] = 0

                            v = self.R[tuple(s), a] + self.gamma * self.Sigma_(tuple(s), a, V_)
                            # if self.R[tuple(s), a] > 0:
                                # print  s, a, self.R[tuple(s), a]
                            Q[tuple(s)][a] = np.exp(v / tau)  # softmax solution
                            # if self.plotKey:
                            #     # # print   v, self.R[tuple(s), a], self.Sigma_(tuple(s), a)

                            if v > max_v:
                                max_v, max_a = v, a

                    sumQ = sum(Q[tuple(s)].values())
                    for choice in Q[tuple(s)]:
                        Pi[tuple(s)][choice] = Q[tuple(s)][choice] / sumQ
                    V[tuple(s)] = tau * np.log(sumQ)

            # V_current, V_last = self.Dict2Vec(self.V, self.S), self.Dict2Vec(self.V_, self.S)

            it += 1

        return Pi

    def goal_probability(self, Pi, P, s_monitor, threshold):
        # hardmax evaluation with fixed policy till convergence, return specific state probability propagation as reference
        V, V_ = {},{}

        for state in self.S:
            s = tuple(state)
            if s not in V:
                V[s], V_[s] = 0.0, 0.0
            if state[1] == list(self.dfa.final_states)[0]:
                V[s], V_[s] = 1.0, 0.0

        V_current, V_last = self.Dict2Vec(V, self.S), self.Dict2Vec(V_, self.S)
        it = 1
        diff = []
        special = []
        diff.append(np.inner(V_current - V_last, V_current - V_last))

        # print (Pi)

        while np.inner(V_current - V_last, V_current - V_last) > threshold:
            # self.plot_map(it)

            if tuple(s_monitor) in V:
                special.append(V[s_monitor])

            for s in self.S:
                V_[tuple(s)] = V[tuple(s)]

            for s in self.S:
                if tuple(s) not in self.T and tuple(s) not in self.interruptions:
                    v = 0
                    # max_v, max_Pi = 0, 0
                    for a in self.A:
                        if (tuple(s), a) in P:
                            v += Pi[tuple(s)][a]*(self.gamma * self.Sigma_(tuple(s), a, V_))
                            # v = self.gamma * self.Sigma_(tuple(s), a, V_)
                            # if Pi[tuple(s)][a] > max_Pi:
                            #     max_v, max_Pi = v, Pi[tuple(s)][a]
                    V[tuple(s)] = v
                    # V[tuple(s)] = max_v

            V_current, V_last = self.Dict2Vec(V, self.S), self.Dict2Vec(V_, self.S)
            diff.append(np.inner(V_current - V_last, V_current - V_last))

            it += 1
        return V
        # return special

    def compute_norm(self, V, V_, level):
        V_current, V_last = self.Dict2Vec(V, self.S), self.Dict2Vec(V_, self.S)
        # norm = np.inner(V_current - V_last, V_current - V_last)
        norm = np.linalg.norm(V_current - V_last, level) / np.linalg.norm(V_last, level)
        print('norm level {}:'.format(level), norm)


    # VERIFICATION
    def evaluation(self, Policy, P, s0, trial = 1000):
        rate = 0.0
        total = 0.0
        succeed = 0.0

        for i in range(trial):
            total += 1

            st = s0
            traj = [s0]
            while st not in self.T:
                piActs = list(Policy[st].keys())
                piVals = list(Policy[st].values())
                try:
                    a = np.random.choice(piActs, 1, p = piVals)[0]
                except ValueError:
                    print('ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print(piActs, st)
                    print(self.T)

                pNeighbors = list(P[st, a].keys())
                pCode = range(len(pNeighbors))
                pVals = list(P[st, a].values())

                # print(P[st, a], st, a)

                ns = np.random.choice(pCode, 1, p = pVals)[0]

                st = pNeighbors[ns]
                traj.append(st)

                if st[1] == list(self.dfa.final_states)[0]: #TODO: fix this problem!!!!
                    # print ('goal:', st)
                    succeed += 1
            # print('nongoal:',st)
            # print(traj)

        rate = succeed*1.0/total
        return rate


    # VIS: plotting heatmap using either matplotlib ot plotly to visualize value function for all options
    def option_plot(self, title=None, dir=None):
        z = {}
        folder = '../LTL-option-framework/composed_option_vis/'
        for key in self.Opt:
                # # print   self.Opt[key].S,  len(self.Opt[key].S)
                # # print  "key name:", key
                # q1 = key[1][0]
                # q2 = key[1][1]
            g = ''
            for element in self.Opt[key].id[0]:
                g += element
                g += ','
            g += self.Opt[key].id[1]
                # # # print   self.V
            temp = np.zeros((6, 8))
            for state in self.Opt[key].S:
                    # key = ((i, j), )
                    # if (i, j) not in self.V:
                    #     temp[(i, j)] = -1
                    # # # print   state[0], state
                i, j = state[0] - 1, state[1] - 1
                temp[i, j] = self.Opt[key].V[tuple(state)]

            name = "Solution to goal-" + str(g)
            z[name] = temp

                # name = 'value function at automata state:' + str(q) + " at it:" + str(iteration)

            fig = plt.figure()
            cax = plt.imshow(temp, interpolation='nearest')
            # cbar = fig.colorbar(cax)  # ticks=[-1, 0, 1]
            plt.savefig(folder + name + ".png")  # "../DFA/comparing test/"

    # VIS: plotting heatmap using either matplotlib ot plotly to visualize value function for all options
    def layer_plot(self, title=None, dir=None):
        z = {}
        folder = '../LTL-option-framework/single_option_vis/'
        for key in self.AOpt:
            # # print   self.Opt[key].S,  len(self.Opt[key].S)
            # # print  "key name:", key
            # q1 = key[1][0]
            # q2 = key[1][1]
            g = key

            # # # print   self.V
            temp = np.zeros((6, 8))
            for state in self.AOpt[key].S:
                    # key = ((i, j), )
                    # if (i, j) not in self.V:
                    #     temp[(i, j)] = -1
                # # # print   state[0], state
                i, j = state[0]-1, state[1]-1
                temp[i, j] = self.AOpt[key].V[tuple(state)]

            name = "Solution to goal-" + str(g)
            z[name] = temp


            # name = 'value function at automata state:' + str(q) + " at it:" + str(iteration)

            fig = plt.figure()
            cax = plt.imshow(temp, interpolation='nearest')
            # cbar = fig.colorbar(cax)  # ticks=[-1, 0, 1]
            plt.savefig(folder + name + ".png")  # "../DFA/comparing test/"

        # title = "test_layers"
        # fname = 'RL/LTL_SVI/' + title
        # fname = dir + title
        #
        # trace = []
        # names = z.keys()
        # fig = tools.make_subplots(rows=2, cols=2,
        #                           subplot_titles=(names[0], names[1], names[2], names[3])
        #                           )
        # # specs = [[{'is_3d': True}, {'is_3d': True}], [{'is_3d': True}, {'is_3d': True}]]
        # # subplot_titles=(names[0], names[1], names[2], names[3])
        #
        # trace.append(go.Heatmap(z=z[names[0]], showscale=True, colorscale='Jet'))
        # trace.append(go.Heatmap(z=z[names[1]], showscale=False, colorscale='Jet'))
        # trace.append(go.Heatmap(z=z[names[2]], showscale=False, colorscale='Jet'))
        # trace.append(go.Heatmap(z=z[names[3]], showscale=False, colorscale='Jet'))
        #
        # fig.append_trace(trace[0], 1, 1)
        # fig.append_trace(trace[1], 1, 2)
        # fig.append_trace(trace[2], 2, 1)
        # fig.append_trace(trace[3], 2, 2)
        #
        # # py.iplot(
        # #     [
        # #      dict(z=z[0]+1.0, showscale=False, opacity=0.9, type='surface'),
        # #      dict(z=z[1]+2.0, showscale=False, opacity=0.9, type='surface'),
        # #      dict(z=z[2]+3.0, showscale=False, opacity=0.9, type='surface'),
        # #      dict(z=z[3]+4.0, showscale=False, opacity=0.9, type='surface')],
        # #     filename=fname)
        #
        # fig['layout'].update(title=title)
        #
        # py.iplot(fig, filename=fname)

        return 0

    # VIS: plot whole product state space value functions

    def plot_map(self, iteration):

        Aphi = self.dfa.state_info

        for q in Aphi:
            # sns.set()
            # if q == 1:
            #     # print 
            goals, obs = [], []
            for ap in Aphi[q]['safe']:
                goals += self.Exp[ap]
            goals = list(set(goals))

            for ap in Aphi[q]['unsafe']:
                obs += self.Exp[ap]
            obs = list(set(obs))

            temp = np.random.random((6, 8))
            for state in self.originS:
                v = 0
                count = 0
                if tuple(state) in goals:
                    for ap in Aphi[q]['safe']:
                        if tuple(state) in self.Exp[ap]:
                            v += self.V[tuple(state), self.dfa.state_transitions[ap,q]]
                            count += 1
                    temp[state[0] - 1, state[1] - 1] = v/count
                elif tuple(state) in obs:
                    temp[state[0] - 1, state[1] - 1] = 0.0
                else:
                    temp[state[0] - 1, state[1] - 1] = self.V[tuple(state), q]
            folder = '../LTL-option-framework/result/'
            name = 'value function at automata state:'+ str(q) + " at it:" + str(iteration)

            fig = plt.figure()
            ax = plt.gca()
            cax = ax.imshow(temp)
            # cbar = fig.colorbar(cax) #ticks=[-1, 0, 1]
            ax.grid(which="minor", color="w", linestyle='-', linewidth=10)
            # plt.show()
            fig.savefig(folder + name + ".png")  # "../DFA/comparing test/"
            # x, y = np.mgrid[-1.0:1.0:6j, -1.0:1.0:8j]
            # # #
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # # ax.plot_wireframe(x, y, temp)
            # surf = ax.plot_surface(x, y, temp, cmap='viridis', edgecolor='black')
            # cbar = fig.colorbar(surf)
            # fig.savefig(folder + name + ".png")  # "../DFA/comparing test/"


    # VIS: plotting single heatmap when necessary, used for debugging!!
    '''
    def plot_heat(self, key, dir=None):

        g, q1, q2 = None, None, None
        folder = "" if dir == None else dir

        if type(key) == type("string") or type(key) == type(tuple([1,2,3])):
            try:
                name = "reaching option" + key
            except:
                name = "reaching option"
                for element in key[0]:
                    name = name + '-' + element
            plt.figure()

            temp = np.random.random((6, 8))
            for state in self.S:
                if state not in self.wall_cord:
                    temp[state[0] - 1, state[1] - 1] = self.V[tuple(state)]

            plt.imshow(temp, cmap='hot', interpolation='nearest')
            plt.savefig(folder + name + ".png")
            return 0
        else:

            temp = np.random.random((6, 8))
            for state in self.S:
                temp[state[0][0] - 1, state[0][1] - 1] = self.V[state]

            q1 = key[1][0]
            q2 = key[1][1]
            g = key[0]


        name = "layer-" + str(q1) + "-" + str(g) + "-" + str(q2)
        plt.figure()
        plt.imshow(temp, cmap='hot', interpolation='nearest')
        plt.savefig(folder + name + ".png")  # "../DFA/comparing test/"

        ct = go.Contour(
            z=temp,
            type='surface',
            colorscale='Jet',
        )
        fname = 'RL/LTL_SVI/' + name

        py.iplot(
            [   ct,
                dict(z=temp, showscale=False, opacity=0.9, type='surface')],
            filename=fname)

        return 0
    '''
    # VIS: comparing the trend of two curves for any function or variable
    def plot_curve(self, curve, name):
        plt.figure()

        # # # print   x
        # # # print   trace1
        # # # print   trace2
        l1, = plt.plot(curve["action"], label="action")
        l2, = plt.plot(curve["option"], label="option")
        l3, = plt.plot(curve["hybrid"], label="hybrid")
        l4, = plt.plot(curve["optimal"], label="optimal")

        plt.legend(handles=[l1, l2, l3, l4])

        plt.ylabel('V-value')
        plt.xlabel('iteration number')
        plt.grid(True)
        plt.savefig(name + ".png")

    # VIS: Visulizing policy direction and weight in grid world
    def draw_quiver(self, name):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # for s in self.Pi:
        for state in self.S:
            s = tuple(state)
            ma, mpi = '', 0
            if s not in self.goal and s not in self.unsafe and s not in self.interruptions:
                for a in self.Pi[s]:
                    if self.Pi[s][a] > mpi:
                        ma, mpi = a, self.Pi[s][a]
                # # print  np.array(s), np.array(self.A[ma])
                # s_ = tuple(np.array(s)+np.array(self.A[ma]))
                ax.quiver(s[0],s[1], self.A[ma][0], self.A[ma][1], angles='xy', scale_units='xy', scale=3/mpi)
        # plt.xticks(range(-5, 6))
        # plt.yticks(range(-5, 6))
        plt.grid()
        plt.draw()
        # plt.show()
        plt.savefig(name + 'policy.png')
        return


