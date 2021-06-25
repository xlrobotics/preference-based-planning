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
import yaml
import json

'''
Comment tags:
    Init:       used for initializing the object parameters
    Tool:       inner function called by API, not transparent to the user
    API:        can be called by the user
    VIS:        visualization
    DISCARD:    on developing or discarded functions in old version
'''


class MDP:
    #static properties
    INFTY = 1000000000
    def __init__(self, gamma = 0.9, epsilon = 0.15, gain = 100.0, tau = 1.0, transition_file=""):
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

        self.vector_V = {}

        self.Q, self.Q_ = {}, {} # state action value function
        self.T = {} # terminal states
        self.Po = {} # transition matrix

        self.R = {} # reward function
        self.init_R = {} # initial reward function

        self.P = {} # 1 step transition probabilities
        self.transition_file = transition_file
        if len(transition_file)>0:
            with open(transition_file, 'rb') as yaml_file:
                # The FullLoader parameter handles the conversion from YAML
                # scalar values to Python the dictionary format
                P = pickle.load(yaml_file)

        self.H = {}
        self.ASF = {}
        self.ASR = {}
        self.Pref_order = []

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

    def less(self, s, t):
        return

    def more(self, s, t):
        return

    def equal(self, s, t):
        for i in range(len(self.ASF)):
            if self.V[i][s] != self.V[i][t]:
                return False
        return True

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
    def AS_reachability(self, F):
        # almost sure reaching the set F.
        # alg: compute the set of states from which the agent can stay with probability one, and has a positive probability of reaching F (in multiple steps).
        X0 = set(self.S)

        counter = 0
        print(X0)
        while True:
            Y0 = set(F)
            Y1 = dcp(Y0)
            # print("iteration: ", counter)
            counter+=1
            while True:
                Y0 = dcp(Y1)
                # print("UPDATED: ", Y1, "|||", Y0)
                for s in X0:
                    # print("UPDATED: ", Y1)
                    for a in self.A_simple: # TODO: action_special is for the toy, for gridworld need to modify it
                        reachableStates = self.nextStateSet(s, a)  # TODO: computing the next states that can be reached with probability >0.

                        # if reachableStates.issubset(X0) and reachableStates.intersect(Y1) != set([]):
                        if reachableStates <= X0 and bool(reachableStates.intersection(Y1)):
                            Y1.add(s)
                            break
                if Y1 == Y0:  # no state is added, reached the inner fixed point.
                    # print("Y1", Y1)
                    break
                # print("Y1", Y1)
            if X0 != Y1: # shrinking X0
                X0 = Y1
                # print("X0", X0)
            else:  # reach the outer fixed point.
                break  # outer loop complete.
        return X0

    def predecessors(self, v): # v as state
        result = set()
        for key in self.P:
            if v in self.P[key] and self.P[key][v] > 0:
                result.add(key)     # PATCH return (u, a)

        return result

    def successors(self, v, a): # v as state, a as action
        result = set()
        if len(self.P[tuple(v), a]):
            for v_ in self.P[tuple(v), a]:
                if type(v_) == tuple and self.P[tuple(v), a][v_] > 0:
                    result.add(v_)

        return result
        # return list(result)

    def enabled_actions(self, v): # v as a state
        result = set()
        for a in self.A:
            flag = False
            if tuple([tuple(v), a]) in self.P:
                for v_ in self.P[tuple(v), a]:
                    if type(v_) == tuple and self.P[tuple(v), a][v_] >= 0:
                        flag = True
                        break
            if flag:
                result.add(a)

        return result

    def get_vector_V(self, s): #TODO: add the function here
        result = {}
        for key in self.dfa.inv_pref_labels.keys():
            if key not in result:
                result[key] = self.vector_V[key][s]
        return result

    def init_ASR_V(self, s):

        visited = []  # List to keep track of visited nodes.
        queue = []  # Initialize a queue
        for key in self.dfa.inv_pref_labels.keys():
            if key not in self.dfa.pref_trans and key not in visited and key not in queue: # key=“I” in the example
                visited.append(key)
                queue.append(key)

        while queue:
            i = queue.pop(0)
            # print(i, end=" ")
            # print(s, i, visited, queue)
            if s in self.ASR[i]: # start from mostly preferred node in the graph (top), e.g. node "I" - q=8
                self.vector_V[i][s] = 1
                if i in self.dfa.pref_trans:
                    for j in self.dfa.pref_trans[i]:
                        if s in self.vector_V[j] and self.vector_V[j][s] == 1 and i not in self.dfa.pref_trans[j]: # j in i and i in j means i,j are equivalent
                            self.vector_V[i][s] = 0
                            break

                    # no matter i is 1 or 0
                    if i in self.dfa.inv_pref_trans:
                        for k in self.dfa.inv_pref_trans[i]: # check if there were less preferred nodes visited already due to the topology
                            if s in self.vector_V[k] and self.vector_V[k][s] == 1 and i not in self.dfa.inv_pref_trans[k] and self.vector_V[k][s] == 1: # j in i and i in j means i,j are equivalent
                                self.vector_V[k][s] = 0
            else:
                self.vector_V[i][s] = 0

            if i in self.dfa.inv_pref_trans:
                for child in self.dfa.inv_pref_trans[i]:
                    # print(child, visited)
                    if child not in visited:
                        visited.append(child)
                        queue.append(child)



    # API: DFA * MDP product
    def product(self, dfa, mdp):
        result = MDP(self.transition_file)
        result.Exp = dcp(mdp.Exp)
        result.L = self.L

        filtered_S = []
        for s in mdp.S:  #new_S
            if s == [0, 3, 0, 0, 0]:
                print(s)
            if mdp.L[tuple(s)][0] != 'failure' and s not in filtered_S: # new_wall
                filtered_S.append(s)
            # filtered_S.append(s) # no need to filter here

        result.originS = dcp(filtered_S)

        # result.init_S = self.crossproduct2(list([dfa.initial_state]), mdp.Exp['phi'])
        #
        # new_success = self.crossproduct2(list(dfa.final_states-dfa.sink_states), filtered_S)
        # new_fail = self.crossproduct2(list(dfa.sink_states), filtered_S)

        result.init_S = self.crossproduct(list([dfa.initial_state]), mdp.Exp['phi'])

        new_success = self.crossproduct(list(dfa.final_states - dfa.sink_states), filtered_S)
        new_fail = self.crossproduct(list(dfa.sink_states), filtered_S)

        # calculate F states for each pref node:
        for key in dfa.inv_pref_labels.keys():
            if key not in result.ASF:
                # result.ASF[key] = result.crossproduct2(dfa.inv_pref_labels[key], filtered_S)
                result.ASF[key] = result.crossproduct(dfa.inv_pref_labels[key], filtered_S)
                # result.ASF[key] = [(3, 1)]

        result.success = new_success[:]

        new_P, new_R = {}, {}
        new_V, new_V_ = {}, {}

        true_new_s = []

        sink = "pref node"
        fail = "fail"

        total = len(mdp.P.keys())
        counter = 0

        if len(self.transition_file) == 0:
            for p in mdp.P.keys():
                counter += 1

                # if counter == 100:
                #     break

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

                            # if new_s_ in new_success and new_P[new_s, new_a][new_s_]>0 and new_s_[1] in dfa.pref_labels: # tagging, debug step, delete after check correct
                            #     new_P[new_s, new_a][sink] = dfa.pref_labels[new_s_[1]]
                            #
                            # if new_s_ in new_fail and new_P[new_s, new_a][new_s_]>0: # tagging, debug step, delete after check correct
                            #     new_P[new_s, new_a][fail] = 1

                    if new_s not in true_new_s:
                        true_new_s.append(tuple(new_s))


        result.set_S(true_new_s)

        result.P = dcp(new_P)

        result.T = dcp(new_success)

        result.dfa = dcp(dfa)
        result.mdp = dcp(mdp)

        for key in result.ASF.keys():
            if key not in result.ASR:
                result.ASR[key] = result.AS_reachability(result.ASF[key])

                result.vector_V[key] = {}
                # for s in result.ASR[key]:
                #     if s not in result.V[key]:
                #         result.V[key][s] = 0

        for s in result.S:
            result.init_ASR_V(s)


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
                            if (s, a, s_) not in self.P:
                                self.P[s, a, s_] = 1/3*1/3
                            else:
                                self.P[s, a, s_] += 1/3 * 1/3

        return

    def add_trans_P(self, s, a, s_, value):
        self.P[s, a, s_] = value







