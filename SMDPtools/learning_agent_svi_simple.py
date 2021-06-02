import numpy as np

from DFA import DFA
from DFA import DRA
from DFA import DRA2
from DFA import Action

from gridworld import Tile, Grid_World
# from q_learner import Q_learning
from MDP_learner import MDP
import pygame, sys, time, random
from pygame.locals import *

import yaml

import matplotlib.pyplot as plt
# from matplotlib.axes import Axes as ax

def get_features(pos):
    return pos[0]*(board_size[0]) + pos[1] # -1

def decode_features(features):
    return [features/board_size[0], features - board_size[0]*(features/board_size[0]) ]


if __name__ == "__main__":

    threshold = 0.00000000001
    length = 4
    width = 4
    board_size = [length, width]
    wall = []

    pygame.init()
    #
    # # Set window size and title, and frame delay
    surfaceSize = (32 * 4, 32 * 4)
    windowTitle = 'UAV Grid World'
    pauseTime = 1  # 0.01  # smaller is faster game

    with open("toy_config.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    # Create the window
    surface = pygame.display.set_mode(surfaceSize, 0, 0)
    pygame.display.set_caption(windowTitle)

    # create and initialize objects
    gameOver = False

    # print surface
    board = Grid_World(surface, **data_loaded)

    A = Action('A')
    B = Action('B')  # set as an globally unsafe obstacle
    C = Action('C')
    phi = Action('phi')
    whole = Action('whole')
    sinks = Action('failure') # set of all 0 energy states
    trial_number = 50

    #TODO: sample state space
    S = list(range(7)) # A:1, B:2, e:0

    # brute force labeling
    s_q = {}
    q_s = {}
    q_s[A.v] = []
    q_s[B.v] = []
    q_s[C.v] = []
    q_s[sinks.v] = []

    q_s[A.v].append(1)
    q_s[B.v].append(2)
    q_s[C.v].append(0)
    q_s[sinks.v].append(6)

    q_s[phi.v] = list(set(S) - set(q_s[A.v] + q_s[B.v] + q_s[C.v] + q_s[sinks.v]))
    q_s[whole.v] = list(set(S))

    for s in S:
        s_q[s] = []
        if s in q_s[A.v] and A.v not in s_q[s]:
            s_q[s].append(A.v)
        if s in q_s[B.v] and B.v not in s_q[s]:
            s_q[s].append(B.v)
        if s in q_s[C.v] and C.v not in s_q[s]:
            s_q[s] = [C.v]
        if s in q_s[sinks.v] and sinks.v not in s_q[s]:
            s_q[s] = [sinks.v]
        if s in q_s[phi.v] and phi.v not in s_q[s]:
            s_q[s].append(phi.v)

    mdp = MDP()
    mdp.set_S(S)
    # mdp.set_WallCord(mdp.add_wall(q_s[sinks.v]))
    # mdp.set_Cs(board.C_coords)
    # mdp.set_Cs(board.goal_coord)
    # mdp.set_full_tank(board.battery_cap)
    # mdp.set_P()
    mdp.set_P_simple()
    mdp.set_L(s_q)  # L: labeling function (L: S -> Q), e.g. self.L[(2, 3)] = 'g1'
    mdp.set_Exp(q_s)  # L^-1: inverse of L (L: Q -> S), e.g. self.Exp['g1'] = (2, 3)
    mdp.set_Size(4, 4)

    dfa = DFA(0, [A, B, C, sinks, phi, whole]) # sink state here means energy used up outside of the C
    dfa.set_final(1)
    dfa.set_final(2)
    dfa.set_final(3)
    dfa.set_final(4)
    dfa.set_sink(5)

    dfa.pref_labeling(1, 'I') # using rome number
    dfa.pref_labeling(3, 'I')
    dfa.pref_labeling(2, 'II')
    dfa.pref_labeling(4, 'III')

    sink = list(dfa.sink_states)[0]

    for i in range(sink+1):
        dfa.add_transition(phi.display(), i, i)
        if i < sink:
            dfa.add_transition(sinks.display(), i, sink)

    dfa.add_transition(A.display(), 1, 1)
    dfa.add_transition(A.display(), 2, 2)
    dfa.add_transition(A.display(), 3, 3)
    dfa.add_transition(A.display(), 4, 4)

    dfa.add_transition(B.display(), 2, 2)
    dfa.add_transition(B.display(), 3, 3)
    dfa.add_transition(B.display(), 4, 4)

    dfa.add_transition(C.display(), 0, 0)
    dfa.add_transition(C.display(), 1, 1)
    dfa.add_transition(C.display(), 2, 2)
    dfa.add_transition(C.display(), 4, 4)

    dfa.add_transition(whole.display(), sink, sink)

    dfa.add_transition(A.display(), 0, 1)
    dfa.add_transition(B.display(), 0, 3)
    # dfa.add_transition(C.display(), 0, 8)

    dfa.add_transition(B.display(), 1, 2)
    # dfa.add_transition(C.display(), 1, 7)

    # dfa.add_transition(C.display(), 2, 5)

    dfa.add_transition(C.display(), 3, 4)
    # dfa.add_transition(C.display(), 3, 7)

    # dfa.add_transition(C.display(), 4, 6)

    dfa.toDot("DFA")
    dfa.prune_eff_transition()
    # dfa.g_unsafe = 'sink'

    curve = {}
    result = mdp.product(dfa, mdp)
    
    result.plotKey = False

    with open("prod_MDP_simple", 'w', encoding = "utf-8") as yaml_file:
        yaml.dump(result.P, yaml_file)
    # curve['action'] = result.SVI(0.001)

    with open("prod_MDP_simple", 'r', encoding = "utf-8") as yaml_file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        saved_prod_space = yaml.load(yaml_file, Loader=yaml.FullLoader)

    print("end.")




