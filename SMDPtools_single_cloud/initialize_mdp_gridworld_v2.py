import numpy as np

from DFA import DFA
from DFA import DRA
from DFA import DRA2
from DFA import Action

from gridworld import Tile, Grid_World
# from q_learner import Q_learning
# from MDP_learner_backup import MDP
from MDP_learner_gridworld_cleaned import MDP
import pygame, sys, time, random
from pygame.locals import *

import yaml
import pickle
import json

import matplotlib.pyplot as plt
# from matplotlib.axes import Axes as ax

def get_features(pos):
    return pos[0]*(board_size[0]) + pos[1] # -1

def decode_features(features):
    return [features/board_size[0], features - board_size[0]*(features/board_size[0]) ]

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


if __name__ == "__main__":

    threshold = 0.00000000001
    length = 4
    width = 4
    board_size = [length, width]
    wall = []

    pygame.init()
    # definition of


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

    A = Action('a&!dead')
    B = Action('b&!dead')
    C = Action('c&!dead')
    bases = Action('base')
    dead_sinks = Action('dead')  # set of all 0 energy states
    base_or_dead = Action('base|dead')
    whole = Action('1')

    n_ABC = Action('!(a|b|c|base|dead)')
    n_AC = Action('!(a|c|base|dead)')
    n_BC = Action('!(b|c|base|dead)')
    n_AB = Action('!(a|b|base|dead)')
    n_A = Action('!(a|base|dead)')
    n_B = Action('!(b|base|dead)')
    n_C = Action('!(c|dead)')
    n_end = Action('!(base|dead)')

    trial_number = 50
    states = board.get_state_space()
    S = board.get_state_space("tuple")

    # brute force labeling
    s_q = {}
    q_s = {}
    q_s[A.v] = []
    q_s[B.v] = []
    q_s[C.v] = []
    q_s[bases.v] = []
    q_s[dead_sinks.v] = []
    q_s[base_or_dead.v] = []

    q_s[n_ABC.v] = []
    q_s[n_AC.v] = []
    q_s[n_BC.v] = []
    q_s[n_AB.v] = []
    q_s[n_A.v] = []
    q_s[n_B.v] = []
    q_s[n_C.v] = []
    q_s[n_end.v] = []
    q_s[whole.v] = []

    for c1_i in range(board.board_size[0]):
        for b in range(1, board.battery_cap):  # !dead
            q_s[A.v].append(tuple(board.goal_coord[0]+[c1_i, b]))
            q_s[B.v].append(tuple(board.goal_coord[1]+[c1_i, b]))
            q_s[C.v].append(tuple(board.goal_coord[2]+[c1_i, b]))

        for station_coord in board.station_coords:  # base, the station only allows full tank states, e.g. 12
            q_s[bases.v].append(tuple(station_coord+[c1_i, board.battery_cap]))  # base
            q_s[base_or_dead.v].append(tuple(station_coord + [c1_i, board.battery_cap]))  # base|dead

    impossible_states = []
    s_A, s_B, s_C = board.goal_coord
    s_ABC = board.goal_coord
    s_AB = [s_A, s_B]
    s_AC = [s_A, s_C]
    s_BC = [s_B, s_C]

    for i in range(board.board_size[0]):
        for j in range(board.board_size[1]):
            for c1_i in range(board.board_size[0]):
                if [i, j] not in board.station_coords:  # !base
                    # dead, when the uav used up its battery outside of the station
                    q_s[dead_sinks.v].append(tuple([i, j, c1_i, 0]))  # dead
                    q_s[base_or_dead.v].append(tuple([i, j, c1_i, 0]))  # base|dead
                    for k in range(1, board.battery_cap):  # !dead <=> battery>0, battery<cap
                        if [i, j] not in s_ABC:  # !(a|b|c|base|dead)
                            q_s[n_ABC.v].append(tuple([i, j, c1_i, k]))
                        if [i, j] not in s_AC:  # !(a|c|base|dead)
                            q_s[n_AC.v].append(tuple([i, j, c1_i, k]))
                        if [i, j] not in s_BC:  # !(b|c|base|dead)
                            q_s[n_BC.v].append(tuple([i, j, c1_i, k]))
                        if [i, j] not in s_AB:  # !(a|b|base|dead)
                            q_s[n_AB.v].append(tuple([i, j, c1_i, k]))
                        if not [i, j] == s_A:  # !(a|base|dead)
                            q_s[n_A.v].append(tuple([i, j, c1_i, k]))
                        if not [i, j] == s_B:  # !(b|base|dead)
                            q_s[n_B.v].append(tuple([i, j, c1_i, k]))
                        if not [i, j] == s_C:  # !(c|dead)
                            q_s[n_C.v].append(tuple([i, j, c1_i, k]))

                        # !(base|dead), as long as the uav state is not in base and not dead
                        q_s[n_end.v].append(tuple([i, j, c1_i, k]))

                else:  # base, remove impossible states
                    if not [i, j] == s_C:  # !(c|dead)  special case
                        q_s[n_C.v].append(tuple([i, j, c1_i, board.battery_cap]))
                    for k in range(board.battery_cap):
                        # when the uav is at the station,
                        # its power could only be full_cap,
                        # 0-11 will be removed from the state space if full_cap = 12
                        impossible_states.append(tuple([i, j, c1_i, k]))

    q_s[whole.v] = list(set(S) - set(impossible_states))

    # get the inverse dict of labeling function
    S_filtered = list(set(S) - set(impossible_states))
    for s in S_filtered:
        s_q[s] = []
        if s in q_s[A.v] and A.v not in s_q[s]:
            s_q[s].append(A.v)
        if s in q_s[B.v] and B.v not in s_q[s]:
            s_q[s].append(B.v)
        if s in q_s[C.v] and C.v not in s_q[s]:
            s_q[s].append(C.v)
        if s in q_s[bases.v] and bases.v not in s_q[s]:
            s_q[s] = [bases.v]
        if s in q_s[dead_sinks.v] and dead_sinks.v not in s_q[s]:
            s_q[s].append(dead_sinks.v)
        if s in q_s[n_ABC.v] and n_ABC.v not in s_q[s]:
            s_q[s].append(n_ABC.v)
        if s in q_s[n_AC.v] and n_AC.v not in s_q[s]:
            s_q[s].append(n_AC.v)
        if s in q_s[n_BC.v] and n_BC.v not in s_q[s]:
            s_q[s].append(n_BC.v)
        if s in q_s[n_AB.v] and n_AB.v not in s_q[s]:
            s_q[s].append(n_AB.v)
        if s in q_s[n_A.v] and n_A.v not in s_q[s]:
            s_q[s].append(n_A.v)
        if s in q_s[n_B.v] and n_B.v not in s_q[s]:
            s_q[s].append(n_B.v)
        if s in q_s[n_C.v] and n_C.v not in s_q[s]:
            s_q[s].append(n_C.v)
        if s in q_s[n_end.v] and n_end.v not in s_q[s]:
            s_q[s].append(n_end.v)
        if s in q_s[whole.v] and whole.v not in s_q[s]:
            s_q[s].append(whole.v)
        if s in q_s[base_or_dead.v] and base_or_dead.v not in s_q[s]:
            s_q[s].append(base_or_dead.v)

    mdp = MDP()
    mdp.set_S(S_filtered)
    mdp.set_unsafe(mdp.add_wall(q_s[dead_sinks.v]))  # should be changed to unsafe
    mdp.set_stations(board.station_coords)
    mdp.set_stations(board.goal_coord)
    mdp.set_full_tank(board.battery_cap)
    mdp.set_P()
    mdp.set_L(s_q)  # L: labeling function (L: S -> Q), e.g. self.L[(2, 3)] = 'g1'
    mdp.set_Exp(q_s)  # L^-1: inverse of L (L: Q -> S), e.g. self.Exp['g1'] = (2, 3)
    mdp.set_Size(4, 4)

    # specify DFA graph structure
    dfa = DFA(0, [A, B, C, bases, dead_sinks, n_ABC, n_AB, n_AC, n_BC, n_A, n_B, n_C, n_end, whole])

    dfa.set_final(4)
    dfa.set_final(7)
    dfa.set_final(8)
    dfa.set_sink(9)

    dfa.pref_labeling(7, 'X1')
    dfa.pref_labeling(8, 'X2')
    dfa.pref_labeling(4, 'X2')


    dfa.add_pref_trans('X1', 'X2')

    dfa.inv_pref_labels = inv_dict(dfa.pref_labels)
    dfa.inv_pref_trans = inv_dict2(dfa.pref_trans)

    dfa.pref_trans['X2'] = []
    dfa.inv_pref_trans['X1'] = []

    sink = list(dfa.sink_states)[0]

    # add self-loop transitions
    dfa.add_transition(n_ABC.display(), 0, 0)
    dfa.add_transition(n_BC.display(), 1, 1)
    dfa.add_transition(n_AC.display(), 2, 2)
    dfa.add_transition(n_AB.display(), 3, 3)
    dfa.add_transition(n_C.display(), 4, 4) # special self loop
    dfa.add_transition(n_B.display(), 5, 5)
    dfa.add_transition(n_A.display(), 6, 6)

    # sinking or static transitions after sink or final
    for i in range(sink+1):
        if i < 7 and not i == 4:
            dfa.add_transition(base_or_dead.display(), i, sink)
        if i == 4:
            dfa.add_transition(dead_sinks.display(), i, sink)
        if i >= 7 and i <= sink: #7, 8, 9
            dfa.add_transition(whole.display(), i, i)

    # progressive transitions
    dfa.add_transition(A.display(), 0, 1)
    dfa.add_transition(B.display(), 0, 2)
    dfa.add_transition(C.display(), 0, 3)

    dfa.add_transition(B.display(), 1, 4)
    dfa.add_transition(C.display(), 1, 5)

    dfa.add_transition(A.display(), 2, 4)
    dfa.add_transition(C.display(), 2, 6)

    dfa.add_transition(A.display(), 3, 5)
    dfa.add_transition(B.display(), 3, 6)

    dfa.add_transition(C.display(), 4, 7)
    dfa.add_transition(B.display(), 5, 7)
    dfa.add_transition(A.display(), 6, 7)

    # succeeding transitions
    dfa.add_transition(bases.display(), 4, 8)


    dfa.toDot("DFA")
    dfa.prune_eff_transition()
    dfa.g_unsafe = 'sink'

    curve = {}
    result = mdp.product(dfa, mdp, "gridworld")
    result.plotKey = False
    # curve['action'] = result.SVI(0.001)

    # with open("prod_MDP_gridworld.json", 'wb') as json_file: # TODO: change to json
    #     json.dump(result, json_file)

    with open("prod_MDP_gridworld_v2.pkl", 'wb') as pkl_file: #pkl
        pickle.dump(result, pkl_file)
    # curve['action'] = result.SVI(0.001)







