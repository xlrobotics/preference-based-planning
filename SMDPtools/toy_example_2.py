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
import pickle

import matplotlib.pyplot as plt


# from matplotlib.axes import Axes as ax

def get_features(pos):
    return pos[0] * (board_size[0]) + pos[1]  # -1


def decode_features(features):
    return [features / board_size[0], features - board_size[0] * (features / board_size[0])]


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

    B = Action('B')  # set as an globally unsafe obstacle
    phi = Action('phi')
    whole = Action('whole')
    sinks = Action('failure')  # set of all 0 energy states
    trial_number = 50

    # TODO: sample state space
    S = list(range(8))  # A:1, B:2, e:0

    # brute force labeling
    s_q = {}
    q_s = {}
    q_s[B.v] = []
    q_s[sinks.v] = []

    q_s[B.v].append(3)
    q_s[sinks.v].append(2)

    q_s[phi.v] = list(set(S) - set(q_s[B.v] + q_s[sinks.v]))
    q_s[whole.v] = list(set(S))

    for s in S:
        s_q[s] = []
        if s in q_s[B.v] and B.v not in s_q[s]:
            s_q[s].append(B.v)
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
    # mdp.set_P_simple()
    alpha, beta = 'l', 'r'
    # add transition probabilities
    mdp.add_trans_P(0, alpha, 1, 1)
    mdp.add_trans_P(0, beta, 4, 2/3)
    mdp.add_trans_P(0, beta, 2, 1/3)

    mdp.add_trans_P(1, alpha, 1, 1/2)
    # mdp.add_trans_P(1, alpha, 2, 7 / 18)
    # mdp.add_trans_P(1, alpha, 3, 1 / 9)


    mdp.add_trans_P(1, alpha, 3, 1 / 2)

    mdp.add_trans_P(2, alpha, 2, 1)
    mdp.add_trans_P(3, alpha, 3, 1)

    mdp.add_trans_P(4, alpha, 5, 1/4)
    mdp.add_trans_P(4, alpha, 6, 3/4)
    mdp.add_trans_P(5, beta, 2, 1/3)
    mdp.add_trans_P(5, beta, 7, 2/3)
    mdp.add_trans_P(5, alpha, 6, 1)

    mdp.add_trans_P(6, alpha, 5, 2 / 5)
    mdp.add_trans_P(6, alpha, 6, 3 / 5)

    mdp.add_trans_P(7, alpha, 2, 1/2)
    mdp.add_trans_P(7, alpha, 3, 1/2)


    mdp.set_L(s_q)  # L: labeling function (L: S -> Q), e.g. self.L[(2, 3)] = 'g1'
    mdp.set_Exp(q_s)  # L^-1: inverse of L (L: Q -> S), e.g. self.Exp['g1'] = (2, 3)
    mdp.set_Size(4, 4)

    dfa = DFA(0, [B, sinks, phi, whole])  # sink state here means energy used up outside of the C
    dfa.set_final(1)
    dfa.set_sink(2)

    dfa.pref_labeling(1, 'II')

    def inv_dict(target):
        result = {}
        for key, value in target.items():
            if value not in result:
                result[value] = [key]
            else:
                if key not in result[value]:
                    result[value].append(key)
        return result


    # dfa.inv_pref_labels = {value: key for key, value in dfa.pref_labels.items()}
    dfa.inv_pref_labels = inv_dict(dfa.pref_labels)

    sink = list(dfa.sink_states)[0]

    for i in range(sink + 1):
        dfa.add_transition(phi.display(), i, i)
        if i < sink:
            dfa.add_transition(sinks.display(), i, sink)

    dfa.add_transition(B.display(), 0, 1)
    dfa.add_transition(B.display(), 1, 1)

    dfa.add_transition(whole.display(), sink, sink)

    dfa.toDot("DFA")
    dfa.prune_eff_transition()
    # dfa.g_unsafe = 'sink'

    curve = {}
    result = mdp.product(dfa, mdp)

    result.plotKey = False
    #
    # with open("prod_MDP_simple2", 'w', encoding="utf-8") as yaml_file:
    #     yaml.dump(result.P, yaml_file)
    # # curve['action'] = result.SVI(0.001)
    #
    # with open("prod_MDP_simple2", 'r', encoding="utf-8") as yaml_file:
    #     # The FullLoader parameter handles the conversion from YAML
    #     # scalar values to Python the dictionary format
    #     saved_prod_space = yaml.load(yaml_file, Loader=yaml.FullLoader)

    with open("prod_MDP_toy.pkl", 'wb') as pkl_file: #pkl
        pickle.dump(result, pkl_file)

    print("end.")




