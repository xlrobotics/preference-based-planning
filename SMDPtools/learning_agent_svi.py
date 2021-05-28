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
    station = Action('e')
    phi = Action('phi')
    whole = Action('whole')
    sinks = Action('sink') # set of all 0 energy states
    trial_number = 50
    states = board.get_state_space()
    S = board.get_state_space("tuple")

    # brute force labeling
    s_q = {}
    q_s = {}
    q_s[A.v] = []
    q_s[B.v] = []
    q_s[C.v] = []
    q_s[station.v] = []
    q_s[sinks.v] = []

    for c1_i in range(board.board_size[0]):
        for c2_i in range(board.board_size[0]):
            for b in range(board.battery_cap):
                q_s[A.v].append(tuple(board.goal_coord[0]+[c1_i, c2_i, b]))
                q_s[B.v].append(tuple(board.goal_coord[1]+[c1_i, c2_i, b]))
                q_s[C.v].append(tuple(board.goal_coord[2]+[c1_i, c2_i, b]))
                q_s[station.v].append(tuple(board.station_coords_tuple[:]+[c1_i, c2_i, b]))

    impossible_states = []
    for i in range(board.board_size[0]):
        for j in range(board.board_size[1]):
            for c1_i in range(board.board_size[0]):
                for c2_i in range(board.board_size[0]):
                    if [i, j] not in board.station_coords:
                        q_s[sinks.v].append(tuple([i, j, c1_i, c2_i, 0])) # when the uav used up its battery outside of the station
                    else:
                        for k in range(board.battery_cap-1):
                            impossible_states.append(tuple([i, j, c1_i, c2_i, k])) #when the uav is at the station, its power could only be full

    q_s[phi.v] = list(set(S) - set(q_s[A.v] + q_s[B.v] + q_s[C.v] + q_s[station.v] + q_s[sinks.v] + impossible_states))
    q_s[whole.v] = list(set(S) - set(impossible_states))

    for s in S:
        s_q[s] = []
        if s in q_s[A.v] and A.v not in s_q[s]:
            s_q[s].append(A.v)
        if s in q_s[B.v] and B.v not in s_q[s]:
            s_q[s].append(B.v)
        if s in q_s[C.v] and C.v not in s_q[s]:
            s_q[s].append(C.v)
        if s in q_s[station.v] and station.v not in s_q[s]:
            s_q[s] = [station.v]
        if s in q_s[sinks.v] and sinks.v not in s_q[s]:
            s_q[s].append(sinks.v)
        if s in q_s[phi.v] and phi.v not in s_q[s]:
            s_q[s].append(phi.v)

    mdp = MDP()
    mdp.set_S(states)
    mdp.set_WallCord(mdp.add_wall(q_s[sinks.v]))
    mdp.set_stations(board.station_coords)
    mdp.set_stations(board.goal_coord)
    mdp.set_full_tank(board.battery_cap)
    mdp.set_P()
    mdp.set_L(s_q)  # L: labeling function (L: S -> Q), e.g. self.L[(2, 3)] = 'g1'
    mdp.set_Exp(q_s)  # L^-1: inverse of L (L: Q -> S), e.g. self.Exp['g1'] = (2, 3)
    mdp.set_Size(4, 4)

    dfa = DFA(0, [A, B, C, station, sinks, phi, whole]) # sink state here means energy used up outside of the station
    dfa.set_final(8)
    dfa.set_final(9)
    dfa.set_final(10)
    dfa.set_final(11)
    dfa.set_final(12)
    dfa.set_final(13)
    dfa.set_sink(14)

    sink = list(dfa.sink_states)[0]

    for i in range(sink+1):
        dfa.add_transition(phi.display(), i, i)
        if i < sink:
            dfa.add_transition(sinks.display(), i, sink)

    dfa.add_transition(whole.display(), sink, sink)

    dfa.add_transition(A.display(), 0, 1)
    dfa.add_transition(B.display(), 0, 2)
    dfa.add_transition(C.display(), 0, 3)
    dfa.add_transition(station.display(), 0, 13)

    dfa.add_transition(B.display(), 1, 4)
    dfa.add_transition(C.display(), 1, 5)
    dfa.add_transition(station.display(), 1, 12)

    dfa.add_transition(A.display(), 2, 4)
    dfa.add_transition(C.display(), 2, 6)
    dfa.add_transition(station.display(), 2, 12)

    dfa.add_transition(A.display(), 3, 5)
    dfa.add_transition(B.display(), 3, 6)
    dfa.add_transition(station.display(), 3, 12)

    dfa.add_transition(C.display(), 4, 7)
    dfa.add_transition(station.display(), 4, 8)
    dfa.add_transition(B.display(), 5, 7)
    dfa.add_transition(station.display(), 5, 10)
    dfa.add_transition(A.display(), 6, 7)
    dfa.add_transition(station.display(), 6, 11)
    dfa.add_transition(station.display(), 7, 8)

    dfa.toDot("DFA")
    dfa.prune_eff_transition()
    dfa.g_unsafe = 'sink'

    curve = {}
    result = mdp.product(dfa, mdp)
    result.plotKey = False
    curve['action'] = result.SVI(0.001)

    # setup action based SVI agent
    # agent = SVI(alpha=0.5, gamma=0.9, lmbda=0.0, epsilon=0.1, n=n, num_actions=4, room=4)
    # agent.set_S()
    # # agent.set_P()
    # agent.set_goal(board.goal_coord)
    # agent.set_WallCord(board.wall_coords)
    # agent.set_P()
    #
    # blocks = agent.NB_initiation()
    #
    # room_agent = {}
    # room_path = {}
    #
    # room_path = agent.get_RoomPath()
    # # print room_path
    #
    # # encapsulate room information into room object, also known as option for higher level agent
    # for key in room_path.keys():
    #     # print key
    #     room_agent[key] = SVI(alpha=0.5, gamma=0.9, lmbda=0.0, epsilon=0.1, n=n, num_actions=4, room=1)
    #     room_agent[key].set_S(blocks[key[0]]["inner"]+blocks[key[0]]["wall_cords"])
    #
    #     room_agent[key].set_goal(list(room_path[key]))
    #     room_agent[key].set_WallCord(blocks[key[0]]["wall_cords"])
    #     room_agent[key].set_P()
    #
    #     room_agent[key].init_V()
    #     # room_agent[key].init_Policy()
    #     # room_agent[key].plot("room"+str(key[0]))
    #     room_agent[key].SVI(threshold)
    #
    #     room_agent[key].transition_matrix()
    #
    #     # room_agent[key].plot(key[0]+"to"+key[1])
    #
    #     agent.add_Opt(key, room_agent[key])
    #
    # # run option_svi
    # agent.init_V()
    # agent.init_Policy()
    # agent.SVI(threshold, "Option")
    #
    # agent.init_V()
    # agent.init_Policy()
    # agent.Ifplot = True
    # agent.SVI(threshold)
    # # agent.init_V()
    #
    # agent.plot_curve(agent.action_diff, agent.hybrid_diff,  "compare diff")
    # agent.plot_curve(agent.action_val, agent.hybrid_val, "compare val")
    # agent.plot_curve(agent.action_pi_diff, agent.hybrid_pi_diff, "compare pi")







