# Grid_World Version 1
# This program allows a player to play a memory game.
# The player selects tiles on a 4x4 board. When a tile is
# selected, an image is exposed until a second image is
# selected. If the two images match, the tiles remain
# exposed and one point is added to the score.
# If the exposed image does not match, both tiles remain
# exposed for one second, then they are hidden.
# Selecting an exposed image has no effect.
# The game ends when all tiles are exposed.
# The score is the time taken to expose all tiles so a lower
# score is better.
# This program requires 9 data files: image0 ... image8,
# where image0 is the hidden form of all tiles and image1
# ... image8 are the exposed tile images.

import pygame, sys, time, random
from pygame.locals import *
import numpy as np
import yaml
import random
import pickle
# from random import random
import logging


action_dict = {'0': "Up",
               '1': "Down",
               '2': "Right",
               '3': "Left",
               '-1': "Stay"
               }
# User-defined classes

class Tile:
    # An object in this class represents a single Tile that
    # has an image

    # initialize the class attributes that are common to all
    # tiles.

    borderColor = pygame.Color('white')
    # borderColor = (255, 255, 255) #r,g,y
    borderWidth = 1  # the pixel width of the tile border
    image_uav = pygame.image.load('uav.png')
    image_station = pygame.image.load('station.png')
    image_cloud = pygame.image.load('storm.png')
    image_goal_A = pygame.image.load('goal_A.png')
    image_goal_B = pygame.image.load('goal_B.png')
    image_goal_C = pygame.image.load('goal_C.png')
    image_charge = pygame.image.load('in_station.png')
    image_trap = pygame.image.load('in_storm.png')

    def __init__(self, x, y, wall, surface, tile_size = (32, 32)):
        # Initialize a tile to contain an image
        # - x is the int x coord of the upper left corner
        # - y is the int y coord of the upper left corner
        # - image is the pygame.Surface to display as the
        # exposed image
        # - surface is the window's pygame.Surface object

        self.wall = wall
        self.origin = (x, y)
        self.tile_coord = [x//32, y//32]
        self.surface = surface
        self.tile_size = tile_size
        rect = self.image_uav.get_rect()
        # print(rect)

    def draw(self, pos, goal, clouds, stations):
        # Draw the tile.
        # print pos, goal, self.origin, self.wall
        rectangle = pygame.Rect(self.origin, self.tile_size)

        if self.wall:
            pygame.draw.rect(self.surface, pygame.Color('gray'), rectangle, 0)
        if self.tile_coord in goal:
            # pygame.draw.rect(self.surface, (32, 0, 0), rectangle, 0) #pygame.Color('green')
            if self.tile_coord == goal[0]:
                self.surface.blit(self.image_goal_A, rectangle)
            if self.tile_coord == goal[1]:
                self.surface.blit(self.image_goal_B, rectangle)
            if self.tile_coord == goal[2]:
                self.surface.blit(self.image_goal_C, rectangle)

        if self.tile_coord in clouds:
            self.surface.blit(self.image_cloud, rectangle)
        if self.tile_coord in stations:
            self.surface.blit(self.image_station, rectangle)

        if not (self.tile_coord in goal or self.tile_coord in clouds or self.tile_coord in stations or self.tile_coord == pos):
            pygame.draw.rect(self.surface, pygame.Color('Lavender'), rectangle, 0)

        if pos == self.tile_coord:
            if self.tile_coord in clouds:
                self.surface.blit(self.image_trap, rectangle)
            elif self.tile_coord in stations:
                self.surface.blit(self.image_charge, rectangle)
            else:
                self.surface.blit(self.image_uav, rectangle)
            # pygame.blit(IMAGE, rect)
            # self.surface.blit(Tile.image, self.origin)

        pygame.draw.rect(self.surface, Tile.borderColor, rectangle, Tile.borderWidth)


class Grid_World():
    # An object in this class represents a Grid_World game.
    tile_width = 32
    tile_height = 32

    def __init__(self, surface, **kwargs):
        # Intialize a Grid_World game.
        # - surface is the pygame.Surface of the window

        self.surface = surface
        self.bgColor = pygame.Color('blue')

        self.board_size = list([kwargs['graph_size']['x'], kwargs['graph_size']['y']])
        self.wall_coords = kwargs['obstacles']
        self.visit = []
        self.cumulated_heat = np.zeros(self.board_size)

        self.wall_coords = []

        self.start_coord = list(kwargs['UAV']['init_state'])
        self.trapped = False
        self.goal_coord = list(kwargs['targets'])
        self.position = self.start_coord[:]
        self.battery = kwargs['UAV']['battery']
        self.battery_cap = self.battery

        self.rendering = kwargs['rendering']

        self.station_coords = list(kwargs['stations'])
        self.station_coords_tuple = []
        for element in self.station_coords:
            self.station_coords_tuple.append(tuple(element))

        self.cloud_coords = kwargs['clouds']['positions']
        self.cloud_dynamics = kwargs['clouds']['dynamics']
        self.cloud_maximum_stay_tol = kwargs['clouds']['maximal_stay_steps']
        self.cloud_stay_counter = np.zeros(len(self.cloud_coords))

        self.actions = range(4)
        self.action_map = {'N':0, 'S':1, 'W':3, 'E':2, 'stay':-1}
        self.cloud_actions = [-1, 0, 1]
        self.reward = 0
        self.manual = kwargs['manual']
        self.sample_path = kwargs['sample_path']


        self.calc_wall_coords()
        self.createTiles()

    def get_state_space(self, dtype="list"):
        state_set = []
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                for c1_i in range(self.board_size[0]):
                    for b in range(self.battery_cap):
                        if dtype=="list":
                            state_set.append([i, j, c1_i, b]) # each cloud only moves in one axis
                        elif dtype=="tuple":
                            state_set.append(tuple([i, j, c1_i, b]))  # each cloud only moves in one axis

        for station in self.station_coords: # adding full battery cap states, only at station
            for c1_i in range(self.board_size[0]):
                if dtype == "list":
                    state_set.append(station+[c1_i, self.battery_cap])
                elif dtype=="tuple":
                    state_set.append(tuple(station + [c1_i, self.battery_cap]))
        return state_set

    def get_current_state_old(self):  # 5 dimension (uav.x, uav.y, c1.x, c2.x, battery)
        temp = self.position[:]
        for element in self.cloud_coords:
            temp.append(element[0])

        temp.append(self.battery)
        return temp

    def calc_wall_coords(self):
        self.board_wall_coords = [[self.board_size[0] - x - 1, y] for x, y in self.wall_coords]

    def find_board_coords(self, pos):
        x = pos[1]
        y = self.board_size[0] - pos[0] -1
        # y = pos[1] #self.board_size[0] - pos[0] - 1
        return [x, y]

    def find_board_coords_multi(self, pos):
        result_goals = []
        for element in pos:
            x = element[1]
            y = self.board_size[0] - element[0] -1
            result_goals.append([x, y])
        return result_goals

    def createTiles(self):
        # Create the Tiles
        # - self is the Grid_World game
        self.board = []
        for rowIndex in range(0, self.board_size[0]):
            row = []
            for columnIndex in range(0, self.board_size[1]):
                imageIndex = rowIndex * self.board_size[1] + columnIndex
                x = columnIndex * Grid_World.tile_width
                y = rowIndex * Grid_World.tile_height
                if [rowIndex, columnIndex] in self.board_wall_coords:
                    wall = True
                else:
                    wall = False
                tile = Tile(x, y, wall, self.surface)
                row.append(tile)
            self.board.append(row)

    def draw(self):
        # Draw the tiles.
        # - self is the Grid_World game
        uav_pos = self.find_board_coords(self.position)
        goals = self.find_board_coords_multi(self.goal_coord)
        clouds = self.find_board_coords_multi(self.cloud_coords)
        stations = self.find_board_coords_multi(self.station_coords)
        self.surface.fill(self.bgColor)
        for row in self.board:
            for tile in row:
                tile.draw(uav_pos, goals, clouds, stations)


    def update(self):
        # Check if the game is over. If so return True.
        # If the game is not over,  draw the board
        # and return False.
        # - self is the TTT game
        self.step_cloud()
        # print(self.cloud_coords)
        # self.step(0)
        if self.battery == 0:
            return True
        else:
            self.draw()
            return False

    def step_cloud(self):
        # determine the cloud dynamics
        # print(self.cloud_coords[0])
        if self.cloud_dynamics == 'random_walk':
            for i in range(len(self.cloud_coords)):

                act = random.choice([-1, 1])  # 0.5 probability of choosing each action
                self.cloud_stay_counter[i] = 0

                if act == 0:
                    self.cloud_stay_counter[i] += 1
                else:
                    temp_new_pos = self.cloud_coords[i][0] + act
                    temp_new_pos_ = self.cloud_coords[i][0] - act
                    if temp_new_pos < 0 or temp_new_pos > self.board_size[0]-1:
                        self.cloud_coords[i][0] = temp_new_pos_
                    else:
                        self.cloud_coords[i][0] = temp_new_pos
        return False


    def step(self, action):
        Done = False
        x, y = self.position
        if self.position in self.cloud_coords:
            self.trapped = True
        else:
            self.trapped = False

        step_back = self.position

        self.battery -= 1

        if self.trapped == True:
            if self.battery == 0:
                logging.warning("battery depleted")
                Done = True
            state = self.get_current_state_old()
            return state, Done


        if action == 0:   # Action Up
            # print("Move Up")
            if [x+1, y] not in self.wall_coords and x+1 < self.board_size[0] and not self.trapped:
                self.position = [x+1, y]
            elif x+1 >= self.board_size[0]:  # world boundary, bounce back
                self.position = [x-1, y]
            # if x+1 > self.board_size[0]:
            #     self.position = [x-1, y]

        elif action == 1:   # Action Down
            # print("Move Down")
            if [x-1, y] not in self.wall_coords and x-1 >= 0 and not self.trapped:
                self.position = [x-1, y]
            elif x-1 < 0:
                self.position = [x+1, y]

        elif action == 2:   # Action Right
            # print("Move Right")
            if [x, y+1] not in self.wall_coords and y+1 < self.board_size[1] and not self.trapped:
                self.position = [x, y+1]
            elif y+1 >= self.board_size[1]:
                self.position = [x, y-1]

        elif action == 3:   # Action Left
            # print("Move Left")
            if [x, y-1] not in self.wall_coords and y-1 >= 0 and not self.trapped:
                self.position = [x, y-1]
            elif y-1 < 0:
                self.position = [x, y+1]


        elif action == -1:
            pass

        if self.position in self.station_coords:
            self.battery = self.battery_cap

        if self.battery == 0:
            logging.warning("battery depleted")
            Done = True
        # state = self.get_current_state()
        state = self.get_current_state_old()

        return state, Done



        #print action_dict[str(action)], self.position
    def get_reward(self, step_back, pos):
        if pos == self.goal_coord:
            self.reward = 3200
        elif pos in self.wall_coords:
            self.position = step_back
            self.reward = -10
        else:
            if self.check_door():
                # print self.position
                self.reward = 0
            else:
                self.reward = 0

    def check_door(self): # manually check whether the agent is passing through a door between two rooms
        p = self.position
        x, y = p

        if p not in self.wall_coords and p != self.goal_coord:
            if [x+1, y] in self.wall_coords and [x-1, y] in self.wall_coords and [x, y+1] not in self.wall_coords and [x, y-1] not in self.wall_coords:
                return 1
            if [x+1, y] not in self.wall_coords and [x-1, y] not in self.wall_coords and [x, y+1] in self.wall_coords and [x, y-1] in self.wall_coords:
                return 1

        return 0

    def change_the_wall(self, wall_coords):
        self.wall_coords = wall_coords
        self.calc_wall_coords()
        self.createTiles()

    def change_the_goal(self, goal):
        self.goal_coord = list(goal)


def trans_parser(obs, board, trans, current_q, model):   # because a single observed mdp (original, not product) state might be mapped to multiple transitions for the labeling function, needs further parsing
    depleted = False
    at_base = False
    at_A = False
    at_B = False
    at_C = False
    if obs[3] == 0:
        depleted = True
    if obs[0:2] in board.station_coords:
        at_base = True
    if obs[0:2] == board.goal_coord[0]:
        at_A = True
    if obs[0:2] == board.goal_coord[1]:
        at_B = True
    if obs[0:2] == board.goal_coord[2]:
        at_C = True

    if current_q in [7, 8, 9]:
        return model.dfa.state_transitions[tuple(['1', current_q])], '1'

    if depleted:
        if 'dead' in trans:
            return model.dfa.state_transitions[tuple(['dead', current_q])], 'dead'
        elif 'base|dead' in trans:
            return model.dfa.state_transitions[tuple(['base|dead', current_q])], 'base|dead'

    if at_base:
        if 'base' in trans:
            return model.dfa.state_transitions[tuple(['base', current_q])], 'base'
        elif 'base|dead' in trans:
            return model.dfa.state_transitions[tuple(['base|dead', current_q])], 'base|dead'

    out_trans = 'None'
    if at_A:
        out_trans = 'a&!dead'
    elif at_B:
        out_trans = 'b&!dead'
    elif at_C:
        out_trans = 'c&!dead'

    if out_trans not in model.dfa.state_trans_dict[current_q]:
        out_trans = list(model.dfa.state_loop[current_q])[0]

    return model.dfa.state_transitions[tuple([out_trans, current_q])], out_trans



if __name__ == "__main__":
    # Initialize pygame
    pygame.init()

    # Set window size and title, and frame delay
    surfaceSize = (32*4, 32*4)
    windowTitle = 'UAV Grid World'
    pauseTime = 1 #0.01  # smaller is faster game

    with open("toy_config.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    # Create the window
    surface = pygame.display.set_mode(surfaceSize, 0, 0)
    pygame.display.set_caption(windowTitle)

    # create and initialize objects
    gameOver = False

    # print surface
    board = Grid_World(surface, **data_loaded)

    # Draw objects
    board.draw()

    # Refresh the display
    pygame.display.update()

    with open("PI_prod_MDP_gridworld_v2.pkl", 'rb') as pkl_file:  # pkl
        pi = pickle.load(pkl_file)

    with open("prod_MDP_gridworld_v2.pkl", 'rb') as pkl_file:  # pkl
        model = pickle.load(pkl_file)
    #
    # for key in pi.keys():
    #     if len(pi[key]) >0:
    #         print(key, pi[key])

    init_obs = board.get_current_state_old()
    # init_obs = board.get_current_state()
    # for key in pi:
    #     if list(key[0][0:2]) == board.goal_coord[2] and key[0][3] == 1:
    #         print(key, pi[key])
    # print('--------------------------')
    # for key2 in model.P:
    #     # print(key2[0][0][0:2])
    #     if list(key2[0][0][0:2]) == board.goal_coord[2] and key2[0][0][3] == 1:
    #         print(key2, model.P[key2])

    init_q = 6
    init_obs[3] = board.battery_cap - 1
    act_set = pi[tuple([tuple(init_obs), init_q, 'improved'])]
    gameOver = False

    if len(act_set) == 0:
        logging.error('failing state %s detected, task terminated', init_obs)
        gameOver = True
        # logging.error('failing state %s detected, random action selected', init_obs)
        # act_set = set(['S', 'N', 'E', 'W', 'stay'])
        # act = random.choice(list(act_set))
        # gameOver = True
    else:
        act = random.choice(list(act_set))

    i = 0
    current_q = init_q
    obs = init_obs
    pref_level = 'bottom'
    while not gameOver:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
                # Handle additional events

        # Update and draw objects for next frame


        old_pos = board.position
        obs, gameOver = board.step(board.action_map[act])
        new_pos = board.position
        # print("taking action: ", act, ' from ', old_pos, ' to ', new_pos, ' (trapping status: ', board.trapped, ')')

        trans = model.mdp.L[tuple(obs)]

        current_q, result_trans = trans_parser(obs, board, trans, current_q, model)



        # current_q = model.dfa.state_transitions[tuple([trans[0], current_q])]

        if current_q in model.dfa.pref_labels:
            pref_level = model.dfa.pref_labels[current_q]
            logging.warning(
                'Improving action set: %s. Taking action: %s from %s to %s, trapping status - %s. Transition: %s, current state (s, q): (%s, %s), pref_level: %s',
                act_set, act, old_pos, new_pos, board.trapped, result_trans, obs, current_q, model.dfa.pref_labels[current_q])
        else:
            logging.warning(
                'Improving action set: %s. Taking action: %s from %s to %s, trapping status - %s. Transition: %s, current state (s, q): (%s, %s), pref_level: None',
                act_set, act, old_pos, new_pos, board.trapped, result_trans, obs, current_q)


        board.update()

        try:
            act_set = pi[tuple([tuple(obs), current_q, 'improved'])]
        except KeyError:
            act_set = pi[tuple([tuple(obs), current_q])]

        if not gameOver:

            if current_q in model.dfa.sink_states:
                print('sink state detected, task terminated at state', obs)
                gameOver = True
            elif pref_level is not 'bottom' and len(model.dfa.inv_pref_trans[pref_level]) == 0:
                print('already satisfied top preference ', pref_level,', task terminated at state', obs)
                gameOver = True
            elif pref_level is not 'bottom' and result_trans == 'base':
                print('already satisfied preference ', pref_level, ' and back to base, task terminated at state', obs)
                gameOver = True
            elif len(act_set) == 0:
                print('no improving policy, try non-improving policy instead at state', obs)
                act_set = pi[tuple([tuple(obs), current_q])]
                if len(act_set) == 0:
                    print('no available policy, task terminated at state', obs)
                    gameOver = True
            else:
                act = random.choice(list(act_set))

        if gameOver:
            break

        i = i+1
        if i == len(board.sample_path):
            i = 0
        # Refresh the display
        pygame.display.update()

        # Set the frame speed by pausing between frames
        time.sleep(pauseTime)


