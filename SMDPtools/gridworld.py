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
# from random import random


action_dict = {'0': "Up",
               '1': "Down",
               '2': "Right",
               '3': "Left",
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
    image_goal = pygame.image.load('goal.png')
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
            self.surface.blit(self.image_goal, rectangle)
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
        self.goal_coord = list(kwargs['targets'])
        self.position = self.start_coord[:]
        self.battery = kwargs['UAV']['battery']
        self.battery_cap = self.battery

        self.station_coords = list(kwargs['stations'])

        self.cloud_coords = kwargs['clouds']['positions']
        self.cloud_dynamics = kwargs['clouds']['dynamics']
        self.cloud_maximum_stay_tol = kwargs['clouds']['maximal_stay_steps']
        self.cloud_stay_counter = np.zeros(len(self.cloud_coords))

        self.actions = range(4)
        self.cloud_actions = [-1, 0, 1]
        self.reward = 0
        self.manual = kwargs['manual']
        self.sample_path = kwargs['sample_path']

        self.calc_wall_coords()
        self.createTiles()

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

    def draw_heat(self, pos):
        # print pos
        gain = self.get_reward(pos, pos)

        if gain == 0:
            gain = 1
        if gain < 0:
            gain = -1

        p = tuple(pos)
        self.cumulated_heat[p] += gain

        # print self.cumulated_heat[pos]

        if self.cumulated_heat[p] < 0:
            self.cumulated_heat[p] = 0
        if self.cumulated_heat[p] > 205:
            self.cumulated_heat[p] = 205

        tempColor = (255 - self.cumulated_heat[p], 0, 0)
        rectangle = pygame.Rect(p, (32, 32))
        pygame.draw.rect(self.surface, tempColor, rectangle, 0)

    def update(self):
        # Check if the game is over. If so return True.
        # If the game is not over,  draw the board
        # and return False.
        # - self is the TTT game
        self.step_cloud()
        print(self.cloud_coords)
        # self.step(0)
        if self.battery == 0:
            return True
        else:
            self.draw()
            return False

    def step_cloud(self):
        # determine the cloud dynamics
        if self.cloud_dynamics == 'random_walk':
            for i in range(len(self.cloud_coords)):
                if self.cloud_stay_counter[i] >= self.cloud_maximum_stay_tol:
                    act = random.choice([-1, 1])
                    self.cloud_stay_counter[i] = 0
                else:
                    act = random.choice(self.cloud_actions)

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
        x, y = self.position

        step_back = self.position

        self.battery -= 1

        if action == 0:   # Action Up
            # print("Move Up")
            if [x+1, y] not in self.wall_coords and x+1 < self.board_size[0]:
                self.position = [x+1, y]
            # if x+1 > self.board_size[0]:
            #     self.position = [x-1, y]

        elif action == 1:   # Action Down
            # print("Move Down")
            if [x-1, y] not in self.wall_coords and x-1 >= 0:
                self.position = [x-1, y]

        elif action == 2:   # Action Right
            # print("Move Right")
            if [x, y+1] not in self.wall_coords and y+1 < self.board_size[1]:
                self.position = [x, y+1]

        elif action == 3:   # Action Left
            # print("Move Left")
            if [x, y-1] not in self.wall_coords and y-1 >= 0:
                self.position = [x, y-1]

        if self.position in self.station_coords:
            self.battery = self.battery_cap
        # Reward definition
        # self.get_reward(step_back, self.position)

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

    # Loop forever
    i = 0
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
                # Handle additional events

        # Update and draw objects for next frame
        board.step(board.sample_path[i])
        gameOver = board.update()
        if gameOver:
            break

        i = i+1
        if i == len(board.sample_path):
            i = 0
        # Refresh the display
        pygame.display.update()

        # Set the frame speed by pausing between frames
        time.sleep(pauseTime)


