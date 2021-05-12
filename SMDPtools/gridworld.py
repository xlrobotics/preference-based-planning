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

    borderColor = pygame.Color('black')
    # borderColor = (255, 255, 255) #r,g,y
    borderWidth = 4  # the pixel width of the tile border
    image = pygame.image.load('marvin.jpg')

    def __init__(self, x, y, wall, surface, tile_size = (50, 50)):
        # Initialize a tile to contain an image
        # - x is the int x coord of the upper left corner
        # - y is the int y coord of the upper left corner
        # - image is the pygame.Surface to display as the
        # exposed image
        # - surface is the window's pygame.Surface object

        self.wall = wall
        self.origin = (x, y)
        self.tile_coord = [x//50, y//50]
        self.surface = surface
        self.tile_size = tile_size

    def draw(self, pos, goal):
        # Draw the tile.
        # print pos, goal, self.origin, self.wall

        rectangle = pygame.Rect(self.origin, self.tile_size)

        if self.wall:
            pygame.draw.rect(self.surface, pygame.Color('gray'), rectangle, 0)
        elif goal == self.tile_coord:
            pygame.draw.rect(self.surface, (50, 0, 0), rectangle, 0) #pygame.Color('green')
        else:
            pygame.draw.rect(self.surface, pygame.Color('white'), rectangle, 0)

        if pos == self.tile_coord:
            pygame.draw.rect(self.surface, pygame.Color('yellow'), rectangle, 0)
            # self.surface.blit(Tile.image, self.origin)




        pygame.draw.rect(self.surface, Tile.borderColor, rectangle, Tile.borderWidth)


class Grid_World():
    # An object in this class represents a Grid_World game.
    tile_width = 50
    tile_height = 50

    def __init__(self, surface, board_size = (13,13), wall_coords=[], start_coord=(2,3), goal_coord=(9,9)):
        # Intialize a Grid_World game.
        # - surface is the pygame.Surface of the window

        self.surface = surface
        self.bgColor = pygame.Color('blue')

        self.board_size = list(board_size)
        self.wall_coords = wall_coords
        self.visit = []
        self.cumulated_heat = np.zeros(board_size)

        if not self.wall_coords:
            self.wall_coords = [[0, i] for i in range(board_size[1])]
            self.wall_coords += [[12, i] for i in range(board_size[1])]
            self.wall_coords += [[j, 0] for j in range(board_size[0])]
            self.wall_coords += [[j, 12] for j in range(board_size[0])]

            self.wall_coords += [[j, 6] for j in range(board_size[0])]
            self.wall_coords += [[6, i] for i in range(0, 6)]

            self.wall_coords += [[7, i] for i in range(6, board_size[1])]

            self.wall_coords.remove([3, 6])
            self.wall_coords.remove([10, 6])
            self.wall_coords.remove([6, 2])
            self.wall_coords.remove([7, 9])

        # self.wall_coords = []



        # else:
        #     self.wall_coords = wall_coords


        self.start_coord = list(start_coord)
        self.goal_coord = list(goal_coord)
        self.position = list(start_coord)
        self.actions = range(4)
        self.reward = 0

        self.calc_wall_coords()
        self.createTiles()

    def calc_wall_coords(self):
        self.board_wall_coords = [[self.board_size[0] - x - 1, y] for x, y in self.wall_coords]

    def find_board_coords(self, pos):
        x = pos[1]
        y = self.board_size[0] - pos[0] -1
        return [x, y]

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
        pos = self.find_board_coords(self.position)
        goal = self.find_board_coords(self.goal_coord)
        self.surface.fill(self.bgColor)
        for row in self.board:
            for tile in row:
                tile.draw(pos, goal)

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
        rectangle = pygame.Rect(p, (50, 50))
        pygame.draw.rect(self.surface, tempColor, rectangle, 0)

    def update(self):
        # Check if the game is over. If so return True.
        # If the game is not over,  draw the board
        # and return False.
        # - self is the TTT game

        if self.position == self.goal_coord:
            return True
        else:
            self.draw()
            return False

    def step(self, action):
        x, y = self.position

        step_back = self.position

        if action == 0:   # Action Up
            # print "Up"
            # if [x+1, y] not in self.wall_coords and x+1 < self.board_size[0]:
            self.position = [x+1, y]

        elif action == 1:   # Action Down
            # print "Down"
            # if [x-1, y] not in self.wall_coords and x-1 >= 0:
            self.position = [x-1, y]

        elif action == 2:   # Action Right
            # print "Right"
            # if [x, y+1] not in self.wall_coords and y+1 < self.board_size[1]:
            self.position = [x, y+1]

        elif action == 3:   # Action Left
            # print "Left"
            # if [x, y-1] not in self.wall_coords and y-1 >= 0:
            self.position = [x, y-1]

        # Reward definition
        self.get_reward(step_back, self.position)

        #print action_dict[str(action)], self.position
    def get_reward(self, step_back, pos):
        if pos == self.goal_coord:
            self.reward = 5000
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
    surfaceSize = (50*13, 50*13)
    windowTitle = 'Grid_World'
    pauseTime = 1  # smaller is faster game

    # Create the window
    surface = pygame.display.set_mode(surfaceSize, 0, 0)
    pygame.display.set_caption(windowTitle)

    # create and initialize objects
    gameOver = False

    # print surface
    board = Grid_World(surface)

    # Draw objects
    board.draw()

    # Refresh the display
    pygame.display.update()

    # Loop forever
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
                # Handle additional events

        # Update and draw objects for next frame
        gameOver = board.update()
        if gameOver:
            break

        # Refresh the display
        pygame.display.update()

        # Set the frame speed by pausing between frames
        time.sleep(pauseTime)


