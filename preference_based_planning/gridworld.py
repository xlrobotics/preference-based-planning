"""
File: gridworld.py
Project: preference-based-planning (https://github.com/xlrobotics/preference-based-planning)

Description: Defines classes to represent a gridworld and pygame rendering of it.
"""

import yaml
import itertools


class GridWorld:
    def __init__(self, config):
        self.board_size = (config['graph_size']['x'], config['graph_size']['y'])
        self.wall_coords = config['obstacles']
        self.board_wall_coords = [[self.board_size[0] - x - 1, y] for x, y in self.wall_coords]
        self.start_coord = list(config['UAV']['init_state'])
        self.goal_coord = list(config['targets'])
        self.trapped = False
        self.battery = config['UAV']['battery']
        self.battery_cap = self.battery
        self.station_coords = list(map(tuple, config['stations']))
        self.cloud_coords = config['clouds']['positions']
        self.cloud_dynamics = config['clouds']['dynamics']
        self.cloud_maximum_stay_tol = config['clouds']['maximal_stay_steps']
        self.cloud_stay_counter = 0
        self.actions = range(4)
        self.action_map = {'N': 0, 'S': 1, 'W': 3, 'E': 2, 'stay': -1}
        self.cloud_actions = [-1, 0, 1]

    def get_state_space(self):
        state_set = set(itertools.product(
            range(self.board_size[0]),              # uav.row
            range(self.board_size[1]),              # uav.col
            range(self.board_size[0]),              # cloud1.row (cloud1.col is fixed)
            range(self.board_size[0]),              # cloud2.row (cloud2.col is fixed)
            range(self.battery_cap)                 # uav.remaining_battery
        ))
        print(len(state_set))

        for row, col in self.station_coords:
            state_set.update({
                (row, col, cloud1, cloud2, self.battery_cap) for cloud1, cloud2 in
                itertools.product(range(self.board_size[0]), range(self.board_size[0]))
            })
        print(len(state_set))

        return state_set


def test_state_space(gw):
    expected_size = (gw.board_size[0] * gw.board_size[1] * gw.board_size[0] * gw.board_size[0] * gw.battery_cap) + \
           (len(list(gw.station_coords)) * gw.board_size[0] * gw.board_size[0])
    print(f"len(gw.get_state_space()) = {len(gw.get_state_space())}")
    print(f"Expected |states| = {expected_size}")


if __name__ == '__main__':
    # Load configuration file
    with open("toy_problem1.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    if config is None:
        raise ValueError("Configuration file could not be added.")

    # Construct gridworld
    gw = GridWorld(config)

    # Run tests
    test_state_space(gw)