"""
File: toy_problem1.py
Project: preference-based-planning (https://github.com/xlrobotics/preference-based-planning)

Description: Defines classes to represent a gridworld and pygame rendering of it.
"""

import yaml
import itertools


class MDPGridWorld:
    def __init__(self, config):
        # Gridworld parameters
        self.board_size = (config['graph_size']['x'], config['graph_size']['y'])
        self.wall_coords = config['obstacles']
        self.board_wall_coords = [[self.board_size[0] - x - 1, y] for x, y in self.wall_coords]
        self.start_coord = list(config['uav']['init_state'])
        self.goal_coord = list(map(tuple, config['targets']))
        self.trapped = False
        self.battery = config['uav']['battery']
        self.battery_cap = self.battery
        self.station_coords = list(map(tuple, config['stations']))
        self.cloud_coords = config['clouds']['positions']
        self.cloud_dynamics = config['clouds']['dynamics']
        self.cloud_maximum_stay_tol = config['clouds']['maximal_stay_steps']

        # MDP parameters
        self.states = self.construct_state_space()
        self.uav_actions = config["uav"]["actions"]
        self.cloud_actions = config["clouds"]["actions"]
        self.transition = self.construct_transitions()
        self.init_state = tuple(self.start_coord)
        self.goal_states = self.goal_coord

        label_atom_state_map, label_state_atom_map = self.construct_labeling_function()
        self.label_atom_state_map = label_atom_state_map
        self.label_state_atom_map = label_state_atom_map

    def construct_state_space(self):
        state_set = set(itertools.product(
            range(self.board_size[0]),              # uav.row
            range(self.board_size[1]),              # uav.col
            range(self.board_size[0]),              # cloud1.row (cloud1.col is fixed)
            range(self.board_size[0]),              # cloud2.row (cloud2.col is fixed)
            range(self.battery_cap)                 # uav.remaining_battery
        ))

        for row, col in self.station_coords:
            state_set.update({
                (row, col, cloud1, cloud2, self.battery_cap) for cloud1, cloud2 in
                itertools.product(range(self.board_size[0]), range(self.board_size[0]))
            })

        state_set.add("sink")
        return state_set

    def construct_labeling_function(self):
        atoms = ['A', 'B', 'C', 'E']

        label_atom_state_map = dict(zip(atoms, [set() for _ in atoms]))
        label_state_atom_map = dict()

        def label(state):
            uav_pos = tuple(state[0:2])
            if uav_pos == self.goal_coord[0]:
                return {'A'}
            elif uav_pos == self.goal_coord[1]:
                return {'B'}
            elif uav_pos == self.goal_coord[2]:
                return {'C'}
            elif uav_pos in self.station_coords:
                return {'E'}
            else:
                return set()

        for st in self.states:
            lbl = label(st)
            for atom in lbl:
                label_atom_state_map[atom].add(st)
            label_state_atom_map[st] = lbl

        return label_atom_state_map, label_atom_state_map

    def construct_transitions(self):
        trans = dict()

        def apply_uav_action(uav_state, uav_action):
            new_uav_state_row = uav_state[0] + self.uav_actions[uav_action][0]
            new_uav_state_col = uav_state[1] + self.uav_actions[uav_action][1]

            # Boundary condition
            if not (0 <= new_uav_state_row < self.board_size[0] and 0 <= new_uav_state_col < self.board_size[1]):
                return uav_state

            return new_uav_state_row, new_uav_state_col

        def apply_cloud_action(cloud_state, cloud_action):
            new_cloud1_row = cloud_state[0] + self.cloud_actions[cloud_action[0]][0]
            new_cloud2_row = cloud_state[1] + self.cloud_actions[cloud_action[1]][0]

            # Boundary condition
            if not 0 <= new_cloud1_row < self.board_size[0]:
                new_cloud1_row = cloud_state[0]

            if not 0 <= new_cloud2_row < self.board_size[0]:
                new_cloud2_row = cloud_state[1]

            return (new_cloud1_row, new_cloud2_row), (1 / len(cloud_state)) ** len(cloud_state)

        for state in self.states:
            # Handle sink state
            if state == "sink":
                for uav_act in self.uav_actions:
                    trans[(state, uav_act, state)] = 1.0
                continue

            # For non-sink state, extract components
            uav_state = state[0:2]
            cloud_state = state[2:4]
            battery = state[4]

            # Apply actions
            for uav_act in self.uav_actions:
                for cloud_act in itertools.product(self.cloud_actions, self.cloud_actions):
                    if battery == 0:
                        trans[(state, uav_act, "sink")] = 1.0
                        continue

                    new_uav_state = apply_uav_action(uav_state, uav_act)
                    new_cloud_state, prob = apply_cloud_action(cloud_state, cloud_act)

                    if tuple(new_uav_state) in self.station_coords:
                        new_state = new_uav_state + new_cloud_state + (self.battery_cap,)
                    else:
                        new_state = new_uav_state + new_cloud_state + (int(battery) - 1,)

                    trans[(state, uav_act, new_state)] = prob

        return trans


def test_state_space(mdp):
    print("$ Testing MDP States")
    expect_size = (mdp.board_size[0] * mdp.board_size[1] * mdp.board_size[0] * mdp.board_size[0] * mdp.battery_cap) + \
                    (len(list(mdp.station_coords)) * mdp.board_size[0] * mdp.board_size[0]) + 1
    print("\t", f"len(gw.get_state_space()) = {len(mdp.states)}")
    print("\t", f"Expected |states| = {expect_size}")
    assert len(mdp.states) == expect_size


def inspect_transitions(mdp):
    # Select uav_state of interest
    state = (0, 0, 0, 0, 9)
    transitions = set(filter(lambda sas_pair: sas_pair[0] == state, mdp.transition))

    print()
    print(f"MDP Outgoing Transition with state = {state}")
    print("\t", f"Count = {len(transitions)}")
    for sas_pair in transitions:
        print("\t", f"{sas_pair}: {mdp.transition[sas_pair]}")


def test_transitions(mdp):
    print("$ Testing MDP Transitions")
    print("\t",
          f"P[((1, 0, 1, 1, 9), 'E', (1, 1, 2, 2, 8))] = {mdp.transition[((1, 0, 1, 1, 9), 'E', (1, 1, 2, 2, 8))]}")
    print("\t",
          f"P[((1, 0, 1, 1, 9), 'E', (1, 1, 2, 0, 8))] = {mdp.transition[((1, 0, 1, 1, 9), 'E', (1, 1, 2, 0, 8))]}")
    print("\t",
          f"P[((1, 0, 1, 1, 9), 'E', (1, 1, 0, 2, 8))] = {mdp.transition[((1, 0, 1, 1, 9), 'E', (1, 1, 0, 2, 8))]}")
    print("\t",
          f"P[((1, 0, 1, 1, 9), 'E', (1, 1, 0, 0, 8))] = {mdp.transition[((1, 0, 1, 1, 9), 'E', (1, 1, 0, 0, 8))]}")
    print("\t", f'P[((1, 0, 1, 1, 0), "E", "sink")] = {mdp.transition[((1, 0, 1, 1, 0), "E", "sink")]}')

    assert mdp.transition[((1, 0, 1, 1, 9), 'E', (1, 1, 2, 2, 8))] == 0.25
    assert mdp.transition[((1, 0, 1, 1, 9), 'E', (1, 1, 2, 0, 8))] == 0.25
    assert mdp.transition[((1, 0, 1, 1, 9), 'E', (1, 1, 0, 2, 8))] == 0.25
    assert mdp.transition[((1, 0, 1, 1, 9), 'E', (1, 1, 0, 0, 8))] == 0.25
    assert mdp.transition[((1, 0, 1, 1, 0), 'E', "sink")] == 1.0

    print("\t",
          f"P[((0, 0, 0, 0, 9), 'S', (0, 0, 0, 1, 8))] = {mdp.transition[((0, 0, 0, 0, 9), 'S', (0, 0, 0, 1, 8))]}")
    print("\t",
          f"P[((0, 0, 0, 0, 9), 'W', (0, 0, 0, 1, 8))] = {mdp.transition[((0, 0, 0, 0, 9), 'W', (0, 0, 0, 1, 8))]}")
    assert mdp.transition[((0, 0, 0, 0, 9), 'S', (0, 0, 0, 1, 8))] == 0.25
    assert mdp.transition[((0, 0, 0, 0, 9), 'W', (0, 0, 0, 1, 8))] == 0.25


def run_tests():
    # Load configuration file
    with open("toy_problem1.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    if config is None:
        raise ValueError("Configuration file could not be added.")

    # Construct gridworld
    mdp = MDPGridWorld(config)

    # Run tests
    test_state_space(mdp)
    # inspect_transitions(mdp)
    test_transitions(mdp)


def run_demo():
    # Load game configuration file
    with open("toy_problem1.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    if config is None:
        raise ValueError("Configuration file could not be added.")

    # Construct gridworld
    mdp = MDPGridWorld(config)

    # Preference Automaton
    pref_aut = generate_pref_aut()

    # Compute product MDP
    prod_mdp = product(mdp, prod_aut)

    # Save generated outputs


if __name__ == '__main__':
    run_tests()
    # run_demo()