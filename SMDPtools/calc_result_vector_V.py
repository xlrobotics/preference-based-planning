import yaml
import pickle
import json


if __name__ == '__main__':
    with open("prod_MDP_gridworld.pkl", 'rb') as pkl_file:  # pkl
        result = pickle.load(pkl_file)

    counter = 0
    total = len(result.S)
    for s in result.S:
        print("processing key:", s, counter, "/", total, " done")
        result.init_ASR_V(s)
        counter += 1

    with open("prod_MDP_gridworld_updated.pkl", 'wb') as pkl_file: #pkl
        pickle.dump(result, pkl_file)