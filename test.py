import os
import time
import json
import random
import torch
import pandas as pd

from environment.env import Stacking
from agent.network import Network
from agent.heuristics import *
from cfg_test import get_cfg


if __name__ == '__main__':
    args = get_cfg()

    num_piles = args.num_piles
    max_height = args.max_height
    random_seed = args.random_seed

    model_path = args.model_path
    param_path = args.param_path

    data_dir = args.data_dir
    result_dir = args.result_dir

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    data_paths = os.listdir(data_dir)
    index = ["P%d" % i for i in range(1, len(data_paths) + 1)] + ["avg"]
    columns = ["MC", "DC", "FO", "RAND"]
    df_move = pd.DataFrame(index=index, columns=columns)
    df_time = pd.DataFrame(index=index, columns=columns)

    for name in columns:
        progress = 0
        moves = []
        times = []

        for prob, path in zip(index, data_paths):
            random.seed(random_seed)

            env = Stacking(data_dir + path, num_piles=num_piles, max_height=max_height)

            if name == "RL":
                with open(param_path, 'r') as f:
                    parameters = json.load(f)

                agent = Network(env.state_size, env.action_size, parameters["n_units"]).to(torch.device('cpu'))
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                agent.load_state_dict(checkpoint['model_state_dict'])

            start = time()
            state, mask = env.reset()

            if name == "RL":
                h_in = np.zeros((1, parameters["n_units"]))
                c_in = np.zeros((1, parameters["n_units"]))

            while True:
                if name == "RL":
                    action, _, _, h_out, c_out = agent.act(state, mask, h_in, c_in, greedy=False)
                elif name == "MC":
                    action = minimize_conflicts(state, mask)
                elif name == "DC":
                    action = delay_conflicts(state, mask)
                elif name == "FO":
                    action = flexibility_optimization(state, mask)
                elif name == "RAND":
                    action = random_stacking(state, mask)
                next_state, reward, done, next_mask = env.step(action)

                state = next_state
                mask = next_mask

                if name == "RL":
                    h_in = h_out
                    c_in = c_out

                if done:
                    finish = time()
                    times.append(finish - start)
                    moves.append(env.crane_move)
                    break

            progress += 1
            print("%d/%d test for %s done" % (progress, len(index) - 1, name))

        df_move[name] = moves + [sum(moves) / len(moves)]
        df_time[name] = times + [sum(times) / len(times)]
        print("==========test for %s finished==========" % name)

    writer = pd.ExcelWriter(result_dir + 'test_results.xlsx')
    df_move.to_excel(writer, sheet_name="move")
    df_time.to_excel(writer, sheet_name="time")
    writer.close()