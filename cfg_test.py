import argparse


def get_cfg():
    parser = argparse.ArgumentParser(description="Stacking")

    parser.add_argument("--model_path", type=str, default=None, help="model file path")
    parser.add_argument("--param_path", type=str, default=None, help="hyper-parameter file path")
    parser.add_argument("--data_dir", type=str, default=None, help="test data path")
    parser.add_argument("--result_dir", type=str, default=None, help="test result file path")

    parser.add_argument("--num_piles", type=int, default=8, help="number of target piles")
    parser.add_argument("--max_height", type=int, default=15, help="maximum number of plates to be stacked in each pile")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")

    return parser.parse_args()