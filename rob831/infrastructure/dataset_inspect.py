import pickle
import numpy as np
import argparse

def compute_returns(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    trajectory_returns = [sum(traj["reward"]) for traj in data]

    mean_return = np.mean(trajectory_returns)
    std_return = np.std(trajectory_returns)  

    print(f"Mean Return: {mean_return:.2f}")
    print(f"Standard Deviation of Returns: {std_return:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()

    compute_returns(args.file_path)
