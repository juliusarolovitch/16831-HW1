import pickle
import numpy as np

file_name = "/home/julius/rob831/16831-S25-HW/hw1/rob831/expert_data/expert_data_Humanoid-v2.pkl"

with open(file_name, "rb") as f:
    data = pickle.load(f)

returns = [np.sum(traj['reward']) for traj in data[:10]]  # Check the first 10
print("Returns of first 10 trajectories:", returns)
mean = np.mean(returns)
std = np.std(returns)

print(f"Mean: {mean}, STD: {std}")
