import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv
import numpy as np

np.random.seed(0)
import seaborn as sns

sns.set_theme()
import utils
import pickle as pkl
import gzip
import os

import torch
import numpy as np
import utils
from model import ACModel, ACModel_adversary
from torch.distributions.categorical import Categorical


# Parse arguments

parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True, help="name of the trained model (REQUIRED)")
parser.add_argument("--budget", required=True, help="budget for the adversary")

parser.add_argument("--stochasticity", required=True, help="amount of stochasticity")
parser.add_argument("--type", required=True, help="whether with 2m or 5m trajectories in the training")
parser.add_argument("--cost", required=True, help="cost for each moove")
parser.add_argument("--episodes", type=int, default=100, help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16, help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False, help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10, help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False, help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False, help="add a GRU to the model")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environments

envs = []
height = None
width = None
for i in range(args.procs):
    env = utils.make_env(args.budget, args.cost, args.seed + 10000 * i)
    height = env.height
    width = env.width
    envs.append(env)
env = ParallelEnv(envs)
print("Environments loaded\n")


# # Load agent

model_dir = utils.get_model_dir(args.model)

# Initialize logs

logs = {"num_frames_per_episode": [], "return_per_episode": []}

# Run agent

start_time = time.time()

obss = env.reset()

log_done_counter = 0
log_episode_return = torch.zeros(args.procs, device=device)
log_episode_num_frames = torch.zeros(args.procs, device=device)

perturbation_map = np.zeros((envs[0].width - 2, envs[0].height - 2))
counter_map = np.zeros((envs[0].width - 2, envs[0].height - 2))


def stochasticity(action, pbt):
    chosen_proba = 1 - pbt
    random_proba = pbt / 3
    proba_vector = [random_proba] * 4
    proba_vector[action] = chosen_proba
    proba_vector = np.nan_to_num(proba_vector, nan=0.0)
    return proba_vector


agent = ACModel(envs[0].observation_space, envs[0].action_space, use_memory=False, use_text=False)
agent.load_state_dict(utils.get_model_state(model_dir, "5m"))
agent.to("cuda")
agent.eval()

adversary = ACModel_adversary()
adversary.load_state_dict(utils.get_adversary_state(model_dir, "5m"))
adversary.to("cuda")
adversary.eval()

while log_done_counter < args.episodes:

    agent_pos, budgets = env._get_infos()

    dist, value = agent(agent_pos)
    init_action = dist.sample()

    state_dist = torch.tensor(
        [stochasticity(act, float(args.stochasticity)) for act in init_action],
        device=agent_pos.device,
        dtype=agent_pos.dtype,
    )
    input_adversary = torch.hstack([agent_pos, state_dist, budgets])
    adv_dist, _ = adversary(input_adversary, state_dist, budgets)

    action = adv_dist.sample()

    a = init_action.cpu().numpy()
    b = action.cpu().numpy()
    liste_differents = np.where(a != b)[0]
    positions = agent_pos.cpu().numpy().astype(int)
    for idx, ob in enumerate(positions):
        counter_map[ob[0] - 1, ob[1] - 1] += 1
        if idx in liste_differents:
            perturbation_map[ob[0] - 1, ob[1] - 1] += 1

    obss, rewards, dones, _ = env.step(
        action.cpu().numpy(), adv_dist.probs.cpu().detach(), state_dist.cpu().numpy(), None, None
    )

    log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
    log_episode_num_frames += torch.ones(args.procs, device=device)

    for i, done in enumerate(dones):
        if done:
            log_done_counter += 1
            logs["return_per_episode"].append(log_episode_return[i].item())
            logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

    mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
    log_episode_return *= mask
    log_episode_num_frames *= mask

    if log_done_counter % 1000 == 0:
        print(log_done_counter)

end_time = time.time()


# get trajectory heat-maps
total = np.divide(perturbation_map, counter_map, out=np.zeros_like(perturbation_map), where=counter_map != 0)

with gzip.open(
    os.path.join(
        "./results", "figure2_{}_{}_{}_{}.pkl.gz".format(args.model, args.stochasticity, args.type, args.episodes)
    ),
    "wb",
) as f:
    pkl.dump(total, f)
    pkl.dump(perturbation_map.T, f)
    pkl.dump(counter_map.T, f)
print("SAVED")

# Print logs

num_frames = sum(logs["num_frames_per_episode"])
fps = num_frames / (end_time - start_time)
duration = int(end_time - start_time)
return_per_episode = utils.synthesize(logs["return_per_episode"])
num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

print(
    "F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}".format(
        num_frames, fps, duration, *return_per_episode.values(), *num_frames_per_episode.values()
    )
)

# Print worst episodes

n = args.worst_episodes_to_show
if n > 0:
    print("\n{} worst episodes:".format(n))

    indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
    for i in indexes[:n]:
        print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
