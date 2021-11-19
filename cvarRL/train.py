import argparse
import time
import datetime
import torch
import torch_ac
import tensorboardX
import sys
import pickle as pkl
import gzip
import os

import utils
from model import ACModel, ACModel_adversary
import numpy as np

np.set_printoptions(suppress=True)


# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--algo", required=True, help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--budget", required=True, help="budget for the adversary")
parser.add_argument("--stochasticity", required=True, help="stochasticity of the actions taken")
parser.add_argument("--cost", required=True, help="cost for each moove")

parser.add_argument("--model", default=None, help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1, help="number of updates between two logs (default: 1)")
parser.add_argument(
    "--save-interval", type=int, default=10, help="number of updates between two saves (default: 10, 0 means no saving)"
)
parser.add_argument("--procs", type=int, default=16, help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10 ** 7, help="number of frames of training (default: 1e7)")

## Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4, help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256, help="batch size for PPO (default: 256)")
parser.add_argument(
    "--frames-per-proc",
    type=int,
    default=None,
    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)",
)
parser.add_argument("--discount", type=float, default=0.99, help="discount factor (default: 0.99)")
parser.add_argument("--max_lr", type=float, default=0.001, help="Maximum learning rate (default: 0.001)")
parser.add_argument("--min_lr", type=float, default=0.00001, help="Minimum learning rate (default: 0.00001)")
parser.add_argument(
    "--gae-lambda", type=float, default=0.95, help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)"
)
parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5, help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5, help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8, help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99, help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2, help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument(
    "--recurrence",
    type=int,
    default=1,
    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.",
)
parser.add_argument("--text", action="store_true", default=False, help="add a GRU to the model to handle text input")
parser.add_argument("--save_interval", type=int, default=10)

args = parser.parse_args()

args.mem = args.recurrence > 1

# Set run dir

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{'stochasticdistshift'}_{args.algo}_seed{args.seed}_{date}"

model_name = args.model or default_model_name

name = model_name  # +'_'+ str(args.budget) +'_' + str(args.stochasticity) + '_' + str(args.reward)

model_dir = utils.get_model_dir(name)


# Load loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)


# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

# Load environments

envs = []
for i in range(args.procs):
    envs.append(utils.make_env(args.budget, args.cost, args.seed + 10000 * i))
txt_logger.info("Environments loaded\n")

# Load training status

try:
    status = utils.get_status(model_dir, "2m")
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor

obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
txt_logger.info("Observations preprocessor loaded")

# Load model

model_to_load = None
adv_model_to_load = None

acmodel = ACModel(obs_space, envs[0].action_space)
# if model_to_load:
#    acmodel.load_state_dict( utils.get_status( utils.get_model_dir( model_to_load ) )['model_state'] )
acmodel.to(device)

txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(acmodel))

acmodel_adversary = ACModel_adversary()
# if adv_model_to_load:
#    acmodel_adversary.load_state_dict( utils.get_status( utils.get_model_dir( model_to_load ) ) ['adv_model_state'] )
acmodel_adversary.to(device)

txt_logger.info("Model adversary loaded\n")
txt_logger.info("{}\n".format(acmodel_adversary))


algo = torch_ac.PPOAlgo(
    envs,
    acmodel,
    acmodel_adversary,
    args.stochasticity,
    device,
    args.frames_per_proc,
    args.discount,
    args.max_lr,
    args.gae_lambda,
    args.entropy_coef,
    args.value_loss_coef,
    args.max_grad_norm,
    args.recurrence,
    args.optim_eps,
    args.clip_eps,
    args.epochs,
    args.batch_size,
    preprocess_obss,
)

adversary_algo = torch_ac.AdversaryPPOAlgo(
    envs,
    acmodel,
    acmodel_adversary,
    args.stochasticity,
    device,
    args.frames_per_proc,
    args.discount,
    args.max_lr,
    args.gae_lambda,
    args.entropy_coef,
    args.value_loss_coef,
    args.max_grad_norm,
    args.recurrence,
    args.optim_eps,
    args.clip_eps,
    args.epochs,
    args.batch_size,
    preprocess_obss,
)


def get_lr_for_frames(max_lr, min_lr, n_frames):
    min_adjust = min_lr / max_lr

    return lambda frames: max((n_frames - frames) / n_frames, min_adjust)


lr_lambda = get_lr_for_frames(args.max_lr, args.min_lr, args.frames)
scheduler = torch.optim.lr_scheduler.LambdaLR(algo.optimizer, lr_lambda)
adversary_scheduler = torch.optim.lr_scheduler.LambdaLR(adversary_algo.optimizer, lr_lambda)

# if model_to_load:
#     algo.optimizer.load_state_dict( utils.get_status( utils.get_model_dir( model_to_load ) )["optimizer_state"] )
# if adv_model_to_load:
#     adversary_algo.optimizer.load_state_dict( utils.get_status( utils.get_model_dir( model_to_load ))['adv_optimizer_state'] )

txt_logger.info("Optimizer loaded\n")


# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

logs2 = None

while num_frames < args.frames:

    update_start_time = time.time()
    exps, logs1, batch_counter_map, batch_perturbation_map = algo.collect_experiences()

    batch = np.divide(
        batch_perturbation_map,
        batch_counter_map,
        out=np.zeros_like(batch_perturbation_map),
        where=batch_counter_map != 0,
    )
    total = np.divide(
        algo.perturbation_map, algo.counter_map, out=np.zeros_like(algo.perturbation_map), where=algo.counter_map != 0
    )

    # print(np.round(batch.T,2))
    # print()
    # print(np.round(total.T,2))
    # print()

    if update % 3 == 0:
        logs2 = algo.update_parameters(exps)
        scheduler.last_epoch = num_frames  # Needed to hack last_epoch to do what I wanted
        scheduler.step()

    else:
        logs2 = adversary_algo.update_parameters(exps)
        adversary_scheduler.last_epoch = num_frames
        adversary_scheduler.step()

    logs = {**logs1, **logs2}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs

    if update % args.log_interval == 0:

        # print(  algo.avg / algo.total )

        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        txt_logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}".format(
                *data
            )
        )

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, num_frames)

    # Save status

    if args.save_interval > 0 and update % args.save_interval == 0:

        status = {
            "num_frames": num_frames,
            "update": update,
            "model_state": acmodel.state_dict(),
            "optimizer_state": algo.optimizer.state_dict(),
            "adv_model_state": acmodel_adversary.state_dict(),
            "adv_optimizer_state": adversary_algo.optimizer.state_dict(),
        }
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab

        if num_frames < int(args.frames / 2):
            utils.save_status(status, model_dir, "2m")
        else:
            utils.save_status(status, model_dir, "5m")

        with gzip.open(os.path.join("{}_maps.pkl.gz".format(model_dir)), "wb") as f:
            pkl.dump(total, f)
            pkl.dump(algo.perturbation_map.T, f)
            pkl.dump(algo.counter_map.T, f)

        txt_logger.info("Status saved")
