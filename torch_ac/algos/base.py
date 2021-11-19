from abc import ABC, abstractmethod
import torch

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
import numpy as np

from torch.distributions.categorical import Categorical


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(
        self,
        envs,
        acmodel,
        acmodel_adversary,
        pbt,
        device,
        num_frames_per_proc,
        discount,
        lr,
        gae_lambda,
        entropy_coef,
        value_loss_coef,
        max_grad_norm,
        recurrence,
        preprocess_obss,
        reshape_reward,
    ):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        self.env = ParallelEnv(envs)

        self.acmodel = acmodel
        self.acmodel_adversary = acmodel_adversary

        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.pbt = float(pbt)

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        self.acmodel_adversary.to(self.device)
        self.acmodel_adversary.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])
        self.inputs_adversary = [None] * (shape[0])
        self.budgets = [None] * (shape[0])

        # if self.acmodel.recurrent:
        #     self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
        #     self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)

        self.values = torch.zeros(*shape, device=self.device)
        self.adv_values = torch.zeros(*shape, device=self.device)

        self.rewards = torch.zeros(*shape, device=self.device)
        self.adv_rewards = torch.zeros(*shape, device=self.device)

        self.advantages = torch.zeros(*shape, device=self.device)
        self.adv_advantages = torch.zeros(*shape, device=self.device)

        self.log_probs = torch.zeros(*shape, device=self.device)
        self.adv_log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values
        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

        self.total = 0
        self.avg = 0

        self.perturbation_map = np.zeros((envs[0].width - 2, envs[0].height - 2))
        self.counter_map = np.zeros((envs[0].width - 2, envs[0].height - 2))

    def stochasticity(self, action, pbt):

        chosen_proba = 1 - pbt
        random_proba = pbt / 3
        proba_vector = [random_proba] * 4
        proba_vector[action] = chosen_proba
        proba_vector = np.nan_to_num(proba_vector, nan=0.0)

        return proba_vector

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        avg = []
        total = []

        batch_perturbation_map = np.zeros(self.perturbation_map.shape)
        batch_counter_map = np.zeros(self.counter_map.shape)

        for i in range(self.num_frames_per_proc):  #
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            agent_pos, budgets = self.env._get_infos()

            with torch.no_grad():

                dist, value = self.acmodel(agent_pos)

                init_action = dist.sample()

                state_dist = torch.tensor(
                    [self.stochasticity(act, self.pbt) for act in init_action],
                    device=agent_pos.device,
                    dtype=agent_pos.dtype,
                )

                input_adversary = torch.hstack([agent_pos, state_dist, budgets])

                adv_dist, adv_value = self.acmodel_adversary(input_adversary, state_dist, budgets)

                action = adv_dist.sample()

                # self.total = self.total +  len(action)
                # self.avg = self.avg + [  True if init_action[i] == action[i] else False for i in range(len(action)) ].count(False)

            a = init_action.cpu().numpy()
            b = action.cpu().numpy()
            liste_differents = np.where(a != b)[0]
            positions = agent_pos.cpu().numpy().astype(int)
            for idx, ob in enumerate(positions):
                batch_counter_map[ob[0] - 1, ob[1] - 1] += 1
                if idx in liste_differents:
                    batch_perturbation_map[ob[0] - 1, ob[1] - 1] += 1

            obs, reward, done, _ = self.env.step(
                action.cpu().numpy(), adv_dist.probs.cpu().numpy(), state_dist.cpu().numpy(), None, None
            )

            self.counter_map = self.counter_map + batch_counter_map
            self.perturbation_map = self.perturbation_map + batch_perturbation_map

            self.budgets[i] = budgets

            self.obss[i] = self.obs
            self.obs = obs

            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)

            self.actions[i] = action
            self.inputs_adversary[i] = input_adversary

            self.values[i] = value
            self.adv_values[i] = adv_value

            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor(
                    [
                        self.reshape_reward(obs_, action_, reward_, done_)
                        for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                    ],
                    device=self.device,
                )
                self.adv_rewards[i] = -1 * self.rewards[i]

            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
                self.adv_rewards[i] = -1 * self.rewards[i]

            self.log_probs[i] = dist.log_prob(action)
            self.adv_log_probs[i] = adv_dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # print( 'mean: ' + str( sum(avg) / len(avg) ) +  ' max: ' + str( max(avg) ) + ' min: '+ str( min(avg) )  )

        # Add advantage and return to experiences for both the agent and the adversary
        # preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        next_agent_pos, next_budgets = self.env._get_infos()
        with torch.no_grad():
            next_dist, next_value = self.acmodel(next_agent_pos)
            next_action = next_dist.sample()
            next_state_dist = torch.tensor([self.stochasticity(act, self.pbt) for act in next_action], device="cuda")

            next_input_adversary = torch.hstack([next_state_dist, next_agent_pos, next_budgets]).float()
            _, next_adv_value = self.acmodel_adversary(next_input_adversary, next_state_dist, next_budgets)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i + 1] if i < self.num_frames_per_proc - 1 else self.mask

            ### for the agent
            next_value = self.values[i + 1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i + 1] if i < self.num_frames_per_proc - 1 else 0
            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

            ### for the adversary
            next_adv_value = self.adv_values[i + 1] if i < self.num_frames_per_proc - 1 else next_adv_value
            next_adv_advantage = self.adv_advantages[i + 1] if i < self.num_frames_per_proc - 1 else 0
            adv_delta = self.adv_rewards[i] + self.discount * next_adv_value * next_mask - self.adv_values[i]
            self.adv_advantages[i] = adv_delta + self.discount * self.gae_lambda * next_adv_advantage * next_mask

        exps = DictList()
        exps.obs = [self.obss[i][j] for j in range(self.num_procs) for i in range(self.num_frames_per_proc)]

        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.adv_value = self.adv_values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.adv_reward = self.adv_rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.adv_advantage = self.adv_advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.adv_returnn = exps.adv_value + exps.adv_advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        exps.adv_log_prob = self.adv_log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        exps.inputs_adversary = torch.tensor(
            [
                self.inputs_adversary[i][j].cpu().numpy()
                for j in range(self.num_procs)
                for i in range(self.num_frames_per_proc)
            ],
            device=self.device,
        )
        exps.budgets = torch.tensor(
            [self.budgets[i][j] for j in range(self.num_procs) for i in range(self.num_frames_per_proc)],
            device=self.device,
        )

        # Log some values
        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs :]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs :]
        self.log_num_frames = self.log_num_frames[-self.num_procs :]

        return exps, logs, batch_counter_map, batch_perturbation_map

    @abstractmethod
    def update_parameters(self):
        pass
