import gym
from multiprocessing import Process, Pipe

from gym_minigrid.minigrid import *
from gym_minigrid.envs import DistShiftEnv
from gym_minigrid.envs import MiniGridEnv
import numpy as np
import torch
import random


def make_env(budget, cst, seed=None, **kwargs):

    env = StochasticDistShiftEnv(adversary_budget=float(budget), cost=float(cst))
    env.seed(seed)
    return env


class StochasticDistShiftEnv(DistShiftEnv):
    """
    Stochastic Distributional shift environment.
    """

    class RestrictedActions(IntEnum):
        # Turn left, turn right, move forward
        up = 0
        down = 1
        right = 2
        left = 3

    ACTION_DIR_VEC = {
        RestrictedActions.up: np.array([0, -1]),
        RestrictedActions.down: np.array([0, 1]),
        RestrictedActions.right: np.array([1, 0]),
        RestrictedActions.left: np.array([-1, 0]),
    }

    def __init__(
        self,
        cost=0.7,
        width=12,
        height=9,
        agent_start_pos=np.array((1, 1)),
        agent_start_dir=0,
        strip2_row=2,
        adversary_budget=1.0,
    ):

        self.adversary_budget = adversary_budget

        self.remaining_budget = adversary_budget
        self.step_cost = cost

        super().__init__(
            width=width,
            height=height,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            strip2_row=strip2_row,
        )

        self.actions = StochasticDistShiftEnv.RestrictedActions
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def _is_oob(self, position):
        ### checks if the next position is out-of-box
        return position[0] < 1 or position[1] < 1 or position[0] > self.width - 2 or position[1] > self.height - 2

    def step(self, action, final_dist, state_dist):

        """
        update the environment given the policy action :
        action: integer between 0 and 3 to characterize respectively (up, down, right, left).
        """

        reward = -self.step_cost
        done = False
        self.step_count += 1

        next_pos = self.agent_pos + self.ACTION_DIR_VEC[action]

        if self._is_oob(next_pos):  # if out of the box
            next_pos = self.agent_pos

        next_cell = self.grid.get(*next_pos)

        if next_cell == None or next_cell.can_overlap():  # move in the chosen direction
            self.agent_pos = next_pos

        if next_cell != None and next_cell.type == "goal":
            done = True
            reward = 20

        if next_cell != None and next_cell.type == "lava":
            done = True
            reward = -20

        reward = reward / 20

        ## update the remianing budget:
        perturbation = final_dist[action] / state_dist[action]

        self.remaining_budget = self.remaining_budget / perturbation
        if self.remaining_budget > 0.99 and self.remaining_budget < 1:
            self.remaining_budget = np.round(self.remaining_budget)

        obs = self.gen_obs()

        return obs, reward, done, {}

    def _get_infos(self):
        return self.agent_pos, self.remaining_budget

    def reset(self):

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0
        self.remaining_budget = self.adversary_budget

        # Return first observation
        obs = self.gen_obs()
        return obs

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        assert hasattr(self, "mission"), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {"image": image, "direction": self.agent_dir, "mission": self.mission, "position": self.agent_pos}

        return obs
