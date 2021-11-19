from multiprocessing import Process, Pipe
import gym
import torch
import numpy as np


def worker(conn, env):

    while True:
        cmd, data = conn.recv()

        if cmd == "step":
            obs, reward, done, info = env.step(data[0], data[1], data[2])
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))

        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)

        elif cmd == "_get_infos":
            agent_pos, remaining_budget = env._get_infos()
            conn.send((agent_pos, remaining_budget))

        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs, device="cuda"):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.device = device

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions, final_dists, state_dists, agent_pos, init_action):

        for local, action, final_dist, state_dist in zip(self.locals, actions[1:], final_dists[1:], state_dists[1:]):
            local.send(("step", (action, final_dist, state_dist)))

        obs, reward, done, info = self.envs[0].step(actions[0], final_dists[0], state_dists[0])

        if done:
            obs = self.envs[0].reset()

        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])

        return results

    def _get_infos(self):

        for local in self.locals:
            local.send(("_get_infos", None))

        result = [self.envs[0]._get_infos()] + [local.recv() for local in self.locals]
        agent_pos, remaining_budget = [res[0] for res in result], [[res[1]] for res in result]

        agent_pos = torch.tensor(agent_pos, device=self.device, dtype=torch.float)
        remaining_budget = torch.tensor(remaining_budget, device=self.device, dtype=torch.float)

        return agent_pos, remaining_budget

    def render(self):
        raise NotImplementedError
