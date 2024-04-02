import os
import sys
from math import tanh
from random import randint
import time

sys.path.insert(0,'{}'.format(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))))
import gymnasium as gym
import memory_gym
import numpy as np
from gymnasium import spaces
from examples.watermaze2d.experiments.watermaze2d import Watermaze2dEnv
from examples.watermaze2d.experiments.wrappers import *

class Watermaze2dShapingWrapper(gym.Env):
    def __init__(self, env, mode='train', seed = None, _cfg=None):
        self.size = _cfg.size
        self.agent_view_size = _cfg.agent_view_size

        self._env = Watermaze2dEnv(seed, mode, size=self.size, agent_view_size=self.agent_view_size, cfg=_cfg)
        self.seed = seed
        # Decrease the agent's view size to raise the agent's memory challenge
        # On MiniGrid-Memory-S7-v0, the default view size is too large to actually demand a recurrent policy.
        # self._env = RGBImgPartialObsWrapper(self._env, tile_size=28)
        # self._env = ImgObsWrapper(self._env)
        # self._env = ViewSizeWrapper(self._env, self.agent_view_size) # 部分可观测
        self._env = RGBImgObsWrapper(self._env, tile_size=8)
        # self._env = ImgObsWrapper(self._env)
        # self._observation_space = spaces.Box(
        #         low = 0,
        #         high = 1.0,
        #         shape = (3, self.agent_view_size*28, self.agent_view_size*28),
        #         dtype = np.float32) # 部分可观测
        self._observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (3, self.size*8, self.size*8),
                dtype = np.float32)
    @property
    def observation_space(self):
        """Returns the shape of the observation space of the agent."""
        return self._observation_space
    
    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return spaces.Discrete(4)
    
    def reset(self, **kwargs):
        self._rewards = []
        obs = self._env.reset()
        # obs = self._env.get_obs_render(obs["image"], tile_size=28).astype(np.float32) / 255. # 部分可观测
        obs = self._env.observation(obs)['image'].astype(np.float32) / 255. # 全部可观测
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        info = None
        return obs, info

    def step(self, action):
        if isinstance(action, list):
            if len(action) == 1:
                action = action[0]
        obs, reward, done, info = self._env.step(action)
        self._rewards.append(reward)
        # obs = self._env.get_obs_render(obs["image"], tile_size=28).astype(np.float32) / 255. # 部分可观测
        obs = self._env.observation(obs)['image'].astype(np.float32) / 255. # 全部可观测
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        truncation = False
        info['agent_pos'] = self._env.unwrapped.agent_pos
        return obs, reward, done, truncation, info
    
    def render(self):
        img = self._env.render()
        time.sleep(0.1)
        return img

    def close(self):
        self._env.close()
