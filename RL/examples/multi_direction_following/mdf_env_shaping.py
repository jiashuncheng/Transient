import os
import sys
from math import tanh
from random import randint

sys.path.insert(0,'{}'.format(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))))
import gymnasium as gym
import memory_gym
import numpy as np
from gymnasium import spaces
from examples.multi_direction_following.experiments.env import GridMortarMayhemEnv

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

class MMShapingWrapper(gym.Env):
    def __init__(self, env, reset_params=None, realtime_mode = False, render_mode=None, cfg=None):
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, 
                                          "num-seeds": 100000,
                                          "agent_scale": 0.25,
                                          "arena_size": 5,
                                          "allowed_commands": 5,
                                          "command_count": [cfg.command_count],
                                          "explosion_duration": [2],
                                          "explosion_delay": [6],
                                          "reward_command_failure": 0.0,
                                          "reward_command_success": 0.1,
                                          "reward_episode_success": 0.0,
                                          "visual_feedback": True,
                                          }
        else:
            self._default_reset_params = reset_params

        if render_mode == "human": 
            render_mode = None
        # self._env = gym.make(env, disable_env_checker = True, render_mode = render_mode)
        self._env = GridMortarMayhemEnv(render_mode = render_mode)

        self._realtime_mode = realtime_mode

        self._observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (self._env.observation_space.shape[2], self._env.observation_space.shape[1], self._env.observation_space.shape[0]),
                dtype = np.float32)
    @property
    def observation_space(self):
        """Returns the shape of the observation space of the agent."""
        return self._observation_space
    
    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space
    
    def reset(self, **kwargs):
        reset_params = self._default_reset_params

        # Sample seed
        self._seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)

        # Remove reset params that are not processed directly by the environment
        options = reset_params.copy()
        options.pop("start-seed", None)
        options.pop("num-seeds", None)
        options.pop("seed", None)

        # Reset the environment to retrieve the initial observation
        vis_obs, info = self._env.reset(seed=self._seed, options=options)
        vis_obs = np.swapaxes(vis_obs, 0, 2)
        vis_obs = np.swapaxes(vis_obs, 2, 1)
        return vis_obs, info

    def step(self, action):
        if isinstance(action, list):
            if len(action) == 1:
                action = action[0]
        # print(action)
        vis_obs, reward, done, truncation, info = self._env.step(action)
        vis_obs = np.swapaxes(vis_obs, 0, 2)
        vis_obs = np.swapaxes(vis_obs, 2, 1)
        # if done:
        #     print(info['length'])
        #     sys.exit()

        return vis_obs, reward, done, truncation, info

    def render(self):
        """Renders the environment."""
        self._env.render()

    def close(self):
        """Shuts down the environment."""
        self._env.close()