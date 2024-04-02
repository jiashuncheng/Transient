from typing import Optional

import gymnasium as gym

from sample_factory.utils.utils import is_module_available
from examples.multi_direction_following.mdf_env_shaping import MMShapingWrapper


def mm_available():
    return is_module_available("MemoryGym")


class MMSpec:
    def __init__(self, name, env_id):
        self.name = name
        self.env_id = env_id


MM_ENVS = [
    MMSpec("MortarMayhemB", "MortarMayhemB-Grid-v0"),
    MMSpec("MortarMayhem", "MortarMayhem-Grid-v0"),
]


def mm_env_by_name(name):
    for cfg in MM_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown MortarMayhem env")


def make_mdf_env(env_name, _cfg, _env_config, render_mode: Optional[str] = None, **kwargs):
    mm_spec = mm_env_by_name(env_name)
    env = MMShapingWrapper(mm_spec.env_id, render_mode=render_mode, cfg=_cfg)
    return env
