from typing import Optional

import gymnasium as gym

from sample_factory.utils.utils import is_module_available
from examples.direction_following.df_env_shaping import DFShapingWrapper


def df_available():
    return is_module_available("MemoryGym")


class DFSpec:
    def __init__(self, name, env_id):
        self.name = name
        self.env_id = env_id


DF_ENVS = [
    DFSpec("MortarMayhemOA", "MortarMayhem-Grid-v0"),
]


def df_env_by_name(name):
    for cfg in DF_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown MortarMayhem env")


def make_df_env(env_name, _cfg, _env_config, render_mode: Optional[str] = None, **kwargs):
    df_spec = df_env_by_name(env_name)
    env = DFShapingWrapper(df_spec.env_id, render_mode=render_mode, cfg=_cfg)
    return env
